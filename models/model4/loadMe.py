import math
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
import os, sys


myPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(myPath)
import sinogap_module as sg


#sg.set_seed(7)
sg.TCfg = sg.TCfgClass(
     exec = 3
    ,nofEpochs = None
    ,latentDim = 64
    ,batchSize = 1
    ,batchSplit = 1
    ,labelSmoothFac = 0.1 # For Fake labels (or set to 0.0 for no smoothing).
    ,learningRateD = 1
    ,learningRateG = 1
    ,dataDir=""
)
TCfg = sg.TCfg


sg.DCfg = sg.DCfgClass(16,False)
DCfg = sg.DCfg

sg.brickMasks = sg.createBrickMasks()

#os.environ["CTAS_MMAP_PATH"] = "/mnt/ssdData/:home/imbl/"



class StripeGenerator2(sg.SubGeneratorTemplate) :
    def __init__(self):
        super().__init__(2, False, batchNorm=True)
        self.lowResGenerator = None
        self.baseChannels = 32
        self.baseChannelsOther = 32

        self.headEncoders =  nn.ModuleList([
            * self.groundFloor(2, 3, 2),
            * self.encFloor(2, 2, 3, 2),
            * self.encFloor(4, 2, 3, 2),
            ])

        self.tailEncoders =  nn.ModuleList([
            * self.encFloor(8, 1/2, 3, (2,1), other = 0),
            * self.encFloor(4, 1,   3, (2,1), other = 0),
            * self.encFloor(4, 1,   3, (2,1), other = 0),
            * self.encFloor(4, 1,   3, (2,1), other = 0),
            ])

        self.fcLink = self.createAttic( self.headEncoders + self.tailEncoders )

        self.tailDecoders = nn.ModuleList([
            * self.decFloor(4, 1,   3, (2,1), other = 0),
            * self.decFloor(4, 1,   3, (2,1), other = 0),
            * self.decFloor(4, 1,   3, (2,1), other = 0),
            * self.decFloor(8, 1/2, 3, (2,1), other = 0),
            ] )

        self.headDecoders = nn.ModuleList([
            * self.decFloor(4, 2, 3, stride=2),
            * self.decFloor(2, 2, 3, stride=2),
            * self.decFloor(1, 2, 3, stride=2, norm=False),
            ])

        self.lastTouch = self.createBasement(1)
        self.amplitude = nn.Parameter(torch.tensor(4.0))


class DBricksGenerator2(sg.SubGeneratorTemplate) :
    def __init__(self):
        super().__init__(2, True, batchNorm=True)
        self.lowResGenerator = None
        self.baseChannels = 32
        self.baseChannelsOther = 32
        self.encoders =  nn.ModuleList([
            * self.groundFloor(2, 3, 2),
            * self.encFloor(2, 2, 3, 2),
            * self.encFloor(4, 2, 3, 2),
            ])
        self.fcLink = self.createAttic(self.encoders)
        self.decoders = nn.ModuleList([
            * self.decFloor(4, 2, 3, stride=2),
            * self.decFloor(2, 2, 3, stride=2),
            * self.decFloor(1, 2, 3, stride=2, norm=False),
            ])
        self.lastTouch = self.createBasement(1)
        self.amplitude = nn.Parameter(torch.tensor(4.0))


class Generator2(sg.GeneratorTemplate) :
    def __init__(self):
        super().__init__(2, batchNorm=True)
        self.lowResGenerator = None
        self.brickGenerator = DBricksGenerator2()
        self.stripeGenerator = StripeGenerator2()


class StripeGenerator4(sg.SubGeneratorTemplate) :
    def __init__(self):
        super().__init__(4, False, batchNorm=True, inChannels=2)
        self.baseChannels = 32
        self.baseChannelsOther = 32

        self.headEncoders =  nn.ModuleList([
            * self.groundFloor(2,3,2),
            * self.encFloor(2, 2,3,2),
            * self.encFloor(4, 2,3,2),
            * self.encFloor(8, 2,3,2),
            ])
        self.tailEncoders =  nn.ModuleList([
            * self.encFloor(16, 1/2,3,(2,1), other = 0),
            * self.encFloor( 8, 1  ,3,(2,1), other = 0),
            * self.encFloor( 8, 1  ,3,(2,1), other = 0),
            * self.encFloor( 8, 1  ,3,(2,1), other = 0),
            ])
        self.fcLink = self.createAttic( self.headEncoders + self.tailEncoders )

        self.tailDecoders = nn.ModuleList([
            * self.decFloor( 8, 1  ,3,(2,1), other = 0),
            * self.decFloor( 8, 1  ,3,(2,1), other = 0),
            * self.decFloor( 8, 1  ,3,(2,1), other = 0),
            * self.decFloor(16, 1/2,3,(2,1), other = 0),
            ] )

        self.headDecoders = nn.ModuleList([
            * self.decFloor(8, 2, 3, stride=2),
            * self.decFloor(4, 2, 3, stride=2),
            * self.decFloor(2, 2, 3, stride=2),
            * self.decFloor(1, 2, 3, stride=2, norm=False),
            ])

        self.lastTouch = self.createBasement(1)
        self.amplitude = nn.Parameter(torch.tensor(4.0))



class DBricksGenerator4(sg.SubGeneratorTemplate) :
    def __init__(self):
        super().__init__(4, True, batchNorm=True, inChannels=2)
        self.baseChannels = 32
        self.baseChannelsOther = 32
        self.encoders =  nn.ModuleList([
            * self.groundFloor(2,3,2),
            * self.encFloor(2, 2,3,2),
            * self.encFloor(4, 2,3,2),
            * self.encFloor(8, 2,3,2),
            ])
        self.fcLink = self.createAttic(self.encoders)
        self.decoders = nn.ModuleList([
            * self.decFloor(8, 2, 3, stride=2),
            * self.decFloor(4, 2, 3, stride=2),
            * self.decFloor(2, 2, 3, stride=2),
            * self.decFloor(1, 2, 3, stride=2, norm=False),
            ])
        self.lastTouch = self.createBasement(1)
        self.amplitude = nn.Parameter(torch.tensor(4.0))


class Generator4(sg.GeneratorTemplate) :
    def __init__(self):
        super().__init__(4, batchNorm=True)
        self.lowResGenerator = Generator2()
        self.brickGenerator = DBricksGenerator4()
        self.stripeGenerator = StripeGenerator4()




class StripeGenerator8(sg.SubGeneratorTemplate) :
    def __init__(self):
        super().__init__(8, False, batchNorm=True, inChannels=2)
        self.baseChannels = 32
        self.baseChannelsOther = 32

        self.headEncoders =  nn.ModuleList([
            * self.groundFloor( 2, 3, 2),
            * self.encFloor( 2, 2, 3, 2),
            * self.encFloor( 4, 2, 3, 2),
            * self.encFloor( 8, 2, 3, 2),
            * self.encFloor(16, 2, 3, 2),
            ])
        self.tailEncoders =  nn.ModuleList([
            * self.encFloor(32, 1/2, 3, (2,1), other = 0),
            * self.encFloor(16, 1  , 3, (2,1), other = 0),
            * self.encFloor(16, 1  , 3, (2,1), other = 0),
            * self.encFloor(16, 1  , 3, (2,1), other = 0),
            ])
        self.fcLink = self.createAttic( self.headEncoders + self.tailEncoders )

        self.tailDecoders = nn.ModuleList([
            * self.decFloor(16, 1  , 3, (2,1), other = 0),
            * self.decFloor(16, 1  , 3, (2,1), other = 0),
            * self.decFloor(16, 1  , 3, (2,1), other = 0),
            * self.decFloor(32, 1/2, 3, (2,1), other = 0),
            ] )

        self.headDecoders = nn.ModuleList([
            * self.decFloor(16, 2, 3, stride=2),
            * self.decFloor( 8, 2, 3, stride=2),
            * self.decFloor( 4, 2, 3, stride=2),
            * self.decFloor( 2, 2, 3, stride=2),
            * self.decFloor( 1, 2, 3, stride=2, norm=False),
            ])

        self.lastTouch = self.createBasement(1)
        self.amplitude = nn.Parameter(torch.tensor(4.0))



class DBricksGenerator8(sg.SubGeneratorTemplate) :
    def __init__(self):
        super().__init__(8, True, batchNorm=True, inChannels=2)
        self.baseChannels = 32
        self.baseChannelsOther = 32
        self.encoders =  nn.ModuleList([
            * self.groundFloor( 2, 3, 2),
            * self.encFloor( 2, 2, 3, 2),
            * self.encFloor( 4, 2, 3, 2),
            * self.encFloor( 8, 2, 3, 2),
            * self.encFloor(16, 2, 3, 2),
            ])
        self.fcLink = self.createAttic(self.encoders)
        self.decoders = nn.ModuleList([
            * self.decFloor(16, 2, 3, stride=2),
            * self.decFloor( 8, 2, 3, stride=2),
            * self.decFloor( 4, 2, 3, stride=2),
            * self.decFloor( 2, 2, 3, stride=2),
            * self.decFloor( 1, 2, 3, stride=2, norm=False),
            ])
        self.lastTouch = self.createBasement(1)
        self.amplitude = nn.Parameter(torch.tensor(4.0))


class Generator8(sg.GeneratorTemplate) :
    def __init__(self):
        super().__init__(4, batchNorm=True)
        self.lowResGenerator = Generator4()
        self.brickGenerator = DBricksGenerator8()
        self.stripeGenerator = StripeGenerator8()


class StripeGenerator16(sg.SubGeneratorTemplate) :
    def __init__(self):
        super().__init__(16, False, batchNorm=True, inChannels=2)
        self.baseChannels = 32
        self.baseChannelsOther = 32

        self.headEncoders =  nn.ModuleList([
            * self.groundFloor( 2, 3, 2),
            * self.encFloor( 2, 2, 3, 2),
            * self.encFloor( 4, 2, 3, 2),
            * self.encFloor( 8, 2, 3, 2),
            * self.encFloor(16, 2, 3, 2),
            * self.encFloor(32, 2, 3, 2),
            ])
        self.tailEncoders =  nn.ModuleList([
            * self.encFloor(64, 1/2, 3, (2,1), other = 0),
            * self.encFloor(32, 1  , 3, (2,1), other = 0),
            * self.encFloor(32, 1  , 3, (2,1), other = 0),
            * self.encFloor(32, 1  , 3, (2,1), other = 0),
            ])
        self.fcLink = self.createAttic( self.headEncoders + self.tailEncoders )

        self.tailDecoders = nn.ModuleList([
            * self.decFloor(32, 1  , 3, (2,1), other = 0),
            * self.decFloor(32, 1  , 3, (2,1), other = 0),
            * self.decFloor(32, 1  , 3, (2,1), other = 0),
            * self.decFloor(64, 1/2, 3, (2,1), other = 0),
            ] )

        self.headDecoders = nn.ModuleList([
            * self.decFloor(32, 2, 3, stride=2),
            * self.decFloor(16, 2, 3, stride=2),
            * self.decFloor( 8, 2, 3, stride=2),
            * self.decFloor( 4, 2, 3, stride=2),
            * self.decFloor( 2, 2, 3, stride=2),
            * self.decFloor( 1, 2, 3, stride=2, norm=False),
            ])

        self.lastTouch = self.createBasement(1)
        self.amplitude = nn.Parameter(torch.tensor(4.0))



class DBricksGenerator16(sg.SubGeneratorTemplate) :
    def __init__(self):
        super().__init__(16, True, batchNorm=True, inChannels=2)
        self.baseChannels = 32
        self.baseChannelsOther = 32
        self.encoders =  nn.ModuleList([
            * self.groundFloor( 2, 3, 2),
            * self.encFloor( 2, 2, 3, 2),
            * self.encFloor( 4, 2, 3, 2),
            * self.encFloor( 8, 2, 3, 2),
            * self.encFloor(16, 2, 3, 2),
            * self.encFloor(32, 2, 3, 2),
            ])
        self.fcLink = self.createAttic(self.encoders)
        self.decoders = nn.ModuleList([
            * self.decFloor(32, 2, 3, stride=2),
            * self.decFloor(16, 2, 3, stride=2),
            * self.decFloor( 8, 2, 3, stride=2),
            * self.decFloor( 4, 2, 3, stride=2),
            * self.decFloor( 2, 2, 3, stride=2),
            * self.decFloor( 1, 2, 3, stride=2, norm=False),
            ])
        self.lastTouch = self.createBasement(1)
        self.amplitude = nn.Parameter(torch.tensor(4.0))


class Generator16(sg.GeneratorTemplate) :
    def __init__(self):
        super().__init__(16, batchNorm=True)
        self.lowResGenerator = Generator8()
        self.brickGenerator = DBricksGenerator16()
        self.stripeGenerator = StripeGenerator16()

        sg.load_model(self, model_path=os.path.join(myPath, "model_gen.pt" ) )


def createGenerator(device=sg.TCfg.device):
    sg.TCfg.device = device
    sg.generator = Generator16()
    sg.generator.to(sg.TCfg.device)
    sg.generator.requires_grad_(False)
    sg.generator.eval()

    return sg.generator

