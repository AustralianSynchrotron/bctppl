#!/usr/bin/env python3

import sys
import os
import random
import time
from dataclasses import dataclass
from enum import Enum

import math
import statistics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch import optim
from torchvision import transforms
from torchinfo import summary

import argparse
import h5py
import tifffile
import tqdm


parser = argparse.ArgumentParser(description=
    'Fill sinograms with sinogap NN.')
parser.add_argument('input', type=str, default="",
                    help='Input stack of CT projections to fill.')
parser.add_argument('output', type=str, default="",
                    help='Output filled stack.')
parser.add_argument('-m', '--mask', type=str, default="",
                    help='Mask of the input stack.')
parser.add_argument('-v', '--verbose', action='store_true', default=False,
                    help='Plot results.')
args = parser.parse_args()

device = torch.device('cuda:0')


#%% FUNCS

def fillWheights(seq) :
    for wh in seq :
        if hasattr(wh, 'weight') :
            torch.nn.init.xavier_uniform_(wh.weight)
            #torch.nn.init.zeros_(wh.weight)
            #torch.nn.init.constant_(wh.weight, 0)
            #torch.nn.init.uniform_(wh.weight, a=0.0, b=1.0, generator=None)
            #torch.nn.init.normal_(wh.weight, mean=0.0, std=0.01)


def unsqeeze4dim(tens):
    orgDims = tens.dim()
    if tens.dim() == 2 :
        tens = tens.unsqueeze(0)
    if tens.dim() == 3 :
        tens = tens.unsqueeze(1)
    return tens, orgDims


def squeezeOrg(tens, orgDims):
    if orgDims == tens.dim():
        return tens
    if tens.dim() != 4 or orgDims > 4 or orgDims < 2:
        raise Exception(f"Unexpected dimensions to squeeze: {tens.dim()} {orgDims}.")
    if orgDims < 4 :
        if tens.shape[1] > 1:
            raise Exception(f"Cant squeeze dimension 1 in: {tens.shape}.")
        tens = tens.squeeze(1)
    if orgDims < 3 :
        if tens.shape[0] > 1:
            raise Exception(f"Cant squeeze dimension 0 in: {tens.shape}.")
        tens = tens.squeeze(0)
    return tens


def loadImage(imageName, expectedShape=None) :
    if not imageName:
        return None
    #imdata = imread(imageName).astype(np.float32)
    imdata = tifffile.imread(imageName).astype(np.float32)
    if len(imdata.shape) == 3 :
        imdata = np.mean(imdata[:,:,0:3], 2)
    if not expectedShape is None  and  imdata.shape != expectedShape :
        raise Exception(f"Dimensions of the input image \"{imageName}\" {imdata.shape} "
                        f"do not match expected shape {expectedShape}.")
    return imdata


def set_seed(SEED_VALUE):
    torch.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed_all(SEED_VALUE)
    np.random.seed(SEED_VALUE)

seed = 7
set_seed(seed)

@dataclass(frozen=True)
class TCfg:
    exec = 1
    device: torch.device = device
    latentDim: int = 64

class DCfg:
    gapW = 16
    sinoSh = (5*gapW,5*gapW) # 80x80
    readSh = (80, 80)
    sinoSize = math.prod(sinoSh)
    gapSh = (sinoSh[0],gapW)
    gapSize = math.prod(gapSh)
    gapRngX = np.s_[ sinoSh[1]//2 - gapW//2 : sinoSh[1]//2 + gapW//2 ]
    gapRng = np.s_[ : , gapRngX ]
    disRng = np.s_[ gapW:-gapW , gapRngX ]


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return model


def addToHDF(filename, containername, data) :
    if len(data.shape) == 2 :
        data=np.expand_dims(data, 0)
    if len(data.shape) != 3 :
        raise Exception(f"Not appropriate input array size {data.shape}.")

    with h5py.File(filename,'a') as file :

        if  containername not in file.keys():
            dset = file.create_dataset(containername, data.shape,
                                       maxshape=(None,data.shape[1],data.shape[2]),
                                       dtype='f')
            dset[()] = data
            return

        dset = file[containername]
        csh = dset.shape
        if csh[1] != data.shape[1] or csh[2] != data.shape[2] :
            raise Exception(f"Shape mismatch: input {data.shape}, file {dset.shape}.")
        msh = dset.maxshape
        newLen = csh[0] + data.shape[0]
        if msh[0] is None or msh[0] >= newLen :
            dset.resize(newLen, axis=0)
        else :
            raise Exception(f"Insufficient maximum shape {msh} to add data"
                            f" {data.shape} to current volume {dset.shape}.")
        dset[csh[0]:newLen,...] = data
        file.close()


    return 0


def getInData(inputString):
    sampleHDF = inputString.split(':')
    if len(sampleHDF) != 2 :
        raise Exception(f"String \"{inputString}\" does not represent an HDF5 format \"fileName:container\".")
    try :
        trgH5F =  h5py.File(sampleHDF[0],'r')
    except :
        raise Exception(f"Failed to open HDF file '{sampleHDF[0]}'.")
    if  sampleHDF[1] not in trgH5F.keys():
        raise Exception(f"No dataset '{sampleHDF[1]}' in input file {sampleHDF[0]}.")
    data = trgH5F[sampleHDF[1]]
    if not data.size :
        raise Exception(f"Container \"{inputString}\" is zero size.")
    sh = data.shape
    if len(sh) != 3 :
        raise Exception(f"Dimensions of the container \"{inputString}\" is not 3: {sh}.")
    return data


def getOutData(outputString, shape) :
    if len(shape) == 2 :
        shape = (1,*shape)
    if len(shape) != 3 :
        raise Exception(f"Not appropriate output array size {shape}.")

    sampleHDF = outputString.split(':')
    if len(sampleHDF) != 2 :
        raise Exception(f"String \"{outputString}\" does not represent an HDF5 format \"fileName:container\".")
    try :
        trgH5F =  h5py.File(sampleHDF[0],'w')
    except :
        raise Exception(f"Failed to open HDF file '{sampleHDF[0]}'.")

    if  sampleHDF[1] not in trgH5F.keys():
        dset = trgH5F.create_dataset(sampleHDF[1], shape, dtype='f')
    else :
        dset = trgH5F[sampleHDF[1]]
        csh = dset.shape
        if csh[0] < shape[0] or csh[1] != shape[1] or csh[2] != shape[2] :
            raise Exception(f"Shape mismatch: input {shape}, file {dset.shape}.")
    return dset, trgH5F

#%% MODELS

modelsRoot = os.path.join( os.path.dirname(__file__) , "models" )

class Generator2(nn.Module):

    def __init__(self):
        super(Generator2, self).__init__()

        self.gapW = 2
        self.sinoSh = (5*self.gapW,5*self.gapW) # 10,10
        self.sinoSize = math.prod(self.sinoSh)
        self.gapSh = (self.sinoSh[0],self.gapW)
        self.gapSize = math.prod(self.gapSh)
        self.gapRngX = np.s_[ self.sinoSh[1]//2 - self.gapW//2 : self.sinoSh[1]//2 + self.gapW//2 ]
        self.gapRng = np.s_[ : , self.gapRngX ]

        latentChannels = 7
        self.noise2latent = nn.Sequential(
            nn.Linear(TCfg.latentDim, self.sinoSize*latentChannels),
            nn.ReLU(),
            nn.Unflatten( 1, (latentChannels,) + self.sinoSh )
        )
        fillWheights(self.noise2latent)

        baseChannels = 64

        self.encode = nn.Sequential(

            nn.Conv2d(latentChannels+1, baseChannels, 3),
            nn.LeakyReLU(0.2),

            nn.Conv2d(baseChannels, baseChannels, 3),
            nn.LeakyReLU(0.2),

            nn.Conv2d(baseChannels, baseChannels, 3),
            nn.LeakyReLU(0.2),

        )
        fillWheights(self.encode)

        encSh = self.encode(torch.zeros((1,latentChannels+1,*self.sinoSh))).shape
        linChannels = math.prod(encSh)
        self.link = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linChannels, linChannels),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, encSh[1:]),
        )
        fillWheights(self.link)


        self.decode = nn.Sequential(

            nn.ConvTranspose2d(baseChannels, baseChannels, (3,1)),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(baseChannels, baseChannels, (3,1)),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(baseChannels, baseChannels, (3,1)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(baseChannels, 1, (1,3)),
            nn.Tanh()

        )
        fillWheights(self.decode)


        self.body = nn.Sequential(
            self.encode,
            self.link,
            self.decode
        )


    def forward(self, input):
        images, noises = input
        images, orgDims = unsqeeze4dim(images)
        latent = self.noise2latent(noises)
        modelIn = torch.cat((images,latent),dim=1)
        mIn = modelIn[:,0,*self.gapRng]
        mIn[()] = self.preProc(images[:,0,:,:])
        patches = self.body(modelIn)
        mIn = mIn.unsqueeze(1)
        #patches = mIn + torch.where( patches < 0 , patches * mIn , patches ) # no normalization
        patches = mIn + patches * torch.where( patches < 0 , mIn+0.5 , 1 ) # normalization
        return squeezeOrg(patches, orgDims)


    def preProc(self, images) :
        images = images.unsqueeze(0) # for the 2D case
        res = torch.zeros(images[...,*self.gapRng].shape, device=images.device)
        res[...,0] += 2*images[...,self.gapRngX.start-1] + images[...,self.gapRngX.stop]
        res[...,1] += 2*images[...,self.gapRngX.stop] + images[...,self.gapRngX.start-1]
        res = res.squeeze(0) # to compensate for the first squeeze
        return res/3

    def generatePatches(self, images, noises=None) :
        if noises is None :
            noises = torch.randn( 1 if images.dim() < 3 else images.shape[0], TCfg.latentDim).to(TCfg.device)
        return self.forward((images,noises))


    def fillImages(self, images, noises=None) :
        images[...,*self.gapRng] = self.generatePatches(images, noises)
        return images


    def generateImages(self, images, noises=None) :
        clone = images.clone()
        return self.fillImages(clone, noises)

generator2 = Generator2()
generator2 = load_model(generator2, model_path = os.path.join(modelsRoot, "gap2_cor.model_gen.pt" ))
generator2.to(TCfg.device)
generator2.requires_grad_(False)
generator2.eval()


class Generator4(nn.Module):

    def __init__(self):
        super(Generator4, self).__init__()

        self.gapW = 4
        self.sinoSh = (5*self.gapW,5*self.gapW) # 20,20
        self.sinoSize = math.prod(self.sinoSh)
        self.gapSh = (self.sinoSh[0],self.gapW)
        self.gapSize = math.prod(self.gapSh)
        self.gapRngX = np.s_[ self.sinoSh[1]//2 - self.gapW//2 : self.sinoSh[1]//2 + self.gapW//2 ]
        self.gapRng = np.s_[ : , self.gapRngX ]

        latentChannels = 7
        self.noise2latent = nn.Sequential(
            nn.Linear(TCfg.latentDim, self.sinoSize*latentChannels),
            nn.ReLU(),
            nn.Unflatten( 1, (latentChannels,) + self.sinoSh )
        )
        fillWheights(self.noise2latent)

        baseChannels = 128

        self.encode = nn.Sequential(

            nn.Conv2d(latentChannels+1, baseChannels, 3),
            nn.LeakyReLU(0.2),

            nn.Conv2d(baseChannels, baseChannels, 3, stride=2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(baseChannels, baseChannels, 3),
            nn.LeakyReLU(0.2),

            nn.Conv2d(baseChannels, baseChannels, 3),
            nn.LeakyReLU(0.2),

        )
        fillWheights(self.encode)


        encSh = self.encode(torch.zeros((1,latentChannels+1,*self.sinoSh))).shape
        linChannels = math.prod(encSh)
        self.link = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linChannels, linChannels),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, encSh[1:]),
        )
        fillWheights(self.link)


        self.decode = nn.Sequential(

            nn.ConvTranspose2d(baseChannels, baseChannels, (3,1)),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(baseChannels, baseChannels, (3,1)),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(baseChannels, baseChannels, (4,1), stride=(2,1)),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(baseChannels, baseChannels, (3,1)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(baseChannels, 1, 1),
            nn.Tanh()

        )
        fillWheights(self.decode)


        self.body = nn.Sequential(
            self.encode,
            self.link,
            self.decode
        )


    def forward(self, input):
        images, noises = input
        images, orgDims = unsqeeze4dim(images)
        latent = self.noise2latent(noises)
        modelIn = torch.cat((images,latent),dim=1)
        mIn = modelIn[:,0,*self.gapRng]
        mIn[()] = self.preProc(images[:,0,:,:])
        patches = self.body(modelIn)
        mIn = mIn.unsqueeze(1)
        #patches = mIn + torch.where( patches < 0 , patches * mIn , patches ) # no normalization
        patches = mIn + patches * torch.where( patches < 0 , mIn+0.5 , 1 ) # normalization
        return squeezeOrg(patches, orgDims)


    def preProc(self, images) :
        images, orgDims = unsqeeze4dim(images)
        preImages = torch.nn.functional.interpolate(images, scale_factor=0.5, mode='area')
        prePatches = generator2.generatePatches(preImages)
        prePatches = torch.nn.functional.interpolate(prePatches, scale_factor=2, mode='bilinear')
        return squeezeOrg(prePatches, orgDims)


    def generatePatches(self, images, noises=None) :
        if noises is None :
            noises = torch.randn( 1 if images.dim() < 3 else images.shape[0], TCfg.latentDim).to(TCfg.device)
        return self.forward((images,noises))


    def fillImages(self, images, noises=None) :
        images[...,*self.gapRng] = self.generatePatches(images, noises)
        return images


    def generateImages(self, images, noises=None) :
        clone = images.clone()
        return self.fillImages(clone, noises)

generator4 = Generator4()
generator4 = load_model(generator4, model_path = os.path.join(modelsRoot, "gap4_cor.model_gen.pt" ))
generator4.to(TCfg.device)
generator4.requires_grad_(False)
generator4.eval()


class Generator8(nn.Module):

    def __init__(self):
        super(Generator8, self).__init__()

        self.gapW = 8
        self.sinoSh = (5*self.gapW,5*self.gapW) # 20,20
        self.sinoSize = math.prod(self.sinoSh)
        self.gapSh = (self.sinoSh[0],self.gapW)
        self.gapSize = math.prod(self.gapSh)
        self.gapRngX = np.s_[ self.sinoSh[1]//2 - self.gapW//2 : self.sinoSh[1]//2 + self.gapW//2 ]
        self.gapRng = np.s_[ : , self.gapRngX ]

        latentChannels = 7
        self.noise2latent = nn.Sequential(
            nn.Linear(TCfg.latentDim, self.sinoSize*latentChannels),
            nn.ReLU(),
            nn.Unflatten( 1, (latentChannels,) + self.sinoSh )
        )
        fillWheights(self.noise2latent)

        baseChannels = 256

        self.encode = nn.Sequential(

            nn.Conv2d(latentChannels+1, baseChannels, 3, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(baseChannels, baseChannels, 3, stride=2, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(baseChannels, baseChannels, 3, stride=2, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(baseChannels, baseChannels, 3, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(baseChannels, baseChannels, 3, bias=False),
            nn.LeakyReLU(0.2),


        )
        fillWheights(self.encode)


        encSh = self.encode(torch.zeros((1,latentChannels+1,*self.sinoSh))).shape
        linChannels = math.prod(encSh)
        self.link = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linChannels, linChannels),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, encSh[1:]),
        )
        fillWheights(self.link)


        self.decode = nn.Sequential(

            nn.ConvTranspose2d(baseChannels, baseChannels, (3,1), bias=False),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(baseChannels, baseChannels, (3,1), bias=False),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(baseChannels, baseChannels, (4,1), stride=(2,1), bias=False),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(baseChannels, baseChannels, (4,3), stride=(2,1), bias=False),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(baseChannels, baseChannels, (3,3), bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(baseChannels, 1, 1, bias=False),
            nn.Tanh()

        )
        fillWheights(self.decode)


        self.body = nn.Sequential(
            self.encode,
            self.link,
            self.decode
        )


    def forward(self, input):
        images, noises = input
        images, orgDims = unsqeeze4dim(images)
        latent = self.noise2latent(noises)
        modelIn = torch.cat((images,latent),dim=1)
        mIn = modelIn[:,0,*self.gapRng]
        mIn[()] = self.preProc(images[:,0,:,:])
        patches = self.body(modelIn)
        #return patches
        mIn = mIn.unsqueeze(1)
        #patches = mIn + torch.where( patches < 0 , patches * mIn , patches ) # no normalization
        patches = mIn + patches * torch.where( patches < 0 , mIn+0.5 , 1 ) # normalization
        return squeezeOrg(patches, orgDims)


    def preProc(self, images) :
        images, orgDims = unsqeeze4dim(images)
        preImages = torch.nn.functional.interpolate(images, scale_factor=0.5, mode='area')
        prePatches = generator4.generatePatches(preImages)
        prePatches = torch.nn.functional.interpolate(prePatches, scale_factor=2, mode='bilinear')
        return squeezeOrg(prePatches, orgDims)


    def generatePatches(self, images, noises=None) :
        if noises is None :
            noises = torch.randn( 1 if images.dim() < 3 else images.shape[0], TCfg.latentDim).to(TCfg.device)
        return self.forward((images,noises))


    def fillImages(self, images, noises=None) :
        images[...,*self.gapRng] = self.generatePatches(images, noises)
        return images


    def generateImages(self, images, noises=None) :
        clone = images.clone()
        return self.fillImages(clone, noises)

generator8 = Generator8()
generator8 = load_model(generator8, model_path = os.path.join(modelsRoot, "gap8_cor.model_gen.pt" ))
generator8.to(TCfg.device)
generator8.requires_grad_(False)
generator8.eval()


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.gapW = DCfg.gapW
        self.sinoSh = (5*self.gapW,5*self.gapW) # 80,80
        self.sinoSize = math.prod(self.sinoSh)
        self.gapSh = (self.sinoSh[0],self.gapW)
        self.gapSize = math.prod(self.gapSh)
        self.gapRngX = np.s_[ self.sinoSh[1]//2 - self.gapW//2 : self.sinoSh[1]//2 + self.gapW//2 ]
        self.gapRng = np.s_[ : , self.gapRngX ]

        latentChannels = 7
        self.noise2latent = nn.Sequential(
            nn.Linear(TCfg.latentDim, self.sinoSize*latentChannels),
            nn.ReLU(),
            nn.Unflatten( 1, (latentChannels,) + self.sinoSh )
        )
        fillWheights(self.noise2latent)

        baseChannels = 64


        def encblock(chIn, chOut, kernel, stride=1) :
            return nn.Sequential (
                nn.Conv2d(chIn, chOut, kernel, stride=stride, bias=True),
                #nn.BatchNorm2d(chOut),
                nn.LeakyReLU(0.2),
                #nn.ReLU(),
            )
        self.encode = nn.Sequential(
            encblock(  latentChannels+1,   baseChannels, 3),
            encblock(  baseChannels,     2*baseChannels, 3, stride=2),
            encblock(2*baseChannels,     2*baseChannels, 3),
            encblock(2*baseChannels,     2*baseChannels, 3),
            encblock(2*baseChannels,     4*baseChannels, 3, stride=2),
            encblock(4*baseChannels,     4*baseChannels, 3),
            encblock(4*baseChannels,     8*baseChannels, 3, stride=2),
            encblock(8*baseChannels,     8*baseChannels, 3),
        )
        fillWheights(self.encode)


        encSh = self.encode(torch.zeros((1,latentChannels+1,*self.sinoSh))).shape
        linChannels = math.prod(encSh)
        self.link = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linChannels, linChannels),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, encSh[1:]),
        )
        fillWheights(self.link)

        def decblock(chIn, chOut, kernel, stride=1) :
            return nn.Sequential (
                nn.ConvTranspose2d(chIn, chOut, kernel, stride, bias=False),
                #nn.BatchNorm2d(chOut),
                nn.LeakyReLU(0.2),
                #nn.ReLU(),
            )
        self.decode = nn.Sequential(
            decblock(8*baseChannels, 8*baseChannels, 3),
            decblock(8*baseChannels, 4*baseChannels, 4, stride=2),
            decblock(4*baseChannels, 4*baseChannels, 3),
            decblock(4*baseChannels, 2*baseChannels, 4, stride=2),
            decblock(2*baseChannels, 2*baseChannels, 3),
            decblock(2*baseChannels, 2*baseChannels, 3),
            decblock(2*baseChannels,   baseChannels, 4, stride=2),
            decblock(baseChannels, baseChannels, 3),

            nn.Conv2d(baseChannels, 1, 1),
            nn.Tanh()
        )
        fillWheights(self.decode)


        self.body = nn.Sequential(
            self.encode,
            self.link,
            self.decode
        )


    def forward(self, input):
        images, noises = input
        images, orgDims = unsqeeze4dim(images)
        latent = self.noise2latent(noises)
        modelIn = torch.cat((images,latent),dim=1)
        mIn = modelIn[:,0,*self.gapRng]
        mIn[()] = self.preProc(images[:,0,:,:])
        patches = self.body(modelIn)[...,self.gapRngX]
        #return patches
        mIn = mIn.unsqueeze(1)
        patches = mIn + patches * torch.where( patches < 0 , mIn+0.5 , 1 ) # normalization
        return squeezeOrg(patches, orgDims)


    def preProc(self, images) :
        images, orgDims = unsqeeze4dim(images)
        preImages = torch.nn.functional.interpolate(images, scale_factor=0.5, mode='area')
        prePatches = generator8.generatePatches(preImages)
        prePatches = torch.nn.functional.interpolate(prePatches, scale_factor=2, mode='bilinear')
        return squeezeOrg(prePatches, orgDims)


    def generatePatches(self, images, noises=None) :
        if noises is None :
            noises = torch.randn( 1 if images.dim() < 3 else images.shape[0], TCfg.latentDim).to(TCfg.device)
        return self.forward((images,noises))


    def fillImages(self, images, noises=None) :
        images[...,*self.gapRng] = self.generatePatches(images, noises)
        return images


    def generateImages(self, images, noises=None) :
        clone = images.clone()
        return self.fillImages(clone, noises)

generator = Generator()
generator = load_model(generator, model_path = os.path.join(modelsRoot, "gap16_cor.model_gen.pt" ))
generator = generator.to(TCfg.device)
generator = generator.requires_grad_(False)
generator = generator.eval()



#%% EXEC

def fillSinogram(sinogram) :

    sinoW = sinogram.shape[-1]
    sinoL = sinogram.shape[-2]
    if sinoW % 5 :
        raise Exception(f"Sinogram width {sinoW} is not devisable bny 5.")
    blockW = sinoW // 5
    sinogram, _ = unsqeeze4dim(sinogram)
    sinogram = sinogram.to(TCfg.device)
    resizedSino = torch.zeros(( 1 , 1 , sinoL , DCfg.sinoSh[1] ), device=TCfg.device)
    resizedSino[ ... , : 2*DCfg.gapW ] = torch.nn.functional.interpolate(
        sinogram[ ... , : 2*blockW ], size=( sinoL , 2*DCfg.gapW ), mode='bilinear')
    resizedSino[ ... , 2*DCfg.gapW : 3*DCfg.gapW ] = torch.nn.functional.interpolate(
        sinogram[ ... , : 2*blockW : 3*blockW ], size=( sinoL , DCfg.gapW ), mode='bilinear')
    resizedSino[ ... , 3*DCfg.gapW:] = torch.nn.functional.interpolate(
        sinogram[ ... , 3*blockW : ], size=( sinoL , 2*DCfg.gapW ), mode='bilinear')

    blockH = DCfg.sinoSh[0]
    sinoCutStep = DCfg.gapW
    lastStart = sinoL - blockH
    nofBlocks, lastBlock = divmod(lastStart, sinoCutStep)
    modelIn = torch.empty( ( nofBlocks + bool(lastBlock) , 1 , *DCfg.sinoSh ), device=TCfg.device )
    for block in range(nofBlocks) :
        modelIn[ block, 0, ... ] = resizedSino[0 , 0, block * sinoCutStep : block * sinoCutStep + blockH , : ]
    if lastBlock :
        modelIn[ -1, 0, ... ] = resizedSino[0,0, -blockH : , : ]

    mytransforms = transforms.Compose([
        transforms.Normalize(mean=(0.5), std=(1))
    ])
    modelIn = mytransforms(modelIn)

    modelIn[ -1, 0, ... ] = modelIn[ -1, 0, ... ].flip(dims=(-2,)) # to get rid of the deffect in the end
    results = None
    with torch.no_grad() :
        results = generator.generatePatches(modelIn)
    results[ -1, 0, ... ] = results[ -1, 0, ... ].flip(dims=(-2,)) # to flip back

    if lastBlock :
        newLast = torch.zeros(DCfg.gapSh, device=TCfg.device)
        newLast[lastBlock:,:] = results[-1,0,lastBlock:,:]
        results[-1,0,...] = newLast
    preBlocks = torch.zeros((4,1,*DCfg.gapSh), device=TCfg.device)
    pstBlocks = torch.zeros((4,1,*DCfg.gapSh), device=TCfg.device)
    for curs in range(4) :
        preBlocks[ -curs-1 , 0 , sinoCutStep*(curs+1) :  , : ] = results[ 0 , 0 , : -sinoCutStep*(curs+1) , : ]
        pstBlocks[ curs , 0 , : (-sinoCutStep*curs) if curs else (blockH+1) , : ] = results[ -1 , 0 , sinoCutStep*curs : , : ]
    resultsPatched = torch.cat( (preBlocks, results, pstBlocks), dim=0 )

    blockCut = blockH / 5
    profileWeight = torch.empty( (blockH,), device=TCfg.device )
    for curi in range(blockH) :
        if curi < blockCut :
            profileWeight[curi] = 0
        elif curi < 2 * blockCut :
            profileWeight[curi] = ( curi - blockCut ) / blockCut
        elif curi < 3 * blockCut :
            profileWeight[curi] = 1
        elif curi < 4 * blockCut :
            profileWeight[curi] = ( 4*blockCut - curi ) / blockCut
        else :
            profileWeight[curi] = 0
    #plotData(profileWeight.numpy())
    resultsProfiled = ( resultsPatched + 0.5 ) * profileWeight.view(1,1,-1,1)
    stitchedGap = torch.zeros( ( (resultsProfiled.shape[0]-1) * sinoCutStep + blockH, DCfg.gapW ), device=TCfg.device )
    for curblock in range(resultsProfiled.shape[0]) :
        stitchedGap[ curblock*sinoCutStep : curblock*sinoCutStep + blockH , : ] += resultsProfiled[curblock,0,...]
    stitchedGap = stitchedGap.unsqueeze(0).unsqueeze(0)
    resizedGap = torch.nn.functional.interpolate(
        stitchedGap, size=( stitchedGap.shape[-2] ,  blockW), mode='bilinear')

    sinogram[..., 2*blockW : 3*blockW ] = resizedGap[0,0, sinoCutStep*4 : sinoCutStep*4 + sinoL, : ] / 2
    return sinogram


inData = getInData(args.input)
fsh = inData.shape[1:]
mask = loadImage(args.mask, fsh)
leftMask = np.ones(fsh, dtype=np.uint8)
outData, outFile = getOutData(args.output, inData.shape)


try :

    for curSl in tqdm.tqdm(range(fsh[-2])):

        gaps = []
        clmn=0
        gapStart=-1
        while clmn < fsh[-1] :
            value = mask[curSl,clmn]
            if ( value < 1 and gapStart >= 0 ) or \
               ( value >= 1 and gapStart < 0 ) :
                   clmn +=1
                   continue
            if value < 1 :
                if gapStart < 0  : # start the gap
                    gapStart = clmn
            else :
                if gapStart >= 0  : # end the gap
                    gaps.append(np.s_[gapStart:clmn])
                    gapStart = -1
            clmn += 1
        if gapStart >= 0:
            gaps.append(np.s_[gapStart:fsh[-1]])

        inSinogram = torch.tensor(inData[:,curSl,:], device=TCfg.device)

        def closeGapsFromList(gapsIn) :
            gapsToRet = []
            for gapI, gap in enumerate(gapsIn) :
                gapW = gap.stop-gap.start
                prevGap = gapsToRet[-1].stop if len(gapsToRet) else 0
                nextGap = gapsIn[gapI+1].start if gapI < len(gapsIn)-1 else fsh[-1]
                if  gapW <= 32 \
                and gap.start - prevGap > 2*gapW \
                and nextGap - gap.stop > 2*gapW :
                    stripe=np.s_[ gap.start - 2*gapW : gap.stop + 2*gapW]
                    stripeData = inSinogram[:,stripe]
                    filledData = fillSinogram(stripeData).squeeze()
                    inSinogram[:,stripe] = filledData
                else :
                    #print(f"Warning. Gap {gap} does not have enough space"
                    #      f" between adjacent gaps {np.s_[prevGap,nextGap]} to process. "
                    #      f" will try in the next iteration.")
                    gapsToRet.append(gap)
            return gapsToRet

        while True :
            gapsOnEnter = len(gaps)
            gaps = closeGapsFromList(gaps)
            if not len(gaps) or len(gaps) == gapsOnEnter :
                break
        curGap = 0
        while curGap < len(gaps)-1 : # try to combine gaps
            combGaps = [ np.s_[ gaps[curGap].start : gaps[curGap+1].stop ], ]
            gapLeft = len(closeGapsFromList(combGaps))
            if gapLeft :
                curGap += 1
            else :
                gaps = [ *gaps[:curGap], *gaps[curGap+2:] ]
        gaps = closeGapsFromList(gaps) # last attempt. needed at all?

        if len(gaps) :
            for gap in gaps :
                leftMask[curSl,gap] = 0
        outData[:,curSl,:] = inSinogram.cpu().numpy()

    leftMaskName = "".join(args.output.split(":")[0].split(".")[:-1])+"_mask.tif"
    leftMask *= 255
    tifffile.imwrite(leftMaskName, leftMask)


except :
    outFile.close()
    raise
outFile.close()
print("Done")


