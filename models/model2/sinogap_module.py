
import IPython

import sys
import os
import random
import time
import gc
import dataclasses
from dataclasses import dataclass, field
from enum import Enum

import math
import statistics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torchvision
from torch import optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave
import h5py
import tifffile
import tqdm


def initIfNew(var, val=None) :
    if var in locals() :
        return locals()[var]
    if var in globals() :
        return globals()[var]
    return val

@dataclass
class TCfgClass:
    exec : int
    latentDim: int
    batchSize: int
    labelSmoothFac: float
    learningRateD: float
    learningRateG: float
    device: torch.device = torch.device('cpu')
    batchSplit : int = 1
    nofEpochs: int = 0
    historyHDF : str = field(repr = True, init = False)
    logDir : str = field(repr = True, init = False)
    def __post_init__(self):
        if self.device == torch.device('cpu')  :
            self.device = torch.device(f"cuda:{self.exec}")
        self.historyHDF = f"train_{self.exec}.hdf"
        self.logDir = f"runs/experiment_{self.exec}"
        if self.batchSize % self.batchSplit :
            raise Exception(f"Batch size {self.batchSize} is not divisible by batch split {self.batchSplit}.")
global TCfg
TCfg = initIfNew('TCfg')


@dataclass
class DCfgClass:
    gapW : int
    sinoSh : tuple = field(repr = True, init = False)
    readSh : tuple = field(repr = True, init = False)
    sinoSize : int = field(repr = True, init = False)
    gapSh : tuple = field(repr = True, init = False)
    gapSize : int = field(repr = True, init = False)
    gapRngX : type(np.s_[:]) = field(repr = True, init = False)
    gapRng : type(np.s_[:]) = field(repr = True, init = False)
    disRng : type(np.s_[:]) = field(repr = True, init = False)
    def __post_init__(self):
        self.sinoSh = (5*self.gapW,5*self.gapW)
        self.readSh = (80, 80)
        self.sinoSize = math.prod(self.sinoSh)
        self.gapSh = (self.sinoSh[0],self.gapW)
        self.gapSize = math.prod(self.gapSh)
        self.gapRngX = np.s_[ self.sinoSh[1]//2 - self.gapW//2 : self.sinoSh[1]//2 + self.gapW//2 ]
        self.gapRng = np.s_[...,self.gapRngX]
        self.disRng = np.s_[ self.gapW:-self.gapW , self.gapRngX ]
DCfg = initIfNew('DCfg')


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def plotData(dataY, rangeY=None, dataYR=None, rangeYR=None,
             dataX=None, rangeX=None, rangeP=None,
             figsize=(16,8), saveTo=None, show=True):

    if type(dataY) is np.ndarray :
        plotData((dataY,), rangeY=rangeY, dataYR=dataYR, rangeYR=rangeYR,
             dataX=dataX, rangeX=rangeX, rangeP=rangeP,
             figsize=figsize, saveTo=saveTo, show=show)
        return
    if type(dataYR) is np.ndarray :
        plotData(dataY, rangeY=rangeY, dataYR=(dataYR,), rangeYR=rangeYR,
             dataX=dataX, rangeX=rangeX, rangeP=rangeP,
             figsize=figsize, saveTo=saveTo, show=show)
        return
    if type(dataY) is not tuple :
        eprint(f"Unknown data type to plot: {type(dataY)}.")
        return
    if type(dataYR) is not tuple and dataYR is not None:
        eprint(f"Unknown data type to plot: {type(dataYR)}.")
        return

    last = min( len(data) for data in dataY )
    if dataYR is not None:
        last = min( last,  min( len(data) for data in dataYR ) )
    if dataX is not None:
        last = min(last, len(dataX))
    if rangeP is None :
        rangeP = (0,last)
    elif type(rangeP) is int :
        rangeP = (0,rangeP) if rangeP > 0 else (-rangeP,last)
    elif type(rangeP) is tuple :
        rangeP = ( 0    if rangeP[0] is None else rangeP[0],
                   last if rangeP[1] is None else rangeP[1],)
    else :
        eprint(f"Bad data type on plotData input rangeP: {type(rangeP)}")
        raise Exception(f"Bug in the code.")
    rangeP = np.s_[ max(0, rangeP[0]) : min(last, rangeP[1]) ]
    if dataX is None :
        dataX = np.arange(rangeP.start, rangeP.stop)

    plt.style.use('default')
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.xaxis.grid(True, 'both', linestyle='dotted')
    if rangeX is not None :
        ax1.set_xlim(rangeX)
    else :
        ax1.set_xlim(rangeP.start,rangeP.stop-1)

    ax1.yaxis.grid(True, 'both', linestyle='dotted')
    nofPlots = len(dataY)
    if rangeY is not None:
        ax1.set_ylim(rangeY)
    colors = [ matplotlib.colors.hsv_to_rgb((hv/nofPlots, 1, 1)) for hv in range(nofPlots) ]
    for idx , data in enumerate(dataY):
        ax1.plot(dataX, data[rangeP], linestyle='-',  color=colors[idx])

    if dataYR is not None : # right Y axis
        ax2 = ax1.twinx()
        ax2.yaxis.grid(True, 'both', linestyle='dotted')
        nofPlots = len(dataYR)
        if rangeYR is not None:
            ax2.set_ylim(rangeYR)
        colors = [ matplotlib.colors.hsv_to_rgb((hv/nofPlots, 1, 1)) for hv in range(nofPlots) ]
        for idx , data in enumerate(dataYR):
            ax2.plot(dataX, data[rangeP], linestyle='dashed',  color=colors[idx])

    if saveTo:
        fig.savefig(saveTo)
    if not show:
        plt.close(fig)


def plotImage(image) :
    plt.imshow(image, cmap='gray')
    plt.axis("off")
    plt.show()


def plotImages(images) :
    for i, img in enumerate(images) :
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.axis("off")
    plt.show()


def sliceShape(shape, sl) :
    if type(shape) is int :
        shape = torch.Size([shape])
    if type(sl) is tuple :
        if len(shape) != len(sl) :
            raise Exception(f"Different sizes of shape {shape} and sl {sl}")
        out = []
        for i in range(0, len(shape)) :
            indeces = sl[i].indices(shape[i])
            out.append(indeces[1]-indeces[0])
        return out
    elif type(sl) is slice :
        indeces = sl.indices(shape[0])
        return indeces[1]-indeces[0]
    else :
        raise Exception(f"Incompatible object {sl}")


def tensorStat(stat) :
    print(f"{stat.mean().item():.3e}, {stat.std().item():.3e}, "
          f"{stat.min().item():.3e}, {stat.max().item():.3e}")


def fillWheights(seq) :
    for wh in seq :
        if hasattr(wh, 'weight') :
            #torch.nn.init.xavier_uniform_(wh.weight)
            #torch.nn.init.zeros_(wh.weight)
            #torch.nn.init.constant_(wh.weight, 0)
            #torch.nn.init.uniform_(wh.weight, a=0.0, b=1.0, generator=None)
            torch.nn.init.normal_(wh.weight, mean=0.0, std=0.01)
        if hasattr(wh, 'bias') :
            torch.nn.init.normal_(wh.bias, mean=0.0, std=0.01)


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


def set_seed(SEED_VALUE):
    torch.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed_all(SEED_VALUE)
    np.random.seed(SEED_VALUE)


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    return


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=TCfg.device, weights_only=True))
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


def createWriter(logDir, addToExisting=False) :
    if not addToExisting and os.path.exists(logDir) :
        raise Exception(f"Log directory \"{logDir}\" for the experiment already exists."
                        " Remove it or implicitry overwrite with setting addToExisting to True.")
    return SummaryWriter(logDir)
writer = initIfNew('writer')


class StripesFromHDF :

    def __init__(self, sampleName, maskName, bgName=None, dfName=None, loadToMem=True):

        sampleHDF = sampleName.split(':')
        if len(sampleHDF) != 2 :
            raise Exception(f"String \"{sampleName}\" does not represent an HDF5 format.")
        with h5py.File(sampleHDF[0],'r') as trgH5F:
            if  sampleHDF[1] not in trgH5F.keys():
                raise Exception(f"No dataset '{sampleHDF[1]}' in input file {sampleHDF[0]}.")
            self.data = trgH5F[sampleHDF[1]]
            if not self.data.size :
                raise Exception(f"Container \"{sampleName}\" is zero size.")
            self.sh = self.data.shape
            if len(self.sh) != 3 :
                raise Exception(f"Dimensions of the container \"{sampleName}\" is not 3 {self.sh}.")
            self.fsh = self.sh[1:3]
            self.volume = None
            if loadToMem :
                self.volume = np.empty(self.sh, dtype=np.float32)
                self.data.read_direct(self.volume)
                trgH5F.close()

            def loadImage(imageName) :
                if not imageName:
                    return None
                imdata = imread(imageName).astype(np.float32)
                if len(imdata.shape) == 3 :
                    imdata = np.mean(imdata[:,:,0:3], 2)
                if imdata.shape != self.fsh :
                    raise Exception(f"Dimensions of the input image \"{imageName}\" {imdata.shape} "
                                    f"do not match the face of the container \"{sampleName}\" {self.fsh}.")
                return imdata

            self.mask = loadImage(maskName)
            if self.mask is None :
                self.mask = np.ones(self.fsh, dtype=np.uint8)
            self.mask = self.mask.astype(bool)
            self.bg = loadImage(bgName)
            self.df = loadImage(dfName)
            if self.bg is not None :
                if self.df is not None:
                    self.bg -= self.df
                self.mask  &=  self.bg > 0.0

            self.forbidenSinos = self.mask
            self.forbidenSinos[:, -DCfg.readSh[-1]:] = 0
            for yCr in range(0,self.fsh[0]) :
                for xCr in range(0,self.fsh[1]) :
                    idx = np.s_[yCr,xCr]
                    if not self.mask[idx] :
                        self.forbidenSinos[ yCr, max(0, xCr-DCfg.readSh[1]) : xCr ] = 0
            self.availableSinos = np.count_nonzero(self.forbidenSinos)


    def __len__(self):
        return self.availableSinos


    def __getitem__(self, index=None):

        if type(index) is tuple and len(index) == 4 :
            zdx, ydx, xdx, sinoHeight = index
        else :
            while True:
                ydx = random.randint(0,self.fsh[0]-1)
                xdx = random.randint(0,self.fsh[1]-DCfg.readSh[-1]-1)
                if self.forbidenSinos[ydx,xdx] :
                    break
            sinoHeight = DCfg.readSh[0] if random.randint(0,1) else \
                random.randint(DCfg.readSh[0]+1, self.sh[0])
            zdx = random.randint(0,self.sh[0]-sinoHeight)
        idx = np.s_[ ydx , xdx : xdx + DCfg.readSh[-1] ]

        if self.volume is not None :
            data = self.volume[zdx:zdx+sinoHeight, *idx ]
        else :
            data = self.data[zdx:zdx+sinoHeight, *idx ]
            if self.df is not None :
                data -= self.df[None,*idx]
            if self.bg is not None :
                data /= self.bg[None,*idx]
        return (data, (zdx, ydx, xdx, sinoHeight) )
        #data = torch.from_numpy(data).clone().unsqueeze(0).to(TCfg.device)
        #if sinoHeight != DCfg.readSh[0] :
        #    data = torch.nn.functional.interpolate(data.unsqueeze(0), size=DCfg.readSh,mode='bilinear').squeeze(0)



    def get_dataset(self, transform=None) :

        class Sinos(torch.utils.data.Dataset) :
            def __init__(self, root, transform=None):
                self.container = root
                self.oblTransform = transforms.Compose( [ transforms.ToTensor(), transforms.Resize(DCfg.readSh)] )
                self.transform = transform
            def __len__(self):
                return self.container.__len__()
            def __getitem__(self, index=None, doTransform=True):
                data, index = self.container.__getitem__(index)
                data = self.oblTransform(data)
                if doTransform and self.transform :
                    data = self.transform(data)
                return (data, index)

        return Sinos(self, transform)


class StripesFromHDFs :

    def __init__(self, bases):
        self.collection = []
        for base in bases :
            print(f"Loading train set {len(self.collection)+1} of {len(bases)}: " + base + " ... ", end="")
            self.collection.append(
                StripesFromHDF(f"storage/{base}.hdf:/data", f"storage/{base}.mask++.tif", None, None) )
            print("Done")


    def __getitem__(self, index=None):

        if type(index) is tuple and len(index) == 5 :
            setdx, zdx, ydx, xdx, sinoHeight = index
            return self.collection[setdx].__getitem__((setdx, zdx, ydx, xdx, sinoHeight))
        else :
            cindex = random.randint(0,len(self)-1)
            leftover = cindex
            for setdx in range(len(self.collection)) :
                setLen = len(self.collection[setdx])
                if leftover >= setLen :
                    leftover -= setLen
                else :
                    data, insetdx = self.collection[setdx].__getitem__()
                    return (data, (setdx, *insetdx) )
            else :
                raise f"No set for index {cindex}. Should never happen."


    def __len__(self):
        return sum( [ len(set) for set in self.collection ] )


    def get_dataset(self, transform=None) :

        class Sinos(torch.utils.data.Dataset) :
            def __init__(self, root, transform=None):
                self.container = root
                self.oblTransform = transforms.Compose( [ transforms.ToTensor(), transforms.Resize(DCfg.readSh)] )
                self.transform = transform
            def __len__(self):
                return self.container.__len__()
            def __getitem__(self, index=None, doTransform=True):
                data, index = self.container.__getitem__(index)
                data = self.oblTransform(data)
                if doTransform and self.transform :
                    data = self.transform(data)
                return (data, index)

        return Sinos(self, transform)


examplesDb = {}
examplesDb[2] = [
                  263185,
                  173496,
                  213234,
                  241201,
                  264646,
                  195114,
                  195999,
                  863528,
                  755484,
                  222701,
                  818392,
                  952538,
                  801601,
                  944579,
                  1082431,
                  842400,
                ]
examplesDb[4] = [
                  263185,
                  173496,
                  213234,
                  241201,
                  264646,
                  195114,
                  195999,
                  863528,
                  755484,
                  222701,
                  818392,
                  #952538,
                  #801601,
                  #944579,
                  #1082431,
                  842400,
                ]
examplesDb[8] = [
                  263185,
                  173496,
                  213234,
                  241201,
                  264646,
                  195114,
                  195999,
                  #863528,
                  #755484,
                  #222701,
                  #818392,
                  #952538,
                  #801601,
                  #944579,
                  #1082431,
                  842400,
                ]
examplesDb[16] = [
                  263185,
                  173496,
                  213234,
                  #241201,
                  #264646,
                  #195114,
                  #195999,
                  #863528,
                  #755484,
                  #222701,
                  #818392,
                  #952538,
                  #801601,
                  #944579,
                  #1082431,
                  842400,
                ]


examples = initIfNew('examples')


listOfTrainData = [ "18515.Lamb1_Eiger_7m_45keV_360Scan"
                  , "18692a.ExpChicken6mGyShift"
                  , "18692b_input_PhantomM"
                  , "18692b.MinceO"
                  , "19022g.11-EggLard"
                  , "19736b.09_Feb.4176862R_Eig_Threshold-4keV"
                  , "19736c.8733147R_Eig_Threshold-8keV.SAMPLE_Y1"
                  , "20982b.04_774784R"
                  , "23574.8965435L.Eiger.32kev_org"
                  ]
def createTrainSet() :
    sinoRoot = StripesFromHDFs(listOfTrainData)
    mytransforms = transforms.Compose([
            transforms.Resize(DCfg.sinoSh),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=(0.5), std=(1))
    ])
    return sinoRoot.get_dataset(mytransforms)


def createDataLoader(tSet, num_workers=os.cpu_count()) :
    return torch.utils.data.DataLoader(
        dataset=tSet,
        batch_size=TCfg.batchSize,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True
    )


class PrepackedHDF :

    def __init__(self, sampleName):
        sampleHDF = sampleName.split(':')
        if len(sampleHDF) != 2 :
            raise Exception(f"String \"{sampleName}\" does not represent an HDF5 format.")
        with h5py.File(sampleHDF[0],'r') as trgH5F:
            if  sampleHDF[1] not in trgH5F.keys():
                raise Exception(f"No dataset '{sampleHDF[1]}' in input file {sampleHDF[0]}.")
            self.data = trgH5F[sampleHDF[1]]
            if not self.data.size :
                raise Exception(f"Container \"{sampleName}\" is zero size.")
            self.sh = self.data.shape
            if len(self.sh) != 3 :
                raise Exception(f"Dimensions of the container \"{sampleName}\" is not 3 {self.sh}.")
            self.fsh = self.sh[1:3]
            if self.fsh != (80,80) :
                raise Exception(f"Dimensions of the container \"{sampleName}\" is not 80,80 {self.fsh}.")
            self.volume = None
            self.volume = np.empty(self.sh, dtype=np.float32)
            self.data.read_direct(self.volume)
            trgH5F.close()

    def get_dataset(self, transform=None) :

        class Sinos(torch.utils.data.Dataset) :

            def __init__(self, root, transform=None):
                self.container = root
                self.transform = transform

            def __len__(self):
                return self.container.sh[0]

            def __getitem__(self, index=None, doTransform=True):
                if index is None :
                    index = random.randint(0,self.container.volume.shape[0]-1)
                data=self.container.volume[index,...]
                data = torch.from_numpy(data).clone().unsqueeze(0)
                if doTransform :
                    data = self.transform(data)
                return (data, index)

        return Sinos(self, transform)

def createTestSet() :
    print("Loading test set ... ", end="")
    sinoRoot = PrepackedHDF("storage/test/testSetSmall.hdf:/data")
    print("Done")
    mytransforms = transforms.Compose([
            transforms.Resize(DCfg.sinoSh),
            transforms.Normalize(mean=(0.5), std=(1))
    ])
    return sinoRoot.get_dataset(mytransforms)



def createReferences(tSet, toShow = 0) :
    global examples
    examples = examplesDb[DCfg.gapW].copy()
    if toShow :
        examples.insert(0, examples.pop(toShow))
    mytransforms = transforms.Compose([
            transforms.Resize(DCfg.sinoSh),
            transforms.Normalize(mean=(0.5), std=(1))
    ])
    refImages = torch.stack( [ mytransforms(tSet.__getitem__(ex, doTransform=False)[0])
                               for ex in examples ] ).to(TCfg.device)
    #refImages = torch.stack( [ tSet.__getitem__(ex, doTransform=False)[0] for ex in examples ] ).to(TCfg.device)
    refNoises = torch.randn((refImages.shape[0],TCfg.latentDim)).to(TCfg.device)
    return refImages, refNoises
refImages = initIfNew('refImages')
refNoises = initIfNew('refNoises')


def showMe(tSet, item=None) :
    global refImages, refNoises
    image = None
    if item is None :
        while True:
            image, index = tSet[random.randint(0,len(tSet)-1)]
            if image.mean() > 0 and image.min() < -0.1 :
                print (f"{index}")
                break
    elif isinstance(item, int) :
        image = refImages[0,...]
    else :
        image, _,_ = tSet.__getitem__(*item)
    image = image.squeeze()
    tensorStat(image)
    plotImage(image.cpu())
    image = image.to(TCfg.device)


save_interim = None


class GeneratorTemplate(nn.Module):

    def __init__(self, gapW, latentChannels=0):
        super(GeneratorTemplate, self).__init__()

        self.gapW = gapW
        self.sinoSh = (5*self.gapW,5*self.gapW) # 10,10
        self.sinoSize = math.prod(self.sinoSh)
        self.gapSh = (self.sinoSh[0],self.gapW)
        self.gapSize = math.prod(self.gapSh)
        self.gapRngX = np.s_[ self.sinoSh[1]//2 - self.gapW//2 : self.sinoSh[1]//2 + self.gapW//2 ]
        self.gapRng = np.s_[...,self.gapRngX]
        self.latentChannels = latentChannels
        self.baseChannels = 64
        self.amplitude = 2
        self.lowResGenerator = None


    def createLatent(self) :
        if self.latentChannels == 0 :
            return None
        toRet =  nn.Sequential(
            nn.Linear(TCfg.latentDim, self.sinoSize*self.latentChannels),
            nn.ReLU(),
            nn.Unflatten( 1, (self.latentChannels,) + self.sinoSh )
        )
        fillWheights(toRet)
        return toRet


    def encblock(self, chIn, chOut, kernel, stride=1, norm=False, dopadding=False) :
        chIn = int(chIn*self.baseChannels)
        chOut = int(chOut*self.baseChannels)
        layers = []
        layers.append( nn.Conv2d(chIn, chOut, kernel, stride=stride, bias=True,
                                padding='same', padding_mode='reflect') \
                                if stride == 1 and dopadding else \
                                nn.Conv2d(chIn, chOut, kernel, stride=stride, bias=True)
                     )
        if norm :
            layers.append(nn.BatchNorm2d(chOut))
        layers.append(nn.LeakyReLU(0.2))
        fillWheights(layers)
        return torch.nn.Sequential(*layers)


    def decblock(self, chIn, chOut, kernel, stride=1, norm=False, dopadding=False) :
        chIn = int(chIn*self.baseChannels)
        chOut = int(chOut*self.baseChannels)
        layers = []
        layers.append( nn.ConvTranspose2d(chIn, chOut, kernel, stride, bias=True,
                                          padding=1) \
                       if stride == 1 and dopadding else \
                       nn.ConvTranspose2d(chIn, chOut, kernel, stride, bias=True)
                      )
        if norm :
            layers.append(nn.BatchNorm2d(chOut))
        layers.append(nn.LeakyReLU(0.2))
        fillWheights(layers)
        return torch.nn.Sequential(*layers)


    def createFClink(self) :
        smpl = torch.zeros((1, 1+self.latentChannels, *self.sinoSh))
        for encoder in self.encoders :
            smpl = encoder(smpl)
        encSh = smpl.shape
        linChannels = math.prod(encSh)
        toRet = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linChannels, linChannels),
            nn.LeakyReLU(0.2),
            nn.Linear(linChannels, linChannels),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, encSh[1:]),
        )
        fillWheights(toRet)
        return toRet


    def createLastTouch(self) :
        toRet = nn.Sequential(
            nn.Conv2d(self.baseChannels+1, 1, 1),
            nn.Tanh(),
        )
        fillWheights(toRet)
        return toRet


    def generatePatches(self, images, noises=None) :
        if noises is None :
            noises = torch.randn( 1 if images.dim() < 3 else images.shape[0], TCfg.latentDim).to(TCfg.device)
        return self.forward((images,noises))


    def fillImages(self, images, noises=None) :
        images[self.gapRng] = self.generatePatches(images, noises)
        return images


    def generateImages(self, images, noises=None) :
        clone = images.clone()
        return self.fillImages(clone, noises)


    def preProc(self, images) :
        images, orgDims = unsqeeze4dim(images)
        if self.gapW == 2:
            res = torch.zeros(images[self.gapRng].shape, device=images.device)
            res[...,0] += 2*images[...,self.gapRngX.start-1] + images[...,self.gapRngX.stop]
            res[...,1] += 2*images[...,self.gapRngX.stop] + images[...,self.gapRngX.start-1]
            res /= 3
        elif self.lowResGenerator is None and not self.gapW//2 in lowResGenerators :
            res = torch.zeros(images[self.gapRng].shape, device=images.device, requires_grad=False)
        else :
            preImages = torch.nn.functional.interpolate(images, scale_factor=0.5, mode='area')
            # lowRes generator to be trained if they are a part of the generator
            lrGen = lowResGenerators[self.gapW//2] if self.lowResGenerator is None else self.lowResGenerator
            with torch.set_grad_enabled(not self.lowResGenerator is None) :
                res = lrGen.generatePatches(preImages)
                res = torch.nn.functional.interpolate(res, scale_factor=2, mode='bilinear')
        return squeezeOrg(res, orgDims)


    def forward(self, input):

        if not save_interim is None :
            save_interim[self.gapW] = {}

        images, noises = input
        if not save_interim is None :
            save_interim[self.gapW]['input'] = images.clone().detach()
        images, orgDims = unsqeeze4dim(images)
        modelIn = images.clone()
        modelIn[self.gapRng] = self.preProc(images)
        if not save_interim is None :
            save_interim[self.gapW]['preproc'] = modelIn.clone().detach()
        stDev = modelIn.std(dim=(-1,-2))[...,None,None]

        if self.latentChannels :
            latent = self.noise2latent(noises)
            dwTrain = [torch.cat((modelIn, latent), dim=1),]
        else :
            dwTrain = [modelIn,]
        for encoder in self.encoders :
            dwTrain.append(encoder(dwTrain[-1]))
        mid = self.fcLink(dwTrain[-1])
        upTrain = [mid]
        for level, decoder in enumerate(self.decoders) :
            upTrain.append( decoder( torch.cat( (upTrain[-1], dwTrain[-1-level]), dim=1 ) ) )
        res = self.lastTouch(torch.cat( (upTrain[-1], modelIn ), dim=1 ))

        patches = modelIn[self.gapRng] + self.amplitude * stDev * res[self.gapRng]
        if not save_interim is None :
            save_interim[self.gapW]['output'] = save_interim[self.gapW]['input'].clone()
            save_interim[self.gapW]['output'][self.gapRng] = patches.clone().detach()
        return squeezeOrg(patches, orgDims)


generator = initIfNew('generator')
lowResGenerators = initIfNew('lowResGenerators', {})


class DiscriminatorTemplate(nn.Module):

    def __init__(self, omitEdges=0):
        super(DiscriminatorTemplate, self).__init__()
        self.baseChannels = 64
        self.omitEdges = omitEdges


    def encblock(self, chIn, chOut, kernel, stride=1, norm=False, dopadding=False) :
        chIn = int(chIn*self.baseChannels)
        chOut = int(chOut*self.baseChannels)
        layers = []
        layers.append( nn.Conv2d(chIn, chOut, kernel, stride=stride, bias=True,
                                padding='same', padding_mode='reflect') \
                                if stride == 1 and dopadding else \
                                nn.Conv2d(chIn, chOut, kernel, stride=stride, bias=True)
                     )
        if norm :
            layers.append(nn.BatchNorm2d(chOut))
        layers.append(nn.LeakyReLU(0.2))
        fillWheights(layers)
        return torch.nn.Sequential(*layers)

    def createHead(self) :
        encSh = self.body(torch.zeros((1,1,*DCfg.sinoSh))).shape
        linChannels = math.prod(encSh)
        toRet = nn.Sequential(
            nn.Flatten(),
            #nn.Dropout(0.4),
            nn.Linear(linChannels, self.baseChannels*4),
            #nn.Linear(linChannels, 1),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.4),
            nn.Linear(self.baseChannels*4, 1),
            nn.Sigmoid(),
        )
        fillWheights(toRet)
        return toRet

    def forward(self, images):
        if images.dim() == 3:
            images = images.unsqueeze(1)
        if self.omitEdges :
            images = images.clone() # I want to exclude two blocks on the edges :
            images[ ..., :self.omitEdges, DCfg.gapRngX ] = 0
            images[ ..., -self.omitEdges:, DCfg.gapRngX ] = 0
        convRes = self.body(images)
        res = self.head(convRes)
        return res

discriminator = initIfNew('discriminator')
noAdv=False


def createOptimizer(model, lr) :
    return optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.5, 0.999)
    )
optimizer_G = initIfNew('optimizer_G')
optimizer_D = initIfNew('optimizer_D')


def restoreCheckpoint(path=None, logDir=None) :
    if logDir is None :
        logDir = TCfg.logDir
    if path is None :
        if os.path.exists(logDir) :
            raise Exception(f"Starting new experiment with existing log directory \"{logDir}\"."
                            " Remove it .")
        try : os.remove(TCfg.historyHDF)
        except : pass
        return 0, 0, 0, 1, 0, TrainResClass()
    else :
        return loadCheckPoint(path, generator, discriminator, optimizer_G, optimizer_D)


def saveModels(path="") :
    save_model(generator, model_path = ( path if path else f"model_{TCfg.exec}" ) + "_gen.pt" )
    save_model(discriminator, model_path = ( path if path else f"model_{TCfg.exec}" ) + "_dis.pt"  )


def createCriteria() :
    BCE = nn.BCELoss(reduction='none')
    MSE = nn.MSELoss(reduction='none')
    L1L = nn.L1Loss(reduction='none')
    return BCE, MSE, L1L
BCE, MSE, L1L = createCriteria()
lossDifCoef = 0
lossAdvCoef = 1.0

def applyWeights(inp, weights, storePerIm=None):
    inp = inp.squeeze()
    if not inp.dim() :
        inp = inp.unsqueeze(0)
    sum = len(inp)
    if not weights is None :
        inp *= weights
        sum = weights.sum()
    if storePerIm is not None : # must be list
        storePerIm.extend(inp.tolist())
    return inp.sum()/sum

def loss_Adv(y_true, y_pred, weights=None, storePerIm=None):
    loss = BCE(y_pred, y_true)
    return applyWeights(loss, weights, storePerIm=storePerIm)

def loss_MSE(p_true, p_pred, weights=None, storePerIm=None):
    loss = MSE(p_pred, p_true).mean(dim=(-1,-2))
    return applyWeights(loss, weights, storePerIm=storePerIm)

def loss_L1L(p_true, p_pred, weights=None, storePerIm=None):
    loss = L1L(p_pred, p_true).mean(dim=(-1,-2))
    return applyWeights(loss, weights, storePerIm=storePerIm)

eDinfo = None
def loss_Rec(p_true, p_pred, weights=None, storePerIm=None, normalizeRec=None):
    global eDinfo
    loss = MSE(p_pred, p_true).mean(dim=(-1,-2))
    if normalizeRec is not None :
        loss *= normalizeRec
    if loss.dim() :
        hDindex = loss.argmax()
        lDindex = loss.argmin()
        eDinfo = (hDindex, loss[hDindex].item(), lDindex, loss[lDindex].item() )
    return applyWeights(loss, weights, storePerIm=storePerIm)


def loss_Gen(y_true, y_pred, p_true, p_pred, weights=None, normalizeRec=None):
    lossAdv = loss_Adv(y_true, y_pred, weights)
    lossDif = loss_Rec(p_pred, p_true, weights, normalizeRec=normalizeRec)
    return lossAdv, lossDif


def summarizeSet(dataloader, onPrep=True, storesPerIm=None):

    MSE_diffs, L1L_diffs, Rec_diffs, Real_probs, Fake_probs = [], [], [], [], []
    totalNofIm = 0
    generator.to(TCfg.device)
    #generator.train()
    generator.eval()
    #discriminator.eval()
    if storesPerIm is not None : # must be list of five lists
        for lst in storesPerIm :
            lst.clear()
    else :
        storesPerIm = [None, None, None, None, None]
    with torch.no_grad() :
        for it , data in tqdm.tqdm(enumerate(dataloader), total=int(len(dataloader))):
            images = data[0].squeeze(1).to(TCfg.device)
            nofIm = images.shape[0]
            subBatchSize = nofIm // TCfg.batchSplit
            totalNofIm += nofIm
            procImages, procData = imagesPreProc(images)
            genImages = procImages.clone()

            rprob = fprob = 0
            for i in range(TCfg.batchSplit) :
                subRange = np.s_[i*subBatchSize:(i+1)*subBatchSize] if TCfg.batchSplit > 1 else np.s_[...]
                subProcImages = procImages[subRange,...]
                patchImages = generator.preProc(subProcImages) \
                              if onPrep else \
                              generator.generatePatches(subProcImages)
                genImages[subRange,:,DCfg.gapRngX] = patchImages
                if not noAdv :
                    rprobs = discriminator(subProcImages)
                    if storesPerIm[3] is not None :
                        storesPerIm[3].extend(rprobs.tolist())
                    rprob += rprobs.sum().item()
                    fprobs = discriminator(genImages[subRange,...])
                    if storesPerIm[4] is not None :
                        storesPerIm[4].extend(fprobs.tolist())
                    fprob += fprobs.sum().item()
            procImages = imagesPostProc(genImages, procData)
            MSE_diffs.append( nofIm * loss_MSE(images[DCfg.gapRng], procImages[DCfg.gapRng]
                                              ,storePerIm = storesPerIm[2]))
            L1L_diffs.append( nofIm * loss_L1L(images[DCfg.gapRng], procImages[DCfg.gapRng]
                                              ,storePerIm = storesPerIm[1]))
            Rec_diffs.append( nofIm * loss_Rec(images[DCfg.gapRng], procImages[DCfg.gapRng]
                                              ,storePerIm = storesPerIm[0]
                                              ,normalizeRec=calculateNorm(images)))
            Real_probs.append(rprob)
            Fake_probs.append(fprob)

    MSE_diff = sum(MSE_diffs) / totalNofIm
    L1L_diff = sum(L1L_diffs) / totalNofIm
    Rec_diff = sum(Rec_diffs) / totalNofIm
    Real_prob = sum(Real_probs) / totalNofIm if not noAdv else 0
    Fake_prob = sum(Fake_probs) / totalNofIm if not noAdv else 0
    print (f"Summary. Rec: {Rec_diff:.3e}, MSE: {MSE_diff:.3e}, L1L: {L1L_diff:.3e}, Dis: {Real_prob:.3e}, Gen: {Fake_prob:.3e}.")
    return Rec_diff, MSE_diff, L1L_diff, Real_prob, Fake_prob


def generateDiffImages(images, layout=None) :
    images, orgDim = unsqeeze4dim(images)
    dif = torch.zeros((images.shape[0], 1, *DCfg.sinoSh))
    hGap = DCfg.gapW // 2
    pre = images.clone()
    gen = images.clone()
    with torch.no_grad() :
        wghts = calculateWeights(images)
        generator.eval()
        pre[DCfg.gapRng] = generator.preProc(images)
        gen[DCfg.gapRng] = generator.generatePatches(images)
        dif[DCfg.gapRng] = (gen - pre)[DCfg.gapRng]
        dif[...,hGap:hGap+DCfg.gapW] = (images - pre)[DCfg.gapRng]
        dif[...,-DCfg.gapW-hGap:-hGap] = (images - gen)[DCfg.gapRng]
        for curim in range(images.shape[0]) :
            if ( cof := max(-dif[curim,...].min(),dif[curim,...].max()) ) != 0 :
                dif[curim,...] /= cof
            else :
                dif[curim,...] = 0
        probs = torch.empty(images.shape[0],3)
        dists = torch.empty(images.shape[0],3)
        #discriminator.eval()
        probs[:,0] = discriminator(images)[:,0]
        probs[:,1] = discriminator(pre)[:,0]
        probs[:,2] = discriminator(gen)[:,0]
        dists[:,0] = loss_Rec(images[DCfg.gapRng], gen[DCfg.gapRng], wghts, normalizeRec=calculateNorm(images))
        dists[:,1] = loss_MSE(images[DCfg.gapRng], gen[DCfg.gapRng], wghts)
        dists[:,2] = loss_L1L(images[DCfg.gapRng], gen[DCfg.gapRng], wghts)

    simages = None
    if not layout is None :
        def stretch(stretchme, mm, aa) :
            return ( stretchme - mm ) * 2 / aa - 1 if ampl > 0 else stretchme * 0
        simages = images.clone()
        for curim in range(images.shape[0]) :
            rng = np.s_[curim,...]
            minv = min(images[rng].min(), pre[rng].min(), gen[rng].min()).item()
            ampl = max(images[rng].max(), pre[rng].max(), gen[rng].max()).item() - minv
            simages[rng] = stretch(simages[rng], minv, ampl)
            pre[rng] = stretch(pre[rng], minv, ampl)
            gen[rng] = stretch(gen[rng], minv, ampl)

    cGap = DCfg.gapW
    if layout == 0 :
        collage = torch.empty(images.shape[0], 4, *DCfg.sinoSh)
        collage[:,0,...] = simages[:,0,...]
        collage[:,1,...] = pre[:,0,...]
        collage[:,2,...] = gen[:,0,...]
        collage[:,3,...] = dif[:,0,...]
    elif layout == 2 :
        collage = torch.zeros((images.shape[0], 1, DCfg.sinoSh[0]*2 + cGap, DCfg.sinoSh[1]*2 + cGap ))
        collage[..., :DCfg.sinoSh[0], :DCfg.sinoSh[1]] = gen
        collage[..., :DCfg.sinoSh[0], DCfg.sinoSh[1]+cGap:] = pre
        collage[..., DCfg.sinoSh[0]+cGap:, :DCfg.sinoSh[1]] = simages
        collage[..., DCfg.sinoSh[0]+cGap:, DCfg.sinoSh[1]+cGap:] = dif
    elif layout == 4 :
        collage = torch.zeros((images.shape[0], 1, DCfg.sinoSh[0], 4*DCfg.sinoSh[1] + 3*cGap))
        collage[..., :DCfg.sinoSh[1]] = simages
        collage[..., DCfg.sinoSh[1]+cGap:2*DCfg.sinoSh[1]+cGap] = gen
        collage[..., 2*DCfg.sinoSh[1]+2*cGap:3*DCfg.sinoSh[1]+2*cGap] = dif
        collage[..., 3*DCfg.sinoSh[1]+3*cGap:4*DCfg.sinoSh[1]+4*cGap] = pre
    elif layout == -4 :
        collage = torch.zeros( (images.shape[0], 1, 4*DCfg.sinoSh[0] + 3*cGap, DCfg.sinoSh[1]))
        collage[... , :DCfg.sinoSh[0] , : ] = simages
        collage[... , DCfg.sinoSh[0]+cGap:2*DCfg.sinoSh[0]+cGap , :] = gen
        collage[... , 2*DCfg.sinoSh[0]+2*cGap:3*DCfg.sinoSh[0]+2*cGap , : ] = dif
        collage[... , 3*DCfg.sinoSh[0]+3*cGap:4*DCfg.sinoSh[0]+4*cGap , : ] = pre
    else :
        collage = dif
    collage = squeezeOrg(collage,orgDim)

    return collage, probs, dists


def logStep(iter, write=True) :
    colImgs, probs, dists = generateDiffImages(refImages, layout=-4)
    probs = probs.mean(dim=0)
    dists = dists.mean(dim=0)
    colImgs = colImgs.squeeze()
    cSh = colImgs.shape
    gapH = DCfg.gapW
    collage = np.zeros( ( cSh[-2], cSh[0]*cSh[-1] + (cSh[0]-1)*gapH ), dtype=np.float32  )
    for curI in range(cSh[0]) :
        collage[ : , curI * (cSh[-1]+gapH) : curI * (cSh[-1]+gapH) + cSh[-1]] = colImgs[curI,...]
    #writer.add_scalars("Probs of ref images",
    #                   {'Ref':probs[0]
    #                   ,'Gen':probs[2]
    #                   ,'Pre':probs[1]
    #                   }, iter )
    #writer.add_scalars("Dist of ref images",
    #                   { 'REC' : dists[0]
    #                   , 'MSE' : dists[1]
    #                   , 'L1L' : dists[2]
    #                   }, iter )
    try :
        addToHDF(TCfg.historyHDF, "data", collage)
    except :
        eprint("Failed to save.")
    return collage, probs, dists


def initialTest() :
    with torch.inference_mode() :
        collage, probs, _ = logStep(iter, not iter)
        print("Probabilities of reference images: "
              f'Ref: {probs[0]:.3e}, '
              f'Gen: {probs[2]:.3e}, '
              f'Pre: {probs[1]:.3e}.')
        #generator.eval()
        pre = generator.preProc(refImages)
        wghts = calculateWeights(refImages)
        ref_loss_Rec = loss_Rec(refImages[DCfg.gapRng], pre, wghts, normalizeRec=calculateNorm(refImages))
        ref_loss_MSE = loss_MSE(refImages[DCfg.gapRng], pre, wghts)
        ref_loss_L1L = loss_L1L(refImages[DCfg.gapRng], pre, wghts)
        print("Distances of reference images: "
              f"REC: {ref_loss_Rec:.3e}, "
              f"MSE: {ref_loss_MSE:.3e}, "
              f"L1L: {ref_loss_L1L:.3e}.")
        #if not epoch :
        #    writer.add_scalars("Dist of ref images",
        #                          { 'REC' : ref_loss_Rec
        #                          , 'MSE' : ref_loss_MSE
        #                          , 'L1L' : ref_loss_L1L
        #                          }, 0 )
        plotImage(collage)


def calculateWeights(images) :
    return None

def calculateNorm(images) :
    mean2 = images[...,:DCfg.gapRngX.start].mean(dim=(-1,-2)) \
          + images[...,DCfg.gapRngX.stop:].mean(dim=(-1,-2))
    return 2 / ( 1 + mean2 + 1e-5 ) # to denorm and adjust for mean



def imagesPreProc(images) :
    return images, None

def imagesPostProc(images, procData=None) :
    return images


@dataclass
class TrainInfoClass:
    bestRealIndex = 0
    worstRealIndex = 0
    bestFakeIndex = 0
    worstFakeIndex = 0
    bestRealProb = 0
    worstRealProb = 0
    bestFakeProb = 0
    worstFakeProb = 0
    bestRealImage = None
    worstRealImage = None
    bestFakeImage = None
    worstFakeImage = None
    highestDifIndex = 0
    lowestDifIndex = 0
    highestDif = 0
    lowestDif = 0
    highestDifImageOrg = None
    highestDifImageGen = None
    ratReal = 0.0
    ratFake = 0.0
    totalImages = 0
    iterations = 0
    disPerformed = 0
    genPerformed = 0
    totPerformed = 0

@dataclass
class TrainResClass:
    lossD : any = 0
    lossGA : any = 0
    lossGD : any = 0
    lossMSE : any = 0
    lossL1L : any = 0
    predReal : any = 0
    predPre : any = 0
    predFake : any = 0
    nofIm : int = 0
    def __add__(self, other):
        toRet = TrainResClass()
        for field in dataclasses.fields(TrainResClass):
            fn = field.name
            setattr(toRet, fn, getattr(self, fn) + getattr(other, fn) )
        return toRet
    def __mul__(self, other):
        toRet = TrainResClass()
        for field in dataclasses.fields(TrainResClass):
            fn = field.name
            setattr(toRet, fn, getattr(self, fn) * other )
        return toRet
    __rmul__ = __mul__




def saveCheckPoint(path, epoch, iterations, minGEpoch, minGdLoss,
                   generator, discriminator,
                   optimizerGen=None, optimizerDis=None,
                   schedulerGen=None, schedulerDis=None,
                   startFrom=0, interimRes=TrainResClass()) :
    checkPoint = {}
    checkPoint['epoch'] = epoch
    checkPoint['iterations'] = iterations
    checkPoint['minGEpoch'] = minGEpoch
    checkPoint['minGdLoss'] = minGdLoss
    checkPoint['startFrom'] = startFrom
    checkPoint['generator'] = generator.state_dict()
    checkPoint['discriminator'] = discriminator.state_dict()
    if not optimizerGen is None :
        checkPoint['optimizerGen'] = optimizerGen.state_dict()
    if not schedulerGen is None :
        checkPoint['schedulerGen'] = schedulerGen.state_dict()
    if not optimizerDis is None :
        checkPoint['optimizerDis'] = optimizerDis.state_dict()
    if not schedulerDis is None :
        checkPoint['schedulerDis'] = schedulerDis.state_dict()
    checkPoint['resAcc'] = interimRes
    torch.save(checkPoint, path)


def loadCheckPoint(path, generator, discriminator,
                   optimizerGen=None, optimizerDis=None,
                   schedulerGen=None, schedulerDis=None) :
    checkPoint = torch.load(path, map_location=TCfg.device, weights_only=False)
    epoch = checkPoint['epoch']
    iterations = checkPoint['iterations']
    minGEpoch = checkPoint['minGEpoch']
    minGdLoss = checkPoint['minGdLoss']
    startFrom = checkPoint['startFrom'] if 'startFrom' in checkPoint else 0
    generator.load_state_dict(checkPoint['generator'])
    discriminator.load_state_dict(checkPoint['discriminator'])
    if not optimizerGen is None :
        optimizerGen.load_state_dict(checkPoint['optimizerGen'])
    if not schedulerGen is None :
        schedulerGen.load_state_dict(checkPoint['schedulerGen'])
    if not optimizerDis is None :
        optimizerDis.load_state_dict(checkPoint['optimizerDis'])
    if not schedulerDis is None :
        schedulerDis.load_state_dict(checkPoint['schedulerDis'])
    interimRes = checkPoint['resAcc'] if 'resAcc' in checkPoint else TrainResClass()

    return epoch, iterations, minGEpoch, minGdLoss, startFrom, interimRes



trainInfo = TrainInfoClass()
normMSE=1
normL1L=1
normRec=1
skipDis = False

def train_step(images):
    global trainDis, trainGen, eDinfo, noAdv, withNoGrad, skipGen, skipDis
    trainInfo.iterations += 1
    trainInfo.totPerformed += 1
    trainRes = TrainResClass()

    nofIm = images.shape[0]
    images = images.squeeze(1).to(TCfg.device)
    procImages, procReverseData = imagesPreProc(images)
    fakeImages = procImages.clone().detach().requires_grad_(False)
    subBatchSize = nofIm // TCfg.batchSplit
    imWeights = calculateWeights(images)
    normDiff = calculateNorm(images)

    labelsTrue = torch.full((subBatchSize, 1),  1 - TCfg.labelSmoothFac,
                        dtype=torch.float, device=TCfg.device, requires_grad=False)
    labelsFalse = torch.full((subBatchSize, 1),  TCfg.labelSmoothFac,
                        dtype=torch.float, device=TCfg.device, requires_grad=False)
    labelsDis = torch.cat( (labelsTrue, labelsFalse), dim=0).to(TCfg.device).requires_grad_(False)

    # train discriminator
    if not noAdv :

        # calculate predictions of prefilled images - purely for metrics purposes
        #discriminator.eval()
        #generator.eval()
        trainRes.predPre = 0
        with torch.no_grad() :
            for i in range(TCfg.batchSplit) :
                subRange = np.s_[i*subBatchSize:(i+1)*subBatchSize] if TCfg.batchSplit > 1 else np.s_[...]
                fakeImages[subRange,:,DCfg.gapRngX] = generator.preProc(procImages[subRange,...])
                trainRes.predPre += discriminator(fakeImages[subRange,...]).mean().item()
            trainRes.predPre /= TCfg.batchSplit

        pred_real = torch.empty((nofIm,1), requires_grad=False)
        pred_fake = torch.empty((nofIm,1), requires_grad=False)
        #discriminator.train()
        for param in discriminator.parameters() :
            param.requires_grad = True
        optimizer_D.zero_grad()
        for i in range(TCfg.batchSplit) :
            subRange = np.s_[i*subBatchSize:(i+1)*subBatchSize] if TCfg.batchSplit > 1 else np.s_[...]
            subFakeImages = fakeImages[subRange,...]
            with torch.no_grad() :
                subFakeImages[DCfg.gapRng] = generator.generatePatches(procImages[subRange,...])
            with torch.set_grad_enabled(not skipDis) :
                subPred_realD = discriminator(procImages[subRange,...])
                subPred_fakeD = discriminator(subFakeImages)
                pred_both = torch.cat((subPred_realD, subPred_fakeD), dim=0)
                wghts = None if imWeights is None else \
                    torch.cat( (imWeights[subRange], imWeights[subRange]) )
                subD_loss = loss_Adv(labelsDis, pred_both, wghts)
            # train discriminator only if it is not too good :
            if not skipDis and ( subPred_fakeD.mean() > 0.2 or subPred_realD.mean() < 0.8 ) :
                trainInfo.disPerformed += 1/TCfg.batchSplit
                subD_loss.backward()
            trainRes.lossD += subD_loss.item()
            pred_real[subRange] = subPred_realD.clone().detach()
            pred_fake[subRange] = subPred_fakeD.clone().detach()
        optimizer_D.step()
        optimizer_D.zero_grad(set_to_none=True)
        trainRes.lossD /= TCfg.batchSplit
        trainRes.predReal = pred_real.mean().item()
        trainRes.predFake = pred_fake.mean().item()

    else :
        pred_real = torch.zeros((1,), requires_grad=False)
        pred_fake = torch.zeros((1,), requires_grad=False)

    # train generator
    #discriminator.eval()
    for param in discriminator.parameters() :
        param.requires_grad = False
    #generator.train()
    optimizer_G.zero_grad()
    for i in range(TCfg.batchSplit) :
        subRange = np.s_[i*subBatchSize:(i+1)*subBatchSize] if TCfg.batchSplit > 1 else np.s_[:]
        wghts = None if imWeights is None else \
                torch.cat( (imWeights[subRange], imWeights[subRange]) )
        subFakeImages = generator.generateImages(procImages[subRange,...])
        if noAdv :
            subG_loss = loss_Rec( procImages[subRange,:,DCfg.gapRngX], subFakeImages[DCfg.gapRng]
                                , wghts, normalizeRec=normDiff[subRange,...])
            subGD_loss = subGA_loss = subG_loss
        else :
            subPred_fakeG = discriminator(subFakeImages)
            subGA_loss, subGD_loss = loss_Gen(labelsTrue, subPred_fakeG,
                                              procImages[subRange,:,DCfg.gapRngX], subFakeImages[DCfg.gapRng],
                                              wghts, normalizeRec=normDiff[subRange,...])
            subG_loss = lossAdvCoef * subGA_loss + lossDifCoef * subGD_loss
            pred_fake[subRange] = subPred_fakeG.clone().detach()
        # train generator only if it is not too good :
        #if noAdv  or  subPred_fakeD.mean() < pred_real[subRange].mean() :
        if True:
            trainInfo.genPerformed += 1/TCfg.batchSplit
            subG_loss.backward()
        trainRes.lossGA += subGA_loss.item()
        trainRes.lossGD += subGD_loss.item()
        fakeImages[subRange,:,DCfg.gapRngX] = subFakeImages[DCfg.gapRng].detach()
    optimizer_G.step()
    optimizer_G.zero_grad(set_to_none=True)
    trainRes.lossGA /= TCfg.batchSplit
    trainRes.lossGD /= TCfg.batchSplit
    trainRes.predFake = pred_fake.mean().item()


    # prepare report
    with torch.no_grad() :

        trainRes.lossMSE = loss_MSE(images[DCfg.gapRng], fakeImages[DCfg.gapRng]).item() / normMSE
        trainRes.lossL1L = loss_L1L(images[DCfg.gapRng], fakeImages[DCfg.gapRng]).item() / normL1L
        trainRes.lossGD /= normRec

        idx = random.randint(0, nofIm-1) if noAdv else pred_real.argmax()
        trainInfo.bestRealImage = fakeImages[idx,...].clone().detach() if noAdv \
                                  else images[idx,...].clone().detach()
        trainInfo.bestRealProb = 0 if noAdv else pred_real[idx].item()
        trainInfo.bestRealIndex = idx

        idx = random.randint(0, nofIm-1) if noAdv else pred_real.argmin()
        trainInfo.worstRealImage = fakeImages[idx,...].clone().detach() if noAdv \
                                   else images[idx,...].clone().detach()
        trainInfo.worstRealProb = 0 if noAdv else pred_real[idx].item()
        trainInfo.worstRealIndex = idx

        idx = random.randint(0, nofIm-1) if noAdv else pred_fake.argmax()
        trainInfo.bestFakeImage = fakeImages[idx,...].clone().detach()
        trainInfo.bestFakeProb = 0 if noAdv else pred_fake[idx].item()
        trainInfo.bestFakeIndex = idx

        idx = random.randint(0, nofIm-1) if noAdv else pred_fake.argmin()
        trainInfo.worstFakeImage = fakeImages[idx,...].clone().detach()
        trainInfo.worstFakeProb = 0 if noAdv else pred_fake[idx].item()
        trainInfo.worstFakeIndex = idx

        trainInfo.highestDifIndex = eDinfo[0]
        trainInfo.highestDif = eDinfo[1]
        trainInfo.highestDifImageOrg = images[trainInfo.highestDifIndex,...].clone().detach()
        trainInfo.highestDifImageGen = fakeImages[trainInfo.highestDifIndex,...].clone().detach()
        trainInfo.lowestDifIndex = eDinfo[2]
        trainInfo.lowestDif = eDinfo[3]

        trainInfo.ratReal += nofIm * torch.count_nonzero(pred_real > 0.5)/nofIm
        trainInfo.ratFake += nofIm * torch.count_nonzero(pred_fake > 0.5)/nofIm
        trainInfo.totalImages += nofIm

    return trainRes


epoch=initIfNew('epoch', 0)
iter = initIfNew('iter', 0)
imer = initIfNew('iter', 0)
minGEpoch = initIfNew('minGEpoch')
minGdLoss = initIfNew('minGdLoss', 1)
startFrom = initIfNew('startFrom', 0)

def beforeEachEpoch(epoch) :
    return

def afterEachEpoch(epoch) :
    return

def beforeReport() :
    return

def afterReport() :
    return

dataLoader=None
testLoader=None
normTestMSE=1
normTestL1L=1
normTestRec=1
resAcc = TrainResClass()

def train(savedCheckPoint):
    global epoch, minGdLoss, minGEpoch, iter, trainInfo, startFrom, imer, resAcc
    lastGdLoss = minGdLoss
    lastGdLossTrain = 1

    discriminator.to(TCfg.device)
    generator.to(TCfg.device)
    lastUpdateTime = time.time()
    lastSaveTime = time.time()

    while TCfg.nofEpochs is None or epoch <= TCfg.nofEpochs :
        epoch += 1
        beforeEachEpoch(epoch)
        generator.train()
        discriminator.train()
        #resAcc = TrainResClass()
        totalIm = 0

        for it , data in tqdm.tqdm(enumerate(dataLoader), total=int(len(dataLoader))):
            if startFrom :
                startFrom -= 1
                continue
            iter += 1
            images = data[0].to(TCfg.device)
            nofIm = images.shape[0]
            imer += nofIm
            totalIm += nofIm
            trainRes = train_step(images)
            resAcc += trainRes * nofIm
            resAcc.nofIm += nofIm

            #if True:
            #if False :
            #if not it or it > len(dataloader)-2 or time.time() - lastUpdateTime > 60 :
            if time.time() - lastUpdateTime > 60 :

                lastUpdateTime = time.time()

                _,_,_ =  logStep(iter)
                collageR, probsR, _ = generateDiffImages(refImages[[0],...], layout=0)
                showMe = np.zeros( (2*DCfg.sinoSh[1] + DCfg.gapW ,
                                    5*DCfg.sinoSh[0] + 4*DCfg.gapW), dtype=np.float32  )
                for clmn in range (5) : # mark gaps
                    showMe[ DCfg.sinoSh[0] : DCfg.sinoSh[0] + DCfg.gapW ,
                            clmn*(DCfg.sinoSh[1]+DCfg.gapW) + 2*DCfg.gapW : clmn*(DCfg.sinoSh[1]+DCfg.gapW) + 3*DCfg.gapW ] = -1
                def addImage(clmn, row, img, stretch=True) :
                    imgToAdd = img.clone().detach().squeeze()
                    if stretch :
                        minv = imgToAdd.min()
                        ampl = imgToAdd.max() - minv
                        imgToAdd[()] = 2 * ( imgToAdd - minv ) / ampl - 1  if ampl!=0.0 else 0
                    showMe[ row * ( DCfg.sinoSh[1]+DCfg.gapW) : (row+1) * DCfg.sinoSh[1] + row*DCfg.gapW ,
                            clmn * ( DCfg.sinoSh[0]+DCfg.gapW) : (clmn+1) * DCfg.sinoSh[0] + clmn*DCfg.gapW ] = \
                        imgToAdd.cpu().numpy()
                addImage(0,0,trainInfo.bestRealImage)
                addImage(0,1,trainInfo.worstRealImage)
                addImage(1,0,trainInfo.bestFakeImage)
                addImage(1,1,trainInfo.worstFakeImage)
                addImage(2,0,trainInfo.highestDifImageGen)
                addImage(2,1,trainInfo.highestDifImageOrg)
                addImage(3,0,collageR[0,2])
                addImage(3,1,collageR[0,0])
                addImage(4,0,collageR[0,1])
                addImage(4,1,collageR[0,3], stretch=False)
                writer.add_scalars("Losses per iter",
                                   {'Dis': trainRes.lossD
                                   ,'Gen': trainRes.lossGA
                                   ,'Rec': lossAdvCoef * trainRes.lossGA + lossDifCoef * trainRes.lossGD * normRec
                                   }, imer )
                writer.add_scalars("Distances per iter",
                                   {'MSE': trainRes.lossMSE
                                   ,'L1L': trainRes.lossL1L
                                   ,'REC': trainRes.lossGD
                                   }, imer )
                writer.add_scalars("Probs per iter",
                                   {'Ref':trainRes.predReal
                                   ,'Gen':trainRes.predFake
                                   ,'Pre':trainRes.predPre
                                   }, imer )

                IPython.display.clear_output(wait=True)
                beforeReport()
                print(f"Epoch: {epoch} ({minGEpoch}). " +
                      ( f" L1L: {trainRes.lossL1L:.3f} " if noAdv \
                          else \
                        f" Dis[{trainInfo.disPerformed/trainInfo.totPerformed:.2f}]: {trainRes.lossD:.3f} ({trainInfo.ratReal/trainInfo.totalImages:.3f})," ) +
                      ( f" MSE: {trainRes.lossMSE:.3f} " if noAdv \
                          else \
                        f" Gen[{trainInfo.genPerformed/trainInfo.totPerformed:.2f}]: {trainRes.lossGA:.3f} ({trainInfo.ratFake/trainInfo.totalImages:.3f})," ) +
                      f" Rec: {trainRes.lossGD:.3f} (Train: {lastGdLossTrain:.3f}, Test: {lastGdLoss/normTestRec:.3f} | {minGdLoss/normTestRec:.3f})."
                      )
                print (f"TT: {trainInfo.bestRealProb:.2f},  "
                       f"FT: {trainInfo.bestFakeProb:.2f},  "
                       f"HD: {trainInfo.highestDif/normMSE:.3e},  "
                       f"GP: {probsR[0,2].item():.3f}, {probsR[0,1].item():.3f} " )
                print (f"TF: {trainInfo.worstRealProb:.2f},  "
                       f"FF: {trainInfo.worstFakeProb:.2f},  "
                       f"LD: {trainInfo.lowestDif/normMSE:.3e},  "
                       f"R : {probsR[0,0].item():.3f}." )
                plotImage(showMe)
                afterReport()
                trainInfo = TrainInfoClass() # reset for the next iteration

            if time.time() - lastSaveTime > 3600 :
                lastSaveTime = time.time()
                saveCheckPoint(savedCheckPoint+"_hourly.pth",
                               epoch-1, imer, minGEpoch, minGdLoss/normRec,
                               generator, discriminator,
                               optimizer_G, optimizer_D,
                               startFrom=it, interimRes=resAcc)
                saveModels(f"model_{TCfg.exec}_hourly")


        resAcc *= 1.0/totalIm
        writer.add_scalars("Losses per epoch",
                           {'Dis': resAcc.lossD
                           ,'Adv': resAcc.lossGA
                           ,'Gen': lossAdvCoef * resAcc.lossGA + lossDifCoef * resAcc.lossGD
                           }, epoch )
        writer.add_scalars("Distances per epoch",
                           {'MSE': resAcc.lossMSE
                           ,'L1L': resAcc.lossL1L
                           ,'REC': resAcc.lossGD
                           }, epoch )
        writer.add_scalars("Probs per epoch",
                           {'Ref': resAcc.predReal
                           ,'Gen': resAcc.predFake
                           ,'Pre': resAcc.predPre
                           }, epoch )
        lastGdLossTrain = resAcc.lossGD

        Rec_test, MSE_test, L1L_test, Gen_test, Dis_test = summarizeSet(testLoader, False)
        writer.add_scalars("Test per epoch",
                           {'MSE': MSE_test / normTestMSE
                           ,'L1L': L1L_test / normTestL1L
                           ,'REC': Rec_test / normTestRec
                           #,'Dis': Dis_test
                           #,'Gen': Gen_test
                           }, epoch )

        lastGdLoss = Rec_test
        if lastGdLoss < minGdLoss  :
            minGdLoss = lastGdLoss
            minGEpoch = epoch
            saveCheckPoint(savedCheckPoint+"_B.pth",
                           epoch, imer, minGEpoch, minGdLoss,
                           generator, discriminator,
                           optimizer_G, optimizer_D)
            os.system(f"cp {savedCheckPoint}.pth {savedCheckPoint}_BB.pth") # BB: before best
            os.system(f"cp {savedCheckPoint}_B.pth {savedCheckPoint}.pth") # B: best
            saveModels()
        else :
            saveCheckPoint(savedCheckPoint+".pth",
                           epoch, imer, minGEpoch, minGdLoss,
                           generator, discriminator,
                           optimizer_G, optimizer_D)

        resAcc = TrainResClass()
        afterEachEpoch(epoch)


def testMe(tSet, imags=1):
    if isinstance(imags, int) :
        testSubSet = [ tSet.__getitem__() for _ in range(imags) ]
    else :
        testSubSet = [ tSet.__getitem__(index) for index in imags ]
    testImages = torch.stack( [ testItem[0] for testItem in testSubSet ] ).to(TCfg.device)
    colImgs, probs, dists = generateDiffImages(testImages, layout=4)
    testIndeces = []
    for im in range(len(testSubSet)) :
        testItem = testSubSet[im]
        testIndeces.append(testItem[1])
        print(f"Index: ({testItem[1]})")
        print(f"Probabilities. Org: {probs[im,0]:.3e},  Gen: {probs[im,2]:.3e},  Pre: {probs[im,1]:.3e}.")
        print(f"Distances. Rec: {dists[im,0]:.4e},  MSE: {dists[im,1]:.4e},  L1L: {dists[im,2]:.4e}.")
        plotImage(colImgs[im].squeeze().cpu())
    return testIndeces




def freeGPUmem() :
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()






