
import IPython

import sys
import os
import random
import time
import gc
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import pickle

import math
import statistics
from cv2 import norm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torchvision
from torch import optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
#import pytorch_msssim
import ssim

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
    loaderWorkers : int
    labelSmoothFac: float
    learningRateD: float
    learningRateG: float
    device: torch.device = torch.device('cpu')
    batchSplit : int = 1
    nofEpochs: int = 0
    historyHDF : str = field(repr = True, init = False)
    logDir : str = field(repr = True, init = False)
    maximumArtificialShift : int = 4
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
    inShape : tuple = (80,80)
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


def fillWheights(seq) :
    for wh in seq :
        if hasattr(wh, 'weight') :
            #torch.nn.init.xavier_uniform_(wh.weight)
            #torch.nn.init.zeros_(wh.weight)
            #torch.nn.init.constant_(wh.weight, 0)
            #torch.nn.init.uniform_(wh.weight, a=0.0, b=1.0, generator=None)
            torch.nn.init.normal_(wh.weight, mean=0.0, std=0.01)
        if hasattr(wh, 'bias') and wh.bias is not None  :
            torch.nn.init.normal_(wh.bias, mean=0.0, std=0.01)




def set_seed(SEED_VALUE):
    torch.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed_all(SEED_VALUE)
    np.random.seed(SEED_VALUE)


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    return


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=TCfg.device))
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


class SamplingMask :
    def __init__(self):
        self.mask = loadImage("sampling_mask.tif")
        mn = self.mask.min()
        mx = self.mask.max()
        if mn < mx :
            self.mask = (self.mask - mn) / (mx - mn)
        else :
            self.mask = 1
        imask = fn.conv2d(torch.from_numpy(self.mask)[None, None,...],
                          torch.ones((1, 1,*DCfg.inShape)))[0,0,...].numpy() / math.prod(DCfg.inShape)
        #imask = np.convolve(self.mask, kernel, mode='valid')
        self.indicies = np.argwhere( np.logical_and(imask < 0.9, imask > 0.5) ).astype(int)
    def __len__(self) :
        return len(self.indicies)
    def __getitem__(self, index=None) :
        if index is None :
            index = random.randint(0, self.__len__()-1)
        else:
            index = index % self.__len__()
        idx = self.indicies[index]
        return self.mask[idx[0]:idx[0]+DCfg.inShape[-2], idx[1]:idx[1]+DCfg.inShape[-1]]
samplingMask = initIfNew('samplingMask')


class SamplingVariations :
    def __init__(self, amplitude=0.2):
        self.amplitude = amplitude
        self.bgvar = loadImage("BGvariations.tif")
        mn = self.bgvar.min()
        mx = self.bgvar.max()
        if mn < mx :
            self.bgvar = 2 * (self.bgvar - mn) / (mx - mn) - 1
        else :
            self.bgvar = 0
        self.samplingShape = (self.bgvar.shape[0] - DCfg.inShape[0],
                              self.bgvar.shape[1] - DCfg.inShape[1])
    def __len__(self) :
        return math.prod(self.samplingShape)
    def __getitem__(self, index=None) :
        constFactor = self.getConstComponent( random.randint(0, self.__len__()-1) \
                                              if index is None else hashAnObject(index) )
        if index is None :
            index = random.randint(0, self.__len__()-1) % self.__len__()
        idx = divmod( index % self.__len__(), self.samplingShape[1] ) \
              if isinstance(index, int) else index
        vari = self.bgvar[idx[0]:idx[0]+DCfg.inShape[-2],
                          idx[1]:idx[1]+DCfg.inShape[-1]]
        return constFactor * (1+vari*self.amplitude)
    def getConstComponent(self, index=None) :
        if index is None :
            index = random.randint(0, self.__len__()-1)
            return self.getConstComponent(index)
        else :
            index = index % self.__len__()
        rat = 2 * index / self.__len__() - 1 # [-1,1]
        return 2 ** rat # [1/2,2]
samplingVari = initIfNew('samplingVari')




def hdfData(inputString):
    nameSplit = inputString.split(':')
    if len(nameSplit) != 2 :
        raise Exception(f"String \"{inputString}\" does not represent an HDF5 format \"fileName:container\".")
    hdfName = nameSplit[0]
    hdfVolume = nameSplit[1]
    try :
        trgH5F =  h5py.File(hdfName,'r')
    except :
        raise Exception(f"Failed to open HDF file '{hdfName}'.")
    if  hdfVolume not in trgH5F.keys():
        raise Exception(f"No dataset '{hdfVolume}' in input file {hdfName}.")
    data = trgH5F[hdfVolume]
    if not data.size :
        raise Exception(f"Container \"{inputString}\" is zero size.")
    sh = data.shape
    if len(sh) != 3 :
        raise Exception(f"Dimensions of the container \"{inputString}\" is not 3: {sh}.")
    return data


def hashAnObject(object) :
    digest = hashlib.sha256(pickle.dumps(object)).digest()
    return int.from_bytes(digest, 'big')


##################################################
#       PAIRS
##################################################

class ShiftedPair :

    def __init__(self, orgVol, sftVol, shifts, orgMask=None, sftMask=None, randomize=False) :
        self.orgData = hdfData(orgVol)
        self.shape = self.orgData.shape
        self.face = self.shape[1:]
        self.imgs = self.shape[0]
        self.sftData = hdfData(sftVol)
        if self.sftData.shape != self.shape :
            raise Exception( "Shape mismatch of shifted volume:"
                            f" {self.sftData.shape} != {self.shape}.")
        self.shifts = np.loadtxt(shifts, dtype=int)
        if self.shifts.shape != (self.imgs, 2) :
            raise Exception( "Shape mismatch of shifts:"
                            f" {self.shifts.shape} != {(self.imgs, 2)}.")
        def cookMask(maskName) :
            if maskName is None:
                mask =  np.ones(self.face)
            else :
                mask = loadImage(maskName, self.face)
                mn = mask.min()
                mx = mask.max()
                if mn < mx :
                    mask = (mask - mn) / (mx - mn)
                else :
                    mask = np.ones(self.face)
            imask = fn.conv2d(torch.from_numpy(mask)[None, None,...],
                              torch.ones((1, 1,*DCfg.inShape))) [0,0,...].numpy() \
                                                                / math.prod(DCfg.inShape)
            # next is to allow for artificial shift
            imask[0:TCfg.maximumArtificialShift, ...] = 0
            imask[-1-TCfg.maximumArtificialShift:, ...] = 0
            imask[...,0:TCfg.maximumArtificialShift] = 0
            imask[...,-1-TCfg.maximumArtificialShift:] = 0
            return mask, imask
        self.orgMask, self.orgImask = cookMask(orgMask)
        self.sftMask, self.sftImask = cookMask(sftMask)
        self.iFace = self.orgImask.shape
        self.goodForTraining = np.argwhere( np.logical_and(self.orgImask > 0.99,
                                                           self.sftImask > 0.99) )


        self.prehash = hashAnObject((orgVol, sftVol, orgMask, sftMask))
        self.randomize = randomize


    def __len__(self):
        return self.goodForTraining.shape[0] * self.imgs


    def __getitem__(self, index=None) :
        if isinstance(index, tuple) and len(index) == 3 :
            zdx, ydx, xdx = index
        elif isinstance(index, int) :
            if index >= self.__len__() :
                raise Exception(f"Index {index} is out of range for shifted pair"
                                f" of size {self.__len__()}.")
            zdx, pos = divmod(index, self.goodForTraining.shape[0])
            ydx, xdx = self.goodForTraining[ pos, : ]
        elif index is None :
            zdx = random.randint(0,self.imgs-1)
            ydx, xdx = self.goodForTraining[ random.randint(0, self.goodForTraining.shape[0]-1) ]
        else :
            raise Exception(f"Bad index type {type(index)} {index} shifted pair.")
        zdx, ydx, xdx = int(zdx), int(ydx), int(xdx) # to make sure hashes are same irrespectively of those variable types
        hashOrg = None if self.randomize else hashAnObject( (0, self.prehash, zdx, ydx, xdx) )
        hashSft = None if self.randomize else hashAnObject( (1, self.prehash, zdx, ydx, xdx) )
        #xShift = random.randint(-TCfg.maximumArtificialShift, TCfg.maximumArtificialShift) \
        #            if self.randomize else \
        #         hashOrg % (2*TCfg.maximumArtificialShift+1) - TCfg.maximumArtificialShift
        #yShift = random.randint(-TCfg.maximumArtificialShift, TCfg.maximumArtificialShift) \
        #            if self.randomize else \
        #         hashSft % (2*TCfg.maximumArtificialShift+1) - TCfg.maximumArtificialShift
        yShift, xShift = tuple(self.shifts[zdx])
        range = np.s_[ydx:ydx+DCfg.inShape[-2],
                      xdx:xdx+DCfg.inShape[-1]]
        shiftedRange = np.s_[ydx-yShift:ydx-yShift+DCfg.inShape[-2],
                             xdx-xShift:xdx-xShift+DCfg.inShape[-1]]
        orgSubMask = self.orgMask[range]
        sftSubMask = self.sftMask[shiftedRange]
        orgTrainMask = samplingMask[hashOrg]
        sftTrainMask = samplingMask[hashSft]
        orgTrainVari = samplingVari[hashOrg]
        sftTrainVari = samplingVari[hashSft]
        data = np.stack([ self.orgData[zdx, *range] * orgTrainVari,
                          self.sftData[zdx, *shiftedRange] * sftTrainVari,
                          orgSubMask * orgTrainMask, sftSubMask * sftTrainMask,
                          orgSubMask, sftSubMask ])
        return data, (zdx, ydx, xdx)


    def masks(self) :
        return self.orgMask, self.sftMask



class ManyShiftedPairs :

    def __init__(self, listOfPairs, randomize=False) :
        self.pairs = []
        for pair in listOfPairs :
            self.pairs.append(ShiftedPair(*pair, randomize=randomize))
        self.randomize=randomize

    def __len__(self):
        return sum( [ len(pair) for pair in self.pairs ] )

    def __getitem__(self, index=None):

        if isinstance(index, tuple) and len(index) == 4 :
            return self.pairs[index[0]].__getitem__(index[1:])[0], index
        elif isinstance(index, int) :
            if index >= self.__len__() :
                raise Exception(f"Index {index} is out of range for collection of length {self.__len__()}.")
            tail = index
            curPair = 0
            while tail >= len(self.pairs[curPair]) and curPair < len(self.pairs) :
                tail -= len(self.pairs[curPair])
                curPair += 1
            data, index = self.pairs[curPair].__getitem__(tail)
            return data, (curPair, *index)
        elif index is None :
            return self.__getitem__( random.randint(0,self.__len__()-1) )
        else :
            raise Exception(f"Bad index type {type(index)} {index} for collection of pairs.")


    def get_dataset(self, transform=None) :

        class InputFromPairs(torch.utils.data.Dataset) :
            def __init__(self, root, transform=None):
                self.container = root
                self.transform = transform
            def __len__(self):
                return sum( [ len(pair) for pair in self.container.pairs ] )
            def __getitem__(self, index=None, doTransform=True):
                data, index = self.container.__getitem__(index) \
                    if index is None or isinstance(index, tuple) else \
                        self.container[int(index)]
                data = torch.tensor(data)
                if doTransform and self.transform :
                    data = self.transform(data)
                return (data, index)
            def flatIndex(index) :
                idx = 0
                pair = None
                for curPair in range(index[0]) :
                    pair = self.container.pairs[curPair]
                    idx += len(pair)
                idx += pair.imgs * index[1]
                item = 0
                for item in range(pair.goodForTraining.shape[0]) :
                    if np.all( pair.goodForTraining[item,:] == index[2:] ) :
                        return idx + item
                return None # not found

        return InputFromPairs(self, transform)


def createTrimage(itemSet, it=None) :
    if it is None :
        return torch.stack( [ createTrimage(itemSet, 0), createTrimage(itemSet, 1) ] )
    masks = itemSet[2:,...]
    mn, mx = itemSet[0+it,...].min(), itemSet[0+it,...].max()
    mn, mx = torch.where( masks[0+it,...] > 0, itemSet[0+it,...], mx ).min(), \
             torch.where( masks[0+it,...] > 0, itemSet[0+it,...], mn ).max()
    return   torch.where( masks[0+it,...] > 0, itemSet[0+it,...],
                          mn + 0.5 * (mx-mn) * masks[1-it,...] *
                          ( masks[2+it,...] * masks[3-it,...] + masks[1-it,...]) )


dataRoot = "/mnt/hddData/shiftpatch/"
TestShiftedPairs = [ [ dataRoot + prefix + postfix
                       for postfix in ["_org.hdf:/data",
                                       "_sft.hdf:/data",
                                       "_pairWiseShifts.txt",
                                       "_org_mask.tif",
                                       "_sft_mask.tif",
                                       ] ]
                         for prefix in [ "01_dir", "01_flp" ] ]
TrainShiftedPairs = [ [ dataRoot + prefix + postfix
                       for postfix in ["_org.hdf:/data",
                                       "_sft.hdf:/data",
                                       "_pairWiseShifts.txt",
                                       "_org_mask.tif",
                                       "_sft_mask.tif",
                                       ] ]
                         for prefix in [ "02_dir", "02_flp",
                                         "03_dir", "03_flp" ] ]
examples = [
    (1, 924, 315, 1580),
    (1, 534, 733, 1298),
    (1, 744, 23, 23),
    (1, 772, 121, 1750)
]

dataMeanNorm = (0.5,0.5,0,0,0,0) # masks not to be normalized


def createSet( pairs, randomize) :
    setRoot = ManyShiftedPairs(pairs, randomize=randomize)
    mytransforms = \
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=dataMeanNorm, std=(1)), ]) \
        if randomize else \
        transforms.Compose([
            transforms.Normalize(mean=dataMeanNorm, std=(1)) ])
    return setRoot.get_dataset(mytransforms)

def createTrainSet() :
    return createSet(TrainShiftedPairs, True)
trainSet = initIfNew('trainSet')

def createTestSet() :
    return createSet(TestShiftedPairs, False)
testSet = initIfNew('testSet')

def createSubSet(tSet, subSetSize=None) :
    if subSetSize is None :
        subSetSize = len(tSet) // math.prod(DCfg.inShape)
    subset_indices = \
        torch.randint( 0, len(tSet), (subSetSize,) ) \
            if tSet.container.randomize else \
        torch.tensor( [ hashAnObject(idx) % len(tSet) for idx in range (subSetSize) ]  )
    return torch.utils.data.Subset(tSet, subset_indices)

def createDataLoader(tSet, num_workers=os.cpu_count()) :
    return torch.utils.data.DataLoader(
        dataset=tSet,
        batch_size=TCfg.batchSize,
        shuffle=False, # no need to shuffle because already done when creating subsets
        num_workers=num_workers,
        drop_last=True
    )


def createReferences(toShow = 0) :
    global examples
    if toShow :
        examples.insert(0, examples.pop(toShow))
    refImages = torch.stack( [ testSet.__getitem__(ex)[0] for ex in examples ] ).to(TCfg.device)
    refNoises = torch.randn((refImages.shape[0],TCfg.latentDim)).to(TCfg.device)
    return refImages, refNoises
refImages = initIfNew('refImages')
refNoises = initIfNew('refNoises')


def showMe(tSet, item=None) :
    global refImages, refNoises
    image = None
    if item is None :
        image, index = tSet[random.randint(0,len(tSet)-1)]
        print (f"{index}")
        #while True:
        #    image, index = tSet[random.randint(0,len(tSet)-1)]
        #    if image[0].mean() > 0 and image[0].min() < -0.1 :
        #        print (f"{index}")
        #        break
    elif isinstance(item, int) :
        image = refImages[item,...]
    else :
        image, _ = tSet.__getitem__(item)
    trImage = createTrimage(image)
    image = image.squeeze()
    plotImages( [image[0].cpu(), image[1].cpu(), trImage[0].cpu(), trImage[1].cpu()] )
    #plotImages( [image[0].cpu(), image[2].cpu(), image[4].cpu()] )
    #plotImages( [image[1].cpu(), image[3].cpu(), image[5].cpu()] )
    image = image.to(TCfg.device)


save_interim = None

##################################################
#       GENERATOR
##################################################
class GeneratorTemplate(nn.Module):

    def __init__(self, latentChannels=0):
        super(GeneratorTemplate, self).__init__()
        self.latentChannels = latentChannels
        self.baseChannels = 64
        #self.amplitude = 4


    def createLatent(self) :
        if self.latentChannels == 0 :
            return None
        realLatent = abs(self.latentChannels)
        toRet =  nn.Sequential(
            nn.Linear(TCfg.latentDim, math.prod(DCfg.inShape) * realLatent),
            nn.ReLU(),
            nn.Unflatten( 1, (realLatent,) + DCfg.inShape )
        )
        fillWheights(toRet)
        return toRet


    def encblock(self, chIn, chOut, kernel, stride=1, norm=False, dopadding=False) :
        chIn = int(chIn*self.baseChannels)
        chOut = int(chOut*self.baseChannels)
        layers = []
        layers.append( nn.Conv2d(chIn, chOut, kernel, stride=stride, bias = not norm,
                                padding='same', padding_mode='reflect') \
                                if stride == 1 and dopadding else \
                                nn.Conv2d(chIn, chOut, kernel, stride=stride, bias= not norm )
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
        layers.append( nn.ConvTranspose2d(chIn, chOut, kernel, stride, bias=not norm,
                                          padding=1) \
                       if stride == 1 and dopadding else \
                       nn.ConvTranspose2d(chIn, chOut, kernel, stride, bias = not norm)
                     )
        if norm :
            layers.append(nn.BatchNorm2d(chOut))
        layers.append(nn.LeakyReLU(0.2))
        fillWheights(layers)
        return torch.nn.Sequential(*layers)


    def createFClink(self) :
        smpl = torch.zeros((1, 4+abs(self.latentChannels), *DCfg.inShape))
        with torch.no_grad() :
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
            nn.Conv2d(self.baseChannels+4, 2, 1),
            nn.Tanh(),
        )
        fillWheights(toRet)
        return toRet


    def preProc(self,images) :
        with torch.no_grad() :
            orgDims = images.dim()
            if orgDims == 3 :
                images = images.view(1, *images.shape)
            images[:,0,...] *= images[:,2,...]
            images[:,1,...] *= images[:,3,...]
        return images, orgDims


    def postProc(self, images, procData=None) :
        if procData == 3 :
            images = images.squeeze(0)
        return images


    def forward(self, input):
        global save_interim

        if not save_interim is None :
            save_interim = {}
        def saveToInterim(key, data) :
            if not save_interim is None :
                save_interim[key] = data.clone().detach()

        images, noises = input
        saveToInterim('input', images )
        images, procInf = self.preProc(images)
        #return self.postProc(images[:,[1,0],...], procInf)

        if self.latentChannels :
            latent = self.noise2latent(noises) \
                if self.latentChannels > 0 else \
                     2 * torch.rand((images.shape[0], -self.latentChannels, *DCfg.inShape),
                                device = TCfg.device) - 1
            dwTrain = [torch.cat((images, latent), dim=1),]
        else :
            dwTrain = [images,]
        for encoder in self.encoders :
            dwTrain.append(encoder(dwTrain[-1]))
        mid = self.fcLink(dwTrain[-1])
        upTrain = [mid]
        for level, decoder in enumerate(self.decoders) :
            upTrain.append( decoder( torch.cat( (upTrain[-1], dwTrain[-1-level]), dim=1 ) ) )
        res = self.lastTouch(torch.cat( (upTrain[-1], images ), dim=1 ))

        res = self.postProc(res, procInf)
        saveToInterim('output', res)
        return res


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
        encSh = self.body(torch.zeros((1,2,*DCfg.inShape))).shape
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


def saveModels(path="") :
    save_model(generator, model_path = ( path if path else f"model_{TCfg.exec}" ) + "_gen.pt" )
    save_model(discriminator, model_path = ( path if path else f"model_{TCfg.exec}" ) + "_dis.pt"  )


BCE = nn.BCELoss(reduction='none')
MSE = nn.MSELoss(reduction='none')
L1L = nn.L1Loss(reduction='none')
SSIM = ssim.SSIM(data_range=2.0, size_average=False, channel=1, win_size=3)
MSSSIM = ssim.MS_SSIM(data_range=2.0, size_average=False, channel=1, win_size=3)
lossDifCoef = 0
lossAdvCoef = 1.0

[
#def applyWeights(inp, weights, storePerIm=None):
#    inp = inp.squeeze()
#    if not inp.dim() :
#        inp = inp.unsqueeze(0)
#    sum = len(inp)
#    if not weights is None :
#        inp *= weights
#        sum = weights.sum()
#    if storePerIm is not None : # must be list
#        storePerIm.extend(inp.tolist())
#    return inp.sum()/sum
]



##################################################
#       LOSSES
##################################################
def loss_Adv(y_true, y_pred, weights=None, storePerIm=None):
    return BCE(y_pred, y_true)

def loss_MSE(p_true, p_pred, masks):
    toRets = masks[:,2,...] * masks[:,1,...] * (1-masks[:,0,...]) * \
             MSE(p_true[:,0,...], p_pred[:,0,...]) + \
             masks[:,3,...] * masks[:,0,...] * (1-masks[:,1,...]) * \
             MSE(p_true[:,1,...], p_pred[:,1,...])
    return toRets.sum()#dim=(-1,-2))

def loss_L1L(p_true, p_pred, masks):
    toRets = masks[:,2,...] * masks[:,1,...] * (1-masks[:,0,...]) * \
             L1L(p_true[:,0,...], p_pred[:,0,...]) + \
             masks[:,3,...] * masks[:,0,...] * (1-masks[:,1,...]) * \
             L1L(p_true[:,1,...], p_pred[:,1,...])
    return toRets.sum()#dim=(-1,-2))

def ssims_common(p_true, p_pred, masks) :
    blind_spots = (masks[:,0,...] + masks[:,1,...]) > 0
    tMasks = masks[:,[2,3],...] * blind_spots[:,None,:,:] > 0
    pixes = tMasks.sum(dim=(-1,-2))
    means = (p_true * tMasks).sum(dim=(-1,-2)) * torch.where(pixes>0, 1/pixes, 0)
    means = means[:,:,None,None]
    pp_true = torch.cat( ( torch.where( tMasks[:,0,...] > 0, p_true[:,0,...],
                                        means[:,0,...]),
                           torch.where( tMasks[:,1,...] > 0, p_true[:,1,...],
                                        means[:,1,...])),
                         dim=0 ).unsqueeze(1)
    pp_pred = torch.cat( ( torch.where( masks[:,0,...] > 0, p_true[:,0,...],
                                        torch.where(blind_spots, p_pred[:,0,...],
                                                    means[:,0,...] ) ),
                           torch.where( masks[:,1,...] > 0, p_true[:,1,...],
                                        torch.where(blind_spots, p_pred[:,1,...],
                                                    means[:,1,...]) )),
                         dim=0 ).unsqueeze(1)
    counts = pixelsCounted(masks)
    counts = torch.cat( ( counts[:,0], counts[:,1] ), dim=0 )
    blind_spots = torch.cat((blind_spots, blind_spots),dim=0 ).unsqueeze(1)
    return blind_spots, pp_true, pp_pred, counts

def loss_SSIM(p_true, p_pred, masks):
    blind_spots, pp_true, pp_pred, counts = ssims_common(p_true, p_pred, masks)
    dissim = (1 - SSIM( pp_true+0.5, pp_pred+0.5 ) ) / 2
    #dissim = (1 - SSIM( pp_true+0.5, pp_pred+0.5, blind_spots ) ) / 2 # very slow
    dissimSum = (dissim*counts).sum()
    return dissimSum


def loss_MSSSIM(p_true, p_pred, masks):
    _, pp_true, pp_pred, counts = ssims_common(p_true, p_pred, masks)
    dissim = (1 - MSSSIM( pp_true+0.5, pp_pred+0.5 ) ) / 2
    dissimSum = (dissim*counts).sum()
    return dissimSum

SSIM_MSE = 1
def loss_Rec(p_true, p_pred, masks):
    loss = ( 1 - SSIM_MSE ) * loss_MSE(p_true, p_pred, masks) + \
           SSIM_MSE * loss_SSIM(p_true, p_pred, masks) * (normTestMSE/normTestSSIM)
    return loss



def pixelsCounted(masks) :
    toRets = torch.stack((masks[:,2,...] * masks[:,1,...] * (1-masks[:,0,...]),
                          masks[:,3,...] * masks[:,0,...] * (1-masks[:,1,...])),
                          dim=1)
    return toRets.sum(dim=(-1,-2))


def loss_Gen(y_true, y_pred, p_true, p_pred):
    lossAdv = loss_Adv(y_true, y_pred)
    lossDif = loss_Rec(p_pred, p_true)
    return lossAdv, lossDif


def summarizeSet(dataloader):

    MSE_diff, L1L_diff, Rec_diff = 0, 0, 0 #[], [], []
    Real_probs, Fake_probs, GA_losses, GD_losses, D_losses = [], [], [], [], []
    totalPixels = 0
    generator.to(TCfg.device)
    generator.eval()
    if discriminator is not None :
        discriminator.eval()
    with torch.no_grad() :
        for it , data in tqdm.tqdm(enumerate(dataloader), total=int(len(dataloader))):

            images = data[0].to(TCfg.device)
            nofIm = images.shape[0]
            fakeImages = torch.empty( (nofIm, 2, *DCfg.inShape) , device=TCfg.device)
            subBatchSize = nofIm // TCfg.batchSplit

            masks = images[:,2:,...]
            for i in range(TCfg.batchSplit) :
                subRange = np.s_[i*subBatchSize:(i+1)*subBatchSize] if TCfg.batchSplit > 1 else np.s_[:]
                fakeImages[subRange,...] = generator.forward((images[subRange,0:4,...],None))
                if not noAdv :
                    pass
                    #subRprobs = discriminator(subProcImages)
                    #rprobs[subRange,...] = subRprobs
                    #rprob += subRprobs.sum().item()
                    #subFprobs = discriminator(genImages[subRange,...])
                    #fprobs[subRange,...] = subFprobs
                    #fprob += subFprobs.sum().item()

            totalPixels += pixelsCounted(masks).sum()
            MSE_diff += loss_MSE( images[:,0:2,...], fakeImages, masks ).item()
            L1L_diff += loss_L1L( images[:,0:2,...], fakeImages, masks ).item()
            Rec_diff += loss_Rec( images[:,0:2,...], fakeImages, masks ).item()
            if not noAdv :
                pass
                #labelsTrue = torch.full((nofIm, 1),  1 - TCfg.labelSmoothFac,
                #            dtype=torch.float, device=TCfg.device, requires_grad=False)
                #labelsFalse = torch.full((nofIm, 1),  TCfg.labelSmoothFac,
                #            dtype=torch.float, device=TCfg.device, requires_grad=False)
                #labelsDis = torch.cat( (labelsTrue, labelsFalse), dim=0).to(TCfg.device).requires_grad_(False)
                #subD_loss = loss_Adv(labelsDis, torch.cat((rprobs, fprobs), dim=0))
                #subGA_loss, subGD_loss = loss_Gen(labelsTrue, fprobs,
                #                                  images[DCfg.gapRng], procImages[DCfg.gapRng],
                #                                  normalizeRec=calculateNorm(images))
                #D_losses.append( nofIm * subD_loss )
                #GA_losses.append( nofIm * subGA_loss )
                #GD_losses.append( nofIm * subGD_loss )
                #Real_probs.append(rprob)
                #Fake_probs.append(fprob)

    MSE_diff *= 1/totalPixels if totalPixels else 0
    L1L_diff *= 1/totalPixels if totalPixels else 0
    Rec_diff *= 1/totalPixels if totalPixels else 0
    Real_prob = 0 #sum(Real_probs) / totalNofIm if not noAdv else 0
    Fake_prob = 0 #sum(Fake_probs) / totalNofIm if not noAdv else 0
    D_loss  = 0 #sum(D_losses) / totalNofIm if not noAdv else 0
    GA_loss = 0 #sum(GA_losses) / totalNofIm if not noAdv else 0
    GD_loss = 0 #sum(GD_losses) / totalNofIm if not noAdv else 0

    print (f"Summary. Rec: {Rec_diff:.3e}, MSE: {MSE_diff:.3e}, L1L: {L1L_diff:.3e}, Dis: {Real_prob:.3e}, Gen: {Fake_prob:.3e}.")
    return Rec_diff, MSE_diff, L1L_diff, Real_prob, Fake_prob, D_loss, GA_loss, GD_loss


def testMe(tSet, item=None, plotMe=True) :
    images = None
    if isinstance(tSet, torch.Tensor) :
        images = tSet
    elif item is None :
        images, index = tSet[ random.randint(0,len(tSet)-1) ]
        if plotMe :
            print (f"{index}")
        #while True:
        #    image, index = tSet[random.randint(0,len(tSet)-1)]
        #    if image[0].mean() > 0 and image[0].min() < -0.1 :
        #        print (f"{index}")
        #        break
    elif isinstance(item, int) :
        images = refImages[item,...]
    elif isinstance(item, tuple) :
        images, _ = tSet.__getitem__(item)
    elif isinstance(item, list) :
        images = torch.empty((len(item),6,*DCfg.inShape), device = TCfg.device)
        for idx , index in enumerate(item) :
            images[idx,...], _ = tSet.__getitem__(index)
    else :
        raise Exception(f"Unexpected input type {type(item)}." )
    images = images.to(TCfg.device)
    orgDim = images.dim()
    if orgDim == 3 :
        images = images.unsqueeze(0)
    generatedImages = torch.empty_like(images)
    generator.eval()
    masks = images[:,2:,...]
    with torch.no_grad() :
        generatedImages[:,0:2,...] = generator.forward( (images[:,0:4,...], None) )
        totalPixels = pixelsCounted(masks).sum()
        MSE_diff , L1L_diff , Rec_diff = \
            ( loss_func( images[:,0:2,...], generatedImages[:,0:2,...], masks ).item() * \
                (1/totalPixels if totalPixels else 0) \
                    for loss_func in [loss_MSE, loss_L1L, loss_Rec ] )
        mn = torch.where( masks[:,0:2,...] > 0 , images[:,0:2,...], images.max() ).amin(dim=(2,3))
        generatedImages[:,0:2,...] = masks[:,0:2,...] * images[:,0:2,...] + \
                          ( 1-masks[:,0:2,...] ) * \
                          ( masks[:,[1,0],...] * generatedImages[:,0:2,...] + \
                            ( 1 - masks[:,[1,0],...] )  * mn[:,:,None,None] )
        generatedImages[:,2:4,...] = torch.where( (masks[:,[0],...] + masks[:,[1],...]) > 0,
                                                  images[:,0:2,...], mn[:,:,None,None] )
    auxImages = images[:,[0,1],...] * masks[:,[0,1],...] + \
                images[:,[1,0],...] * masks[:,[1,0],...] * (1-masks[:,[0,1],...])
    auxMasks  = masks[:,[0,1],...] + masks[:,[1,0],...] * (1-masks[:,[0,1],...])
    totMax = auxImages.max()
    auxMins = torch.where( auxMasks > 0, auxImages, totMax ).amin(dim=(2,3))
    auxImages = torch.where(auxMasks < 1, auxMins[:,:,None,None], auxImages)

    auxImages = auxImages.cpu()
    images = images.cpu()
    if plotMe :
        print(f"Losses: Rec {Rec_diff}, MSE {MSE_diff}, L1L {L1L_diff}. Pixels: {totalPixels}.")
        for idx in range(images.shape[0]) :
            trImages = createTrimage(images[idx,...])
            plotImages( [generatedImages[idx,0].cpu(), auxImages[idx,0].cpu(),
                         trImages[0,...], images[idx,0] ] )
            plotImages( [generatedImages[idx,1].cpu(), auxImages[idx,1].cpu(),
                         trImages[1,...], images[idx,1] ] )
    if orgDim == 3 :
        generatedImages = generatedImages.squeeze(0)

    return generatedImages, totalPixels, (Rec_diff, MSE_diff, L1L_diff), auxImages

[
#def generateDiffImages(images, layout=None) :
#    images, orgDim = unsqeeze4dim(images)
#    dif = torch.zeros((images.shape[0], 1, *DCfg.sinoSh))
#    hGap = DCfg.gapW // 2
#    pre = images.clone()
#    gen = images.clone()
#    with torch.no_grad() :
#        wghts = calculateWeights(images)
#        generator.eval()
#        pre[DCfg.gapRng] = generator.preProc(images)
#        gen[DCfg.gapRng] = generator.generatePatches(images)
#        dif[DCfg.gapRng] = (gen - pre)[DCfg.gapRng]
#        dif[...,hGap:hGap+DCfg.gapW] = (images - pre)[DCfg.gapRng]
#        dif[...,-DCfg.gapW-hGap:-hGap] = (images - gen)[DCfg.gapRng]
#        for curim in range(images.shape[0]) :
#            if ( cof := max(-dif[curim,...].min(),dif[curim,...].max()) ) != 0 :
#                dif[curim,...] /= cof
#            else :
#                dif[curim,...] = 0
#        probs = torch.empty(images.shape[0],3)
#        dists = torch.empty(images.shape[0],3)
#        #discriminator.eval()
#        probs[:,0] = discriminator(images)[:,0]
#        probs[:,1] = discriminator(pre)[:,0]
#        probs[:,2] = discriminator(gen)[:,0]
#        dists[:,0] = loss_Rec(images[DCfg.gapRng], gen[DCfg.gapRng], wghts, normalizeRec=calculateNorm(images))
#        dists[:,1] = loss_MSE(images[DCfg.gapRng], gen[DCfg.gapRng], wghts)
#        dists[:,2] = loss_L1L(images[DCfg.gapRng], gen[DCfg.gapRng], wghts)
#
#    simages = None
#    if not layout is None :
#        def stretch(stretchme, mm, aa) :
#            return ( stretchme - mm ) * 2 / aa - 1 if ampl > 0 else stretchme * 0
#        simages = images.clone()
#        for curim in range(images.shape[0]) :
#            rng = np.s_[curim,...]
#            minv = min(images[rng].min(), pre[rng].min(), gen[rng].min()).item()
#            ampl = max(images[rng].max(), pre[rng].max(), gen[rng].max()).item() - minv
#            simages[rng] = stretch(simages[rng], minv, ampl)
#            pre[rng] = stretch(pre[rng], minv, ampl)
#            gen[rng] = stretch(gen[rng], minv, ampl)
#
#    cGap = DCfg.gapW
#    if layout == 0 :
#        collage = torch.empty(images.shape[0], 4, *DCfg.sinoSh)
#        collage[:,0,...] = simages[:,0,...]
#        collage[:,1,...] = pre[:,0,...]
#        collage[:,2,...] = gen[:,0,...]
#        collage[:,3,...] = dif[:,0,...]
#    elif layout == 2 :
#        collage = torch.zeros((images.shape[0], 1, DCfg.sinoSh[0]*2 + cGap, DCfg.sinoSh[1]*2 + cGap ))
#        collage[..., :DCfg.sinoSh[0], :DCfg.sinoSh[1]] = gen
#        collage[..., :DCfg.sinoSh[0], DCfg.sinoSh[1]+cGap:] = pre
#        collage[..., DCfg.sinoSh[0]+cGap:, :DCfg.sinoSh[1]] = simages
#        collage[..., DCfg.sinoSh[0]+cGap:, DCfg.sinoSh[1]+cGap:] = dif
#    elif layout == 4 :
#        collage = torch.zeros((images.shape[0], 1, DCfg.sinoSh[0], 4*DCfg.sinoSh[1] + 3*cGap))
#        collage[..., :DCfg.sinoSh[1]] = simages
#        collage[..., DCfg.sinoSh[1]+cGap:2*DCfg.sinoSh[1]+cGap] = gen
#        collage[..., 2*DCfg.sinoSh[1]+2*cGap:3*DCfg.sinoSh[1]+2*cGap] = dif
#        collage[..., 3*DCfg.sinoSh[1]+3*cGap:4*DCfg.sinoSh[1]+4*cGap] = pre
#    elif layout == -4 :
#        collage = torch.zeros( (images.shape[0], 1, 4*DCfg.sinoSh[0] + 3*cGap, DCfg.sinoSh[1]))
#        collage[... , :DCfg.sinoSh[0] , : ] = simages
#        collage[... , DCfg.sinoSh[0]+cGap:2*DCfg.sinoSh[0]+cGap , :] = gen
#        collage[... , 2*DCfg.sinoSh[0]+2*cGap:3*DCfg.sinoSh[0]+2*cGap , : ] = dif
#        collage[... , 3*DCfg.sinoSh[0]+3*cGap:4*DCfg.sinoSh[0]+4*cGap , : ] = pre
#    else :
#        collage = dif
#    collage = squeezeOrg(collage,orgDim)
#
#    return collage, probs, dists
#
#
#def logStep(iter, write=True) :
#    colImgs, probs, dists = generateDiffImages(refImages, layout=-4)
#    probs = probs.mean(dim=0)
#    dists = dists.mean(dim=0)
#    colImgs = colImgs.squeeze()
#    cSh = colImgs.shape
#    gapH = DCfg.gapW
#    collage = np.zeros( ( cSh[-2], cSh[0]*cSh[-1] + (cSh[0]-1)*gapH ), dtype=np.float32  )
#    for curI in range(cSh[0]) :
#        collage[ : , curI * (cSh[-1]+gapH) : curI * (cSh[-1]+gapH) + cSh[-1]] = colImgs[curI,...]
#    #writer.add_scalars("Probs of ref images",
#    #                   {'Ref':probs[0]
#    #                   ,'Gen':probs[2]
#    #                   ,'Pre':probs[1]
#    #                   }, iter )
#    #writer.add_scalars("Dist of ref images",
#    #                   { 'REC' : dists[0]
#    #                   , 'MSE' : dists[1]
#    #                   , 'L1L' : dists[2]
#    #                   }, iter )
#    try :
#        addToHDF(TCfg.historyHDF, "data", collage)
#    except :
#        eprint("Failed to save.")
#    return collage, probs, dists


#def initialTest() :
#    with torch.inference_mode() :
#        collage, probs, _ = logStep(iter, not iter)
#        print("Probabilities of reference images: "
#              f'Ref: {probs[0]:.3e}, '
#              f'Gen: {probs[2]:.3e}, '
#              f'Pre: {probs[1]:.3e}.')
#        #generator.eval()
#        pre = generator.preProc(refImages)
#        wghts = calculateWeights(refImages)
#        ref_loss_Rec = loss_Rec(refImages[DCfg.gapRng], pre, wghts, normalizeRec=calculateNorm(refImages))
#        ref_loss_MSE = loss_MSE(refImages[DCfg.gapRng], pre, wghts)
#        ref_loss_L1L = loss_L1L(refImages[DCfg.gapRng], pre, wghts)
#        print("Distances of reference images: "
#              f"REC: {ref_loss_Rec:.3e}, "
#              f"MSE: {ref_loss_MSE:.3e}, "
#              f"L1L: {ref_loss_L1L:.3e}.")
#        #if not epoch :
#        #    writer.add_scalars("Dist of ref images",
#        #                          { 'REC' : ref_loss_Rec
#        #                          , 'MSE' : ref_loss_MSE
#        #                          , 'L1L' : ref_loss_L1L
#        #                          }, 0 )
#        plotImage(collage)
]


def calculateWeights(images) :
    return None

def calculateNorm(images) :
    mean2 = images[...,:DCfg.gapRngX.start].mean(dim=(-1,-2)) \
          + images[...,DCfg.gapRngX.stop:].mean(dim=(-1,-2))
    return 2 / ( 1 + mean2 + 1e-5 ) # to denorm and adjust for mean


@dataclass
class TrainResClass:
    lossD : any = 0
    lossGA : any = 0
    lossGD : any = 0
    lossMSE : any = 0
    lossL1L : any = 0
    lossRec : any = 0
    predReal : any = 0
    predPre : any = 0
    predFake : any = 0
    nofPixels : int = 0
    nofImages : int = 0
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




def saveCheckPoint(path, epoch, iterations,
                   minRecTest, minRecTrain, minTestEpoch,
                   generator, discriminator,
                   optimizerGen=None, optimizerDis=None,
                   schedulerGen=None, schedulerDis=None,
                   startFrom=0, interimRes=TrainResClass()) :
    checkPoint = {}
    checkPoint['epoch'] = epoch
    checkPoint['iterations'] = iterations
    checkPoint['minTestEpoch'] = minTestEpoch
    checkPoint['minRecTest'] = minRecTest
    checkPoint['minRecTrain'] = minRecTrain
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
    minTestEpoch = checkPoint['minTestEpoch']
    minRecTest = checkPoint['minRecTest']
    minRecTrain = checkPoint['minRecTrain']
    startFrom = checkPoint['startFrom'] if 'startFrom' in checkPoint else 0
    generator.load_state_dict(checkPoint['generator'])
    discriminator.load_state_dict(checkPoint['discriminator'])
    if not optimizerGen is None and 'optimizerGen' in checkPoint:
        optimizerGen.load_state_dict(checkPoint['optimizerGen'])
    if not schedulerGen is None and 'schedulerGen' in checkPoint :
        schedulerGen.load_state_dict(checkPoint['schedulerGen'])
    if not optimizerDis is None and 'optimizerDis' in checkPoint :
        optimizerDis.load_state_dict(checkPoint['optimizerDis'])
    if not schedulerDis is None and 'schedulerDis' in checkPoint :
        schedulerDis.load_state_dict(checkPoint['schedulerDis'])
    interimRes = checkPoint['resAcc']

    return epoch, iterations, minRecTest, minRecTrain, minTestEpoch, startFrom, interimRes

#trainInfo = TrainInfoClass()
normMSE=1
normL1L=1
normRec=1
normSSIM=1
skipDis = False

def restoreCheckpoint(path=None, logDir=None) :
    if logDir is None :
        logDir = TCfg.logDir
    if path is None :
        if os.path.exists(logDir) :
            raise Exception(f"Starting new experiment with existing log directory \"{logDir}\"."
                            " Remove it .")
        try : os.remove(TCfg.historyHDF)
        except : pass
        return 0, 0, 0, 0, 0, 0, TrainResClass()
    else :
        return loadCheckPoint(path, generator, discriminator, optimizer_G, optimizer_D)


def train_step(images):

    global trainDis, trainGen, eDinfo, noAdv, withNoGrad, skipGen, skipDis
    #trainInfo.iterations += 1
    #trainInfo.totPerformed += 1
    trainRes = TrainResClass()

    nofIm = images.shape[0]
    images = images.to(TCfg.device)
    fakeImages = torch.empty( (nofIm,2,*images.shape[2:]), device=TCfg.device, requires_grad=False)
    subBatchSize = nofIm // TCfg.batchSplit
    #labelsTrue = torch.full((subBatchSize, 1),  1 - TCfg.labelSmoothFac,
    #                    dtype=torch.float, device=TCfg.device, requires_grad=False)
    #labelsFalse = torch.full((subBatchSize, 1),  TCfg.labelSmoothFac,
    #                    dtype=torch.float, device=TCfg.device, requires_grad=False)
    #labelsDis = torch.cat( (labelsTrue, labelsFalse), dim=0).to(TCfg.device).requires_grad_(False)

    # train discriminator
    if not noAdv :
        pass
        # calculate predictions of prefilled images - purely for metrics purposes
        #discriminator.eval()
        #generator.eval()
        #trainRes.predPre = 0
        #with torch.no_grad() :
        #    for i in range(TCfg.batchSplit) :
        #        subRange = np.s_[i*subBatchSize:(i+1)*subBatchSize] if TCfg.batchSplit > 1 else np.s_[...]
        #        fakeImages[subRange,:,DCfg.gapRngX] = generator.preProc(procImages[subRange,...])
        #        trainRes.predPre += discriminator(fakeImages[subRange,...]).mean().item()
        #    trainRes.predPre /= TCfg.batchSplit

        #pred_real = torch.empty((nofIm,1), requires_grad=False)
        #pred_fake = torch.empty((nofIm,1), requires_grad=False)
        ##discriminator.train()
        #for param in discriminator.parameters() :
        #    param.requires_grad = True
        #optimizer_D.zero_grad()
        #for i in range(TCfg.batchSplit) :
        #    subRange = np.s_[i*subBatchSize:(i+1)*subBatchSize] if TCfg.batchSplit > 1 else np.s_[...]
        #    subFakeImages = fakeImages[subRange,...]
        #    with torch.no_grad() :
        #        subFakeImages = generator.generatePatches(procImages[subRange,...])
        #    with torch.set_grad_enabled(not skipDis) :
        #        subPred_realD = discriminator(procImages[subRange,...])
        #        subPred_fakeD = discriminator(subFakeImages)
        #        pred_both = torch.cat((subPred_realD, subPred_fakeD), dim=0)
        #        subD_loss = loss_Adv(labelsDis, pred_both)
        #    # train discriminator only if it is not too good :
        #    if not skipDis and ( subPred_fakeD.mean() > 0.2 or subPred_realD.mean() < 0.8 ) :
        #        trainInfo.disPerformed += 1/TCfg.batchSplit
        #        subD_loss.backward()
        #    trainRes.lossD += subD_loss.item()
        #    pred_real[subRange] = subPred_realD.clone().detach()
        #    pred_fake[subRange] = subPred_fakeD.clone().detach()
        #optimizer_D.step()
        #optimizer_D.zero_grad(set_to_none=True)
        #trainRes.lossD /= TCfg.batchSplit
        #trainRes.predReal = pred_real.mean().item()
        #trainRes.predFake = pred_fake.mean().item()

    else :
        pred_real = torch.zeros((1,), requires_grad=False)
        pred_fake = torch.zeros((1,), requires_grad=False)

    # train generator
    discriminator.eval()
    for param in discriminator.parameters() :
        param.requires_grad = False
    generator.train()
    optimizer_G.zero_grad()
    for i in range(TCfg.batchSplit) :
        subRange = np.s_[i*subBatchSize:(i+1)*subBatchSize] if TCfg.batchSplit > 1 else np.s_[:]
        masks = images[subRange,2:,...]
        subFakeImages = generator.forward((images[subRange,0:4,...],None))
        if noAdv :
            subG_loss = loss_Rec( images[subRange,0:2,...], subFakeImages, masks)
        else :
            pass
            #subPred_fakeG = discriminator(subFakeImages)
            #subGA_loss, subGD_loss = loss_Gen(labelsTrue, subPred_fakeG,
            #                                  procImages[subRange,...], subFakeImages)
            #subG_loss = lossAdvCoef * subGA_loss + lossDifCoef * subGD_loss
            #pred_fake[subRange] = subPred_fakeG.clone().detach()
        subG_loss.backward()
        #trainRes.lossGA += subGA_loss.item()
        #trainRes.lossGD += subGD_loss.item()
        fakeImages[subRange,...] = subFakeImages.detach()
    optimizer_G.step()
    optimizer_G.zero_grad(set_to_none=True)
    #trainRes.lossGA /= TCfg.batchSplit
    #trainRes.lossGD /= TCfg.batchSplit
    #trainRes.predFake = pred_fake.mean().item()


    # prepare report
    with torch.no_grad() :

        masks = images[:,2:,...]
        trainRes.nofPixels = pixelsCounted(masks).sum()
        trainRes.nofImages = nofIm
        trainRes.lossMSE = loss_MSE( images[:,0:2,...], fakeImages, masks ).item()
        trainRes.lossL1L = loss_L1L( images[:,0:2,...], fakeImages, masks ).item()
        trainRes.lossRec = loss_Rec( images[:,0:2,...], fakeImages, masks ).item()

        #idx = random.randint(0, nofIm-1)
        #trainInfo.testIndex = idx
        #trainInfo.orgImage = images[idx,0,...].clone().detach()
        #trainInfo.sftImage = images[idx,1,...].clone().detach()
        #trainInfo.orgRecImage = fakeImages[idx,0,...].clone().detach()
        #trainInfo.sftRecImage = fakeImages[idx,1,...].clone().detach()
        #trainInfo.orgMask = images[idx,2,...].clone().detach()
        #trainInfo.sftMask = images[idx,3,...].clone().detach()

    return trainRes



def beforeEachEpoch(herelocals) :
    return

def afterEachEpoch(herelocals) :
    return

def beforeReport(herelocals) :
    return

def afterReport(herelocals) :
    return

epoch=initIfNew('epoch', 0)
iter = initIfNew('iter', 0)
imer = initIfNew('iter', 0)
minRecTrain = initIfNew('minGEpoch')
minRecTest = initIfNew('minGdLoss', 1)
minTestEpoch = 0
startFrom = initIfNew('startFrom', 0)
normTestMSE=1
normTestSSIM=1
normTestL1L=1
normTestRec=1
normTestDis=1
normTestGen=1
normTestGDloss=1
normTestGAloss=1
normTestDloss=1
resAcc = TrainResClass()

dataLoader=None
testLoader=None

def train(savedCheckPoint):
    global epoch, minRecTest, minRecTrain, minTestEpoch, iter, imer, startFrom, resAcc
    lastRec_test = minRecTest
    lastRec_train = minRecTrain

    discriminator.to(TCfg.device)
    generator.to(TCfg.device)
    lastUpdateTime = time.time()
    lastSaveTime = time.time()

    while TCfg.nofEpochs is None or epoch <= TCfg.nofEpochs :
        epoch += 1
        beforeEachEpoch(locals())
        dataLoader = createDataLoader(createSubSet(trainSet), num_workers=TCfg.loaderWorkers)

        generator.train()
        discriminator.train()
        resAcc = TrainResClass()
        updAcc = TrainResClass()
        totalIm = 0

        for it , data in tqdm.tqdm(enumerate(dataLoader), total=int(len(dataLoader))):
            if startFrom :
                startFrom -= 1
                continue
            iter += 1
            images, indecies = data
            images = images.to(TCfg.device)
            nofIm = images.shape[0]
            imer += nofIm
            totalIm += nofIm
            trainRes = train_step(images)
            resAcc += trainRes
            updAcc += trainRes

            #if True:
            #if False :
            #if not it or it > len(dataloader)-2 or time.time() - lastUpdateTime > 60 :
            if time.time() - lastUpdateTime > 60 :
                lastUpdateTime = time.time()
                refRes = testMe(refImages, plotMe=False)
                rndIdx = random.randint(0,nofIm-1)
                rndInp = images[rndIdx,...]
                rndRes = testMe(rndInp, plotMe=False)

                imGap = 16
                showMe = np.zeros( (2*DCfg.inShape[1] + imGap ,
                                    5*DCfg.inShape[0] + 4*imGap), dtype=np.float32  )
                def addImage(clmn, row, img=None, stretchSimm=False) :
                    imgToAdd = img.clone().detach().squeeze()
                    minv = imgToAdd.min().item()
                    maxv = imgToAdd.max().item()
                    if stretchSimm :
                        lrg = max( abs(minv), abs(maxv) )
                        minv = -lrg
                        maxv = lrg
                    ampl = maxv - minv
                    imgToAdd[()] = 2 * ( imgToAdd - minv ) / ampl - 1  if ampl!=0.0 else 0
                    showMe[ row  * ( DCfg.inShape[1]+imGap) : (row+1 ) * DCfg.inShape[1] + row  * imGap ,
                            clmn * ( DCfg.inShape[0]+imGap) : (clmn+1) * DCfg.inShape[0] + clmn * imGap ] = \
                        imgToAdd.cpu().numpy()
                addImage(0,0, rndRes[0][0,...])
                addImage(0,1, rndRes[0][1,...])
                addImage(1,0, rndRes[0][2,...]) #createTrimage(rndInp, 0))
                addImage(1,1, rndRes[0][3,...]) #createTrimage(rndInp, 1))
                addImage(2,0, refRes[0][0,0,...])
                addImage(2,1, refRes[0][0,1,...])
                addImage(3,0, refRes[0][0,2,...])
                addImage(3,1, refRes[0][0,3,...])
                addImage(4,0, createTrimage(refImages[0,...],0))
                addImage(4,1, createTrimage(refImages[0,...],1))
                #addImage(4,0, stretchSimm=True, img =
                #              ( refRes[0][0,0,...] - refImages[0,0,...] ) * \
                #              (1-refImages[0,2,...]) * refImages[0,3,...] * refImages[0,4,...]  )
                #addImage(4,1, stretchSimm=True, img =
                #              ( refRes[0][0,1,...] - refImages[0,1,...] ) * \
                #              (1-refImages[0,3,...]) * refImages[0,2,...] * refImages[0,5,...] )
                normalizedLosses = updAcc * (1/updAcc.nofPixels if updAcc.nofPixels else 0)
                #writer.add_scalars("Losses per iter",
                #                   {'Dis': trainRes.lossD
                #                   ,'Gen': trainRes.lossGA
                #                   ,'Rec':   lossAdvCoef * trainRes.lossGA \
                #                           + lossDifCoef * trainRes.lossGD * normRec
                #                   }, imer )
                writer.add_scalars("Distances per iter",
                                   {'MSE': normalizedLosses.lossMSE / normMSE
                                   ,'L1L': normalizedLosses.lossL1L / normL1L
                                   ,'REC': normalizedLosses.lossRec / normRec
                                   }, imer )
                #writer.add_scalars("Probs per iter",
                #                   {'Ref':trainRes.predReal
                #                   ,'Gen':trainRes.predFake
                #                   ,'Pre':trainRes.predPre
                #                   }, imer )

                IPython.display.clear_output(wait=True)
                beforeReport(locals())
                print(f"Epoch: {epoch:3} ({minTestEpoch:3})." +
                      f" L1L: {normalizedLosses.lossL1L / normL1L :.3f} " +
                      f" MSE: {normalizedLosses.lossMSE / normMSE :.3f} " +
                      f" Rec: {normalizedLosses.lossRec / normRec :.3f} " +
                      f" (Train: {lastRec_train/normRec:.3f}/{minRecTrain/normRec:.3f},"
                      f" Test: {lastRec_test/normTestRec:.3f}/{minRecTest/normTestRec:.3f})."
                      )
                indexInSet = [ indColumn[rndIdx].item() for indColumn in indecies ]
                print(f"Image {indexInSet}." +
                      f" L1L: {rndRes[2][2] / normL1L :.3f} " +
                      f" MSE: {rndRes[2][1] / normMSE :.3f} " +
                      f" Rec: {rndRes[2][0] / normRec :.3f} "
                      )
                print(f"Reference images." +
                      f" L1L: {refRes[2][2] / normL1L:.3f} " +
                      f" MSE: {refRes[2][1] / normMSE:.3f} " +
                      f" Rec: {refRes[2][0] / normRec:.3f} "
                      )
                #print (f"TT: {trainInfo.bestRealProb:.2f},  "
                #       f"FT: {trainInfo.bestFakeProb:.2f},  "
                #       f"HD: {trainInfo.highestDif/normMSE:.3e},  "
                #       f"GP: {probsR[0,2].item():.3f}, {probsR[0,1].item():.3f} " )
                #print (f"TF: {trainInfo.worstRealProb:.2f},  "
                #       f"FF: {trainInfo.worstFakeProb:.2f},  "
                #       f"LD: {trainInfo.lowestDif/normMSE:.3e},  "
                #       f"R : {probsR[0,0].item():.3f}." )
                plotImage(showMe)
                try :
                    addToHDF(TCfg.historyHDF, "data", showMe)
                except :
                    eprint("Failed to save HDF")
                afterReport(locals())
                updAcc = TrainResClass()

            if time.time() - lastSaveTime > 3600 :
                lastSaveTime = time.time()
                saveCheckPoint(savedCheckPoint+"_hourly.pth", epoch-1, imer,
                               minRecTest, minRecTrain, minTestEpoch,
                               generator, discriminator,
                               optimizer_G, optimizer_D,
                               startFrom=it, interimRes=resAcc)
                saveModels(f"model_{TCfg.exec}_hourly")


        resAcc *=  (1/resAcc.nofPixels) if resAcc.nofPixels else 0
        lastRec_train = resAcc.lossRec
        if not minRecTrain or lastRec_train < minRecTrain :
            minRecTrain = lastRec_train
        writer.add_scalars("Distances per epoch",
                           {'MSE': resAcc.lossMSE / normMSE
                           ,'L1L': resAcc.lossL1L / normL1L
                           ,'REC': resAcc.lossRec / normRec
                           }, epoch )
        #writer.add_scalars("Losses per epoch",
        #                   {'Dis': resAcc.lossD
        #                   ,'Adv': resAcc.lossGA
        #                   ,'Gen': lossAdvCoef * resAcc.lossGA + lossDifCoef * resAcc.lossGD
        #                   }, epoch )
        #writer.add_scalars("Probs per epoch",
        #                   {'Ref': resAcc.predReal
        #                   ,'Gen': resAcc.predFake
        #                   ,'Pre': resAcc.predPre
        #                   }, epoch )
        _ = testMe(refImages)
        Rec_test, MSE_test, L1L_test = summarizeSet(testLoader)[0:3]
        writer.add_scalars("Test per epoch",
                           {'MSE': MSE_test / normTestMSE
                           ,'L1L': L1L_test / normTestL1L
                           ,'REC': Rec_test / normTestRec
                           #,'Dis': Dis_test
                           #,'Gen': Gen_test
                           }, epoch )
        #writer.add_scalars("Test losses per epoch",
        #                   { 'Dis': Dloss_test
        #                   , 'Adv': GAloss_test
        #                   , 'Gen': lossAdvCoef * GAloss_test + lossDifCoef * GDloss_test
        #                   }, epoch )
        #writer.add_scalars("Test probs per epoch",
        #                   {'Ref': Rprob_test
        #                   ,'Gen': Fprob_test
        #                   }, epoch )

        lastRec_test = Rec_test
        if not minRecTest or lastRec_test < minRecTest  :
            minRecTest = lastRec_test
            minTestEpoch = epoch
            saveCheckPoint(savedCheckPoint+"_B.pth", epoch, imer,
                           minRecTest, minRecTrain, minTestEpoch,
                           generator, discriminator,
                           optimizer_G, optimizer_D)
            os.system(f"cp {savedCheckPoint}.pth {savedCheckPoint}_BB.pth") # BB: before best
            os.system(f"cp {savedCheckPoint}_B.pth {savedCheckPoint}.pth") # B: best
            saveModels(f"model_{TCfg.exec}_B")
        else :
            saveCheckPoint(savedCheckPoint+".pth", epoch, imer,
                           minRecTest, minRecTrain, minTestEpoch,
                           generator, discriminator,
                           optimizer_G, optimizer_D)
        saveModels()

        afterEachEpoch(locals())
        resAcc = TrainResClass()





def freeGPUmem() :
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()






