
from re import sub
from weakref import ref
import IPython

import sys
import os
import random
import time
import gc
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import glob
import copy

import math
import statistics
from cv2 import norm
import numpy as np
import test
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torchvision
from torch import optim, rand, randint
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave
import h5py
from h5py import h5d
import tifffile
import tqdm

#import torchmetrics.image
import ssim


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
    dataDir : str
    device: torch.device = torch.device('cpu')
    batchSplit : int = 1 # negative to load multiple batches at a time.
    nofEpochs: int = 0
    num_workers : int = os.cpu_count()
    historyHDF : str = field(repr = True, init = False)
    logDir : str = field(repr = True, init = False)
    def __post_init__(self):
        if self.device == torch.device('cpu')  :
            self.device = torch.device(f"cuda:{self.exec}")
        self.historyHDF = f"train_{self.exec}.hdf"
        self.logDir = f"runs/experiment_{self.exec}"
        if self.batchSplit > 1 and self.batchSize % self.batchSplit :
            raise Exception(f"Batch size {self.batchSize} is not divisible by batch split {self.batchSplit}.")
global TCfg
TCfg = initIfNew('TCfg')


@dataclass
class DCfgClass:
    gapW : int
    brick : bool = field(repr = False)
    sinoSh : tuple = field(repr = True, init = False)
    gapSh : tuple = field(repr = True, init = False)
    gapRngX : type(np.s_[:]) = field(repr = True, init = False)
    gapRng : type(np.s_[:]) = field(repr = True, init = False)
    readSh : tuple = field(repr = True, init = False)
    def __post_init__(self):
        self.readSh : tuple = (128 if self.brick else None ,128)
        self.sinoSh = ( (8 if self.brick else 256) * self.gapW , 8*self.gapW )
        self.gapSh = (self.sinoSh[0],self.gapW)
        self.gapRngX = np.s_[ self.sinoSh[1]//2 - self.gapW//2 : self.sinoSh[1]//2 + self.gapW//2 ]
        self.gapRng = np.s_[...,self.gapRngX]
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

    #plt.style.use('default')
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


def plotImage(image, frameon=False) :
    plt.figure(frameon=frameon)
    plt.imshow(image, cmap='gray')
    plt.axis("off")
    plt.show()


def plotImages(images, frameon=False) :
    plt.figure(frameon=frameon)
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


def set_seed(SEED_VALUE):
    torch.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed_all(SEED_VALUE)
    np.random.seed(SEED_VALUE)


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    return


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
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



def residesInMemory(hdfName) :
    mmapPrefixes = ["/dev/shm",]
    if "CTAS_MMAP_PATH" in os.environ :
        mmapPrefixes.extend(os.environ["CTAS_MMAP_PATH"].split(':'))
    hdfName = os.path.realpath(hdfName)
    for mmapPrefix in mmapPrefixes :
        if hdfName.startswith(mmapPrefix) :
            return True
    return False

def goodForMmap(trgH5F, data) :
    fileSize = trgH5F.id.get_filesize()
    offset = data.id.get_offset()
    plist = data.id.get_create_plist()
    if offset < 0 \
    or not plist.get_layout() in (h5d.CONTIGUOUS, h5d.COMPACT) \
    or plist.get_external_count() \
    or plist.get_nfilters() \
    or fileSize - offset < math.prod(data.shape) * data.dtype.itemsize :
        return None, None
    else :
        return offset, data.id.dtype


def getInData(inputString, verbose=False, preread=False):
    nameSplit = inputString.split(':')
    if len(nameSplit) == 1 : # tiff image
        data = loadImage(nameSplit[0])
        data = np.expand_dims(data, 1)
        return data
    if len(nameSplit) != 2 :
        raise Exception(f"String \"{inputString}\" does not represent an HDF5 format \"fileName:container\".")
    hdfName = nameSplit[0]
    hdfVolume = nameSplit[1]
    try :
        trgH5F =  h5py.File(hdfName,'r', swmr=True)
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
    try : # try to mmap hdf5 if it is in memory
        if not residesInMemory(hdfName) :
            raise Exception()
        fileSize = trgH5F.id.get_filesize()
        offset = data.id.get_offset()
        dtype = data.id.dtype
        plist = data.id.get_create_plist()
        if offset < 0 \
        or not plist.get_layout() in (h5d.CONTIGUOUS, h5d.COMPACT) \
        or plist.get_external_count() \
        or plist.get_nfilters() \
        or fileSize - offset < math.prod(sh) * data.dtype.itemsize :
            raise Exception()
        # now all is ready
        dataN = np.memmap(hdfName, shape=sh, dtype=dtype, mode='r', offset=offset)
        data = dataN
        trgH5F.close()
        #plist = trgH5F.id.get_access_plist()
        #fileno = trgH5F.id.get_vfd_handle(plist)
        #dataM = mmap.mmap(fileno, fileSize, offset=offset, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
    except :
        if preread :
            dataN = np.empty(data.shape, dtype=np.float32)
            if verbose :
                print("Reading input ... ", end="", flush=True)
            data.read_direct(dataN)
            if verbose :
                print("Done.")
            data = dataN
            trgH5F.close()
    return data



def createWriter(logDir, addToExisting=False) :
    if not addToExisting and os.path.exists(logDir) :
        raise Exception(f"Log directory \"{logDir}\" for the experiment already exists."
                        " Remove it or implicitry overwrite with setting addToExisting to True.")
    return SummaryWriter(logDir)
writer = initIfNew('writer')


class DevicePlace:
    def __call__(self, x):
        return x.to(TCfg.device)



class StripesFromHDF :

    # Setting exclusize to True makes dataset consisting only out of non-overlapping sinograms
    def __init__(self, sampleName, maskName, exclusive=False):
        self.volume = getInData(sampleName, False, False)
        self.sh = self.volume.shape
        self.fsh = self.sh[1:3]
        self.mask = loadImage(maskName, self.fsh)
        self.mask /= self.mask.max()
        if self.mask is None :
            self.mask = np.ones(self.fsh, dtype=np.uint8)
        #self.mask = self.mask.astype(bool)
        forbidenSinos = self.mask.copy()
        for shft in range (1, DCfg.readSh[-1]) :
            forbidenSinos[:,:-shft] *= self.mask[:,shft:]
        forbidenSinos[:, -DCfg.readSh[-1]:] = 0
        if exclusive : # non-overlapping sinograms
            for yCr in range(0,self.fsh[0]) :
                xCr = 0
                while xCr < self.fsh[1]-DCfg.readSh[-1] :
                    if np.all(forbidenSinos[yCr, xCr:xCr+DCfg.readSh[-1]] > 0) :
                        forbidenSinos[ yCr, xCr+1 : xCr+DCfg.readSh[-1] ] = 0
                        xCr += DCfg.readSh[-1]
                    else :
                        forbidenSinos[ yCr, xCr ] = 0
                        xCr += 1
        self.allAvailableSinos = np.argwhere(forbidenSinos)
        self.availableFragments = 1 if DCfg.readSh[0] is None else \
            ( (self.sh[0] - DCfg.readSh[0] + 1) // ( DCfg.readSh[0] if exclusive else 1 ) )


    def __len__(self):
        return self.allAvailableSinos.shape[0] * self.availableFragments


    def __getitem__(self, index=None):
        if index is None :
            index = random.randint(0, len(self)-1)
            return self.__getitem__(index)
        elif isinstance(index, int) :
            fdx, zdx = divmod(index, self.availableFragments)
            #ydx, xdx = tuple(self.availableSinos[fdx])
            ydx, xdx = tuple( int(dx) for dx in self.allAvailableSinos[fdx,:] )
            #ydx, xdx = tuple( int(dx) for dx in self.exposedSinos[fdx,:] )
            return self.__getitem__((zdx, ydx, xdx))
        elif isinstance(index, tuple)  and len(index) == 3 :
            zdx = 0 if DCfg.readSh[0] is None else index[0]
            data = self.volume[ zdx : -1 if DCfg.readSh[0] is None else (zdx+DCfg.readSh[0]),
                                index[1],
                                index[2]:index[2]+DCfg.readSh[1]
                              ].copy()
            return data, index
        raise Exception(f"Bad index for data set: {index}.")


class StripesFromHDFs :

    def __init__(self, bases, exclusive=False):
        self.collection = []
        for base in bases :
            print(f"Loading train set {len(self.collection)+1} of {len(bases)}: " + base + " ... ", end="")
            self.collection.append(
                StripesFromHDF(f"{base}.hdf:/data", f"{base}.mask++.tif", exclusive) )
            print("Done")

    def __getitem__(self, index=None):
        if index is None:
            index = random.randint(0,len(self)-1)
            return self.__getitem__(index)
        elif isinstance(index, int) :
            leftover = index
            for setdx in range(len(self.collection)) :
                setLen = len(self.collection[setdx])
                if leftover >= setLen :
                    leftover -= setLen
                else :
                    data, subIndex = self.collection[setdx].__getitem__(leftover)
                    return data, (setdx, *subIndex)
        elif type(index) is tuple and len(index) == 4 :
            return self.collection[index[0]].__getitem__(index[1:])[0], index
        raise Exception(f"Bad index for collection of data sets: {index}.")

    def __len__(self):
        return sum( [ len(set) for set in self.collection ] )


    def get_dataset(self, transform=None, expose=1, shuffle=False) :

        class Sinos(torch.utils.data.Dataset) :
            def __init__(self, root, transform=None, expose=1, shuffle=False):
                self.container = root
                if not ( 0 < expose <= 1 ) :
                    raise f"Provided exposure {expose} is outside (0,1] range."
                self.expose = expose
                self.shuffle = shuffle
                self.transform = transform
                self.oblTransform = transforms.Compose( [transforms.ToTensor(),
                                                         #DevicePlace(),
                                                         transforms.Resize(DCfg.sinoSh)] )
            def __len__(self):
                return int(self.container.__len__() * self.expose)
            def __getitem__(self, index=None, doTransform=True):
                if self.shuffle and isinstance(index, int) : # randomize dataset
                    index = random.randint(0,self.container.__len__()-1)
                data, rIndex = self.container.__getitem__(index)
                data = self.oblTransform(data)
                if doTransform and self.transform :
                    data = self.transform(data)
                return data, rIndex
            def originalSinoLen(self,setdx) :
                return self.container.collection[setdx].sh[0]

        return Sinos(self, transform, expose, shuffle)


listOfTrainData = [
    "18692a.ExpChicken6mGyShift",
    "23574.8965435L.Eiger.32kev_sft",
    "19022g.11-EggLard",
    "18692b.MinceO",
    "23574.8965435L.Eiger.32kev_org",
    "19736b.09_Feb.4176862R_Eig_Threshold-4keV",
    "20982b.04_774784R",
    "18515.Lamb1_Eiger_7m_45keV_360Scan",
    "19736c.8733147R_Eig_Threshold-8keV.SAMPLE_Y1",
    "18692b_input_PhantomM",
    "21836b.2024-08-15-mastectomies.4201381L.35kev.20Hz",
    "23574h.9230799R.35kev",
    "18515.Lamb4_Excised_Eiger_7m_30keV_360Scan.Y1",
    "18648.B_Edist.80keV_0m_Eig_Neoprene.Y2",
    "19932.10_8093920_35keV",
    "19932.14_2442231_23keV",
    "19932.16_4193759_60keV",
    "25763c.11_training"
]
listOfTestData = [
    "19603a.Exposures.70keV_7m_Calf2_Threshold35keV_25ms_Take2",
    "22280a_input_Day_4_40keV_7m_Threshold20keV_50ms_Y04_no_shell__0.05deg",
    "18515.Lamb4_Eiger_5m_50keV_360Scan.SAMPLE_Y1",
    "18692b_input_Phantom0",
    #"19603a.ROI-CTs.50keV_7m_Eiger_Sheep1",
]

def createDataSet(path, listOfData, exclusive=False, expose=1) :
    #listOfData = [file.removesuffix(".hdf") for file in glob.glob(path+ "/*.hdf", recursive=False)]
    listOfData = [ '/'.join((path,file))  for file in listOfData ]
    print(listOfData)
    sinoRoot = StripesFromHDFs(listOfData, exclusive)
    transList = []
    #transList.append(transforms.Resize(DCfg.sinoSh))
    if not exclusive :
        transList.append(transforms.RandomHorizontalFlip()),
        transList.append(transforms.RandomVerticalFlip()),
    #transList.append(transforms.Normalize(mean=(0.5), std=(1)))
    mytransforms = transforms.Compose(transList)
    return sinoRoot.get_dataset( transform=mytransforms, expose=expose, shuffle = True )
trainSet = initIfNew('trainSet')
testSet = initIfNew('testSet')


def createDataLoader(tSet, num_workers=os.cpu_count(), bSizeMult = 1) :
    return torch.utils.data.DataLoader(
        dataset=tSet,
        batch_size = int( bSizeMult * TCfg.batchSize * max(1, -TCfg.batchSplit) ),
        shuffle=False, # randomize dataset instead of the dataloader because it takes enormous amount of time otherwise
        num_workers=num_workers,
        drop_last=True
    )


examples = [
    #(11142, 3150), # (0, 417, 1877)
    (38576, 2560), # (3, 476, 2855)
    (26289, 6300), # (2, 280, 828)
    (24299, 7160), # (2, 113, 988)
    (3186, 2455), # (0, 119, 240)
    ]

def createReferences(tSet, majorIdx = 0) :
    if majorIdx :
        examples.insert(0, examples.pop(majorIdx))
    mytransforms = transforms.Compose([
            transforms.Resize(DCfg.sinoSh),
            #transforms.Normalize(mean=(0.5), std=(1))
    ])
    refImages = torch.empty((len(examples), 1, *DCfg.sinoSh), dtype=torch.float32).to(TCfg.device)
    refBoxes = []
    for idx, ex in enumerate(examples) :
        if DCfg.readSh[0] is None :
            index = (ex[0][0], 0, ex[0][1], ex[0][2])
            refBoxes.append( int(ex[1] * refImages.shape[-2]) )
        else :
            index = (ex[0][0], int(ex[1]*tSet.originalSinoLen(ex[0][0])) , ex[0][1], ex[0][2])
            refBoxes.append(0)
        data = tSet.__getitem__(index, doTransform=False)[0]
        refImages[idx,0,...] = mytransforms(data)


    refNoises = torch.randn((refImages.shape[0],TCfg.latentDim)).to(TCfg.device)
    return refImages, refNoises, refBoxes
refImages = initIfNew('refImages')
refNoises = initIfNew('refNoises')
refBoxes = initIfNew('refBoxes')


def showMe(tSet, index=None) :
    global refImages, refNoises
    index = random.randint(0,len(tSet)-1) if index is None else index
    image, rindex = tSet[index]
    image = image.squeeze().transpose(0,1)
    print(index, rindex)
    tensorStat(image)
    plotImage(image.cpu())
    return rindex


def tensorStat(stat) :
    if not torch.numel(stat) :
        print("Empty tensor.")
        return
    absstat = stat.abs()
    print(f"Mean {stat.mean().item():.3e}, Std {stat.std().item():.3e}, "
          f"Ext [{stat.min().item():.3e}, {stat.max().item():.3e}], "
          f"Abs {absstat.mean().item():.3e}, Tin {absstat.min().item():.3e}, "
          f"Zrs { 1 - torch.count_nonzero(absstat)/torch.numel(absstat):.3e}, "
          f"Nan { 1 - torch.count_nonzero(torch.isfinite(stat))/torch.numel(stat):.3e}")

def normalizeImages(images) :
    images, orgDims = unsqeeze4dim(images)
    #images = images.clone().detach()
    #with torch.no_grad() :
    stds, means = torch.std_mean(images, dim=(-1,-2), keepdim=True)
    stds = stds + 1e-7
    images = (images - means) / stds # normalize per image
    return images, (orgDims, stds, means)

def reNormalizeImages(images, norms, stdOnly=False) :
    images = images * norms[1][:,[0],...].to(images.device)
    if not stdOnly :
        images = images + norms[2][:,[0],...].to(images.device) # renormalise
    images = squeezeOrg(images, norms[0])
    return images

def reinit(model, mean=0, std=1):
    for param in model.parameters() :
        torch.nn.init.normal_(param, mean=mean, std=std)

def addnoise(model, std=1):
    for param in model.parameters() :
        param.data += torch.randn_like(param.data) * std

def fillWheights(seq, std=0.001) :
    for wh in seq :
        if hasattr(wh, 'weight') :
            #torch.nn.init.xavier_uniform_(wh.weight)
            #torch.nn.init.zeros_(wh.weight)
            #torch.nn.init.constant_(wh.weight, 0)
            #torch.nn.init.uniform_(wh.weight, a=0.0, b=1.0, generator=None)
            torch.nn.init.normal_(wh.weight, mean=0.0, std=std)
        if hasattr(wh, 'bias') and wh.bias is not None :
            torch.nn.init.normal_(wh.bias, mean=0.0, std=0)

def unsqeeze4dim(tens):
    orgDims = tens.dim()
    if tens.dim() == 2 :
        tens = tens.unsqueeze(0)
    if tens.dim() == 3 :
        tens = tens.unsqueeze(1)
    return tens, orgDims

def squeezeOrg(tens, orgDims):
    if orgDims == tens.dim():
        pass
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

def stripe2bricks(stripes) :
    width = stripes.shape[-1]
    channels = stripes.shape[1]
    bricks = stripes.unfold(-2,width,width//2).permute(0,2,1,4,3).reshape(-1,channels,width,width)
    return  bricks

brickMasks = {}
def bricks2stripe(bricks, ratio=None) :
    global brickMasks
    if ratio is None:
        ratio = DCfg.sinoSh[-2] // DCfg.sinoSh[-1]
    nofIm = bricks.shape[0] // (2*ratio-1)
    width = bricks.shape[-1]
    channels = bricks.shape[1]
    myMask = brickMasks[width].to(bricks.device)
    bricks = myMask * bricks
    stripesOrg = bricks.view(nofIm,-1,channels,width,width)[:, ::2,...].transpose(1,2).reshape((nofIm,channels,-1,width))
    stripesAux = bricks.view(nofIm,-1,channels,width,width)[:,1::2,...].transpose(1,2).reshape((nofIm,channels,-1,width))
    edge = width//2
    stripes = torch.cat([ stripesOrg[:,:,:edge,:] / myMask[:,:,:edge,:],
                          stripesOrg[:,:,edge:-edge,:] + stripesAux,
                          stripesOrg[:,:,-edge:,:] / myMask[:,:,-edge:,:]
                        ],
                        dim=-2)
    return stripes

def createBrickMasks():
    brickMasks = {}
    brickLen = DCfg.sinoSh[-2]
    while brickLen >= 2 :
        halfLine = [ i + 0.5 for i in range(brickLen//2)]
        halfLine = torch.tensor(halfLine, dtype=torch.float32)
        halfLine /= brickLen//2
        line = torch.cat( (halfLine, halfLine.flip(0)), dim=0)
        myMask = line.view(-1,1).repeat(1,brickLen)
        myMask = myMask.unsqueeze(0).unsqueeze(0) # add batch and channel dims
        brickMasks[brickLen] = myMask
        brickLen //= 2
    return brickMasks

def fillTheGap(images, gap) :
    if images.shape[-2] != gap.shape[-2] or images.shape[0] != gap.shape[0] :
        raise Exception(f"Filling gaps requires inputs of the same size except last dimension. Got {images.shape} and {gap.shape}.")
    if DCfg.sinoSh[-1] % images.shape[-1] != 0 :
        raise Exception(f"Width of the images {images.shape[-1]} is an integer of {DCfg.sinoSh[-1]}.")
    ratio = DCfg.sinoSh[-1] // images.shape[-1]
    if DCfg.gapW % ratio + DCfg.gapRngX.start % ratio != 0 :
        raise Exception(f"Gap width {DCfg.gapW} and gap start {DCfg.gapRngX.start} must be integer multiples of {ratio}.")
    gapWidth = DCfg.gapW // ratio
    gapStart = DCfg.gapRngX.start // ratio
    if images.shape[-1] == gap.shape[-1] :
        gapRng = np.s_[gapStart:gapStart+gapWidth]
    elif gap.shape[-1] == gapWidth :
        gapRng = np.s_[:]
    else :
        raise Exception(f"Bad gap width {gap.shape[-1]} for filling images of width {images.shape[-1]}.")
    channels = min(images.shape[1], gap.shape[1])
    gapped = torch.cat( [ images[:,:channels,:, : gapStart],
                          gap   [:,:channels,:, gapRng].to(images.device),
                          images[:,:channels,:, gapStart+gapWidth : ]
                        ],
                        dim=-1
                      )
    gapped = torch.cat( (gapped, images[:,channels:,...]), dim=1 )
    return gapped














class SubTemplate(nn.Module):

    def __init__(self, gapW, brick):
        super().__init__()
        self.cfg = DCfgClass(gapW, brick)
        self.baseChannels = None
        self.baseChannelsOther = None
        self.lastTouch = None
        self.batchNorm = False

    def device(self):
        return next(self.parameters()).device

    def encblock(self, chIn, chOut, kernel, stride=1, norm=None, padding=1) :
        if norm is None :
            norm = self.batchNorm
        #chIn = chIn*self.baseChannels if chIn >= 0 else -chIn
        #chOut = chOut*self.baseChannels if chOut >= 0 else -chOut
        layers = []
        layers.append( nn.Conv2d(chIn, chOut, kernel, stride=stride, bias = not norm,
                                padding=padding, padding_mode='reflect')  )
        if norm :
            layers.append(nn.BatchNorm2d(chOut))
        layers.append(nn.LeakyReLU(0.2))
        fillWheights(layers)
        return torch.nn.Sequential(*layers)

    def groundFloor(self, mult, kernel, stride=1) :
        block1 = self.encblock(self.inChannels,
                               self.baseChannels,
                               kernel=kernel, stride=1, norm=False)
        block2 = self.encblock(self.baseChannels+self.baseChannelsOther,
                               mult*self.baseChannels,
                               kernel=kernel, stride=stride, norm=self.batchNorm)
        return (block1, block2)

    def encFloor(self, chIn, mult, kernel, stride=1, norm=None, other=1) :
        block1 = self.encblock( chIn*(self.baseChannels + other * self.baseChannelsOther),
                               chIn*self.baseChannels,
                               kernel, stride=1, norm=norm)
        block2 = self.encblock( chIn*(self.baseChannels + other * self.baseChannelsOther),
                               int(chIn*self.baseChannels*mult),
                               kernel, stride=stride, norm=norm)
        return (block1, block2)

    def forward(self, images):
        raise Exception("this is not for direct use")



class SubGeneratorTemplate(SubTemplate):


    def __init__(self, gapW, brick, batchNorm=True, inChannels=1):
        super().__init__(gapW, brick)
        self.lowResGenerator = None
        self.inChannels = inChannels
        self.amplitude = 4
        self.batchNorm = batchNorm


    def zeroOthers(self) :
        self.requires_grad_(False)
        smpl = torch.zeros((1, self.inChannels, *self.cfg.sinoSh))
        for encoder in self.encoders :
            otherChannels = smpl.shape[1]
            encoder[0].weight[:,otherChannels:,...] = 0
            smpl = torch.zeros((1, encoder[0].in_channels, *smpl.shape[2:]),
                               device=encoder[0].weight.device)
            smpl = encoder(smpl)
        for decoder in self.decoders :
            otherChannels = 2*smpl.shape[1]
            decoder[0].weight[otherChannels:,...] = 0
            smpl = torch.zeros((1, decoder[0].in_channels, *smpl.shape[2:]),
                               device=encoder[0].weight.device)
            smpl = decoder(smpl)
        otherChannels = smpl.shape[1] + self.inChannels
        self.lastTouch[0].weight[:,otherChannels:,...]=0


    def createAttic(self, encoders) :
        smpl = torch.zeros((1, self.inChannels, *self.cfg.sinoSh))
        for encoder in encoders :
            smpl = torch.zeros((1, encoder[0].in_channels, *smpl.shape[2:]))
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
        return toRet


    def decblock(self, chIn, chOut, kernel, stride=1, norm=None, padding=1, outputPadding=None) :
        if norm is None :
            norm = self.batchNorm
        if outputPadding is None :
            if isinstance(stride, int) :
                outputPadding = stride - 1
            else :
                outputPadding = tuple( strd - 1 for strd in stride )
        #chIn = chIn*self.baseChannels if chIn >= 0 else -chIn
        #chOut = chOut*self.baseChannels if chOut >= 0 else -chOut
        layers = []
        layers.append( nn.ConvTranspose2d(chIn, chOut, kernel, stride=stride, bias = not norm,
                                          padding=padding, padding_mode='zeros', output_padding=outputPadding) )
        if norm :
            layers.append(nn.BatchNorm2d(chOut))
        layers.append(nn.LeakyReLU(0.2))
        fillWheights(layers)
        return torch.nn.Sequential(*layers)


    def decFloor(self, chOut, reduce, kernel, stride=1, norm=None, other=1) :
        block1 = self.decblock( int(reduce*2*chOut*(self.baseChannels + other * self.baseChannelsOther)),
                                chOut*self.baseChannels,
                                kernel, stride=stride, norm=self.batchNorm)
        block2 = self.decblock( 2*chOut*(self.baseChannels + other * self.baseChannelsOther),
                                chOut*self.baseChannels,
                                kernel, stride=1, norm=norm)
        return (block1, block2)


    def createBasement(self, chIn) :
        toRet = nn.Sequential(
            nn.Conv2d(chIn*(self.baseChannels+self.baseChannelsOther)+self.inChannels, 1, 1),
            nn.Tanh(),
        )
        fillWheights(toRet)
        return toRet


    def createLatentGenerator(self, decoders=None) :
        latInSh = self.fcLink[-1].unflattened_size[-2:]
        latInSh = (1,self.baseChannels,*latInSh)
        latInChannels = math.prod(latInSh)
        toRet = nn.Sequential(
            nn.Linear(latInChannels, latInChannels),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, latInSh[1:]),
        )
        if decoders is None :
            decoders = self.decoders
        for decoder in decoders :
            conv = decoder[0]
            toRet.append( self.decblock(self.baseChannels, self.baseChannels,
                                        kernel=conv.kernel_size,
                                        stride=conv.stride,
                                        norm=False,
                                        padding=conv.padding,
                                        outputPadding=conv.output_padding
                                        ) )
        toRet.append(( self.encblock(self.baseChannels, 1, kernel=1, padding=0, norm=False) ))
        fillWheights(toRet)
        return toRet


    def addLatent(self, images) :
        if self.inChannels == images.shape[1] :
            return images
        if isinstance(self.latentGenerator, float) :
            lSh = list(images.shape)
            lSh[1] = self.inChannels - lSh[1]
            latentChannels = self.latentGenerator * torch.randn( lSh, device=images.device )
        else :
            latentIn = torch.randn( (images.shape[0], self.latentGenerator[0].in_features), device=images.device )
            latentChannels = self.latentGenerator(latentIn)
        latentChannels, _ = normalizeImages(latentChannels)
        return torch.cat( (images, latentChannels), dim=1 )


    def lowResProc(self, images) :
        images, orgDims = unsqeeze4dim(images)
        images = images.to(self.device())
        if self.cfg.gapW == 2:
            gap = torch.cat( [ ( 2*images[:,0:1,:,[self.cfg.gapRngX.start-1]] + images[:,0:1,:,[self.cfg.gapRngX.stop]   ] ) / 3,
                               ( 2*images[:,0:1,:,[self.cfg.gapRngX.stop]   ] + images[:,0:1,:,[self.cfg.gapRngX.start-1]] ) / 3,
                             ],
                             dim=-1
                           )
            res = fillTheGap(images, gap)
            #res = images.clone()
            #res[...,self.cfg.gapRngX.start]  = ( 2*images[:,[0],:,self.cfg.gapRngX.start-1] + images[:,[0],:,self.cfg.gapRngX.stop] ) / 3
            #res[...,self.cfg.gapRngX.stop-1] = ( 2*images[:,[0],:,self.cfg.gapRngX.stop] + images[:,[0],:,self.cfg.gapRngX.start-1] ) / 3
        elif self.lowResGenerator is None :
            res = images
        else :
            preImages = torch.nn.functional.interpolate(images, scale_factor=0.5, mode='area')
            res = self.lowResGenerator.forward(preImages)
            res = torch.nn.functional.interpolate(res, scale_factor=2, mode='bilinear')
        return squeezeOrg(res, orgDims)


    def generateImages(self, images, noises=None) :
        return fillTheGap(images, self.forward(images)[:,[0],...])


    def forward(self, images):
        #raise Exception("this is not for direct use")

        # preform inputs
        lrImages = self.lowResProc(images)
        filledImages = fillTheGap(images.to(lrImages.device), lrImages[:,[0],...])
        if self.lowResGenerator is None :
            imgsIn = filledImages.to(self.device())
        else :
            imgsIn = torch.cat( [ img.to(self.device()) for img in (
                                    filledImages,
                                    lrImages
                                )  ], dim=1)
        imgsIn, norms = normalizeImages(imgsIn)
        imgsIn = self.addLatent(imgsIn)
        dwTrain = [imgsIn,]

        # encoding
        for level, encoder in enumerate(self.encoders):
            inSh = dwTrain[-1].shape
            otherChannels = encoder[0].weight.shape[1] - dwTrain[-1].shape[1]
            imgsI = torch.cat( [dwTrain[-1],
                                torch.zeros((inSh[0], otherChannels, inSh[2], inSh[3]), device=self.device())
                               ], dim=1 )
            dwTrain.append( encoder( imgsI ) )
        # bricks linear link
        mid = self.fcLink(dwTrain[-1])
        upTrain = [mid,]
        # decoding
        for level, decoder in enumerate( self.decoders) :
            inSh = upTrain[-1].shape
            otherChannels = decoder[0].weight.shape[0] - upTrain[-1].shape[1] - dwTrain[-1-level].shape[1]
            imgsI = torch.cat( [ img.to(self.device()) for img in (
                                    upTrain[-1],
                                    dwTrain[-1-level],
                                    torch.zeros((inSh[0], otherChannels, inSh[2], inSh[3]))
                                ) ], dim=1)
            upTrain.append( decoder(imgsI) )

        # last touches
        inSh = upTrain[-1].shape
        otherChannels = self.lastTouch[0].weight.shape[1] - upTrain[-1].shape[1] - imgsIn.shape[1]
        imgsI = torch.cat( [ img.to(self.device()) for img in (
                upTrain[-1],
                imgsIn,
                torch.zeros((inSh[0], otherChannels, inSh[2], inSh[3]))
            ) ], dim=1 )
        results = self.lastTouch(imgsI) * self.amplitude
        results = reNormalizeImages(results, norms, stdOnly=True)

        # final result
        results = lrImages + results.to(lrImages.device)
        return results



class GeneratorTemplate(SubGeneratorTemplate):

    def __init__(self, gapW, batchNorm=True):
        super().__init__(gapW, False, batchNorm)
        self.brickGenerator = None
        self.stipeGenerator = None


    def procTail(self, stripe_starter, bricksStriped_starter,
                       bricks_starter, stripeBricked_starter) :
        stripe_dwTrainTail = [stripe_starter,]
        for encoder in self.stripeGenerator.tailEncoders  :
            stripe_dwTrainTail.append(encoder(stripe_dwTrainTail[-1]))
        # stripes linear link
        stripe_mid = self.stripeGenerator.fcLink(stripe_dwTrainTail[-1])
        stripe_upTrainTail = [stripe_mid,]
        # stripes tail decoders
        for level, decoder in enumerate(self.stripeGenerator.tailDecoders) :
            stripe_upTrainTail.append( decoder( torch.cat( (stripe_upTrainTail[-1], stripe_dwTrainTail[-1-level]), dim=1 ) ) )
        return stripe_upTrainTail[-1]


    def forward(self, images):


        # preform inputs
        lrImages = self.lowResProc(images)
        filledImages = fillTheGap(images.to(lrImages.device), lrImages[:,[0],...])
        if self.lowResGenerator is None :
            stripeIn = filledImages.to(self.stripeGenerator.device())
        else :
            stripeIn = torch.cat( [ img.to(self.stripeGenerator.device()) for img in (
                                    filledImages,
                                    lrImages
                                )  ], dim=1)
        stripeIn, stripe_norms = normalizeImages(stripeIn)
        #lrImagesNormalized = stripeIn[:,[-1],...]
        stripeIn = self.stripeGenerator.addLatent(stripeIn)
        stripe_dwTrain = [stripeIn,]

        lrImagesBricked = stripe2bricks(lrImages)
        filledImagesBricked = stripe2bricks(filledImages)
        if self.lowResGenerator is None :
            bricksIn = lrImagesBricked.to(self.brickGenerator.device())
        else :
            bricksIn = torch.cat( [ img.to(self.brickGenerator.device()) for img in (
                                    filledImagesBricked,
                                    lrImagesBricked
                                ) ], dim=1)
        bricksIn, bricks_norms = normalizeImages(bricksIn)
        #lrImagesBrickedNormalized = bricksIn[:,[-1],...]
        bricksIn = self.brickGenerator.addLatent(bricksIn)
        bricks_dwTrain = [bricksIn,]

        stripeBricked_dwTrain = [torch.empty((bricksIn.shape[0],0,bricksIn.shape[2],bricksIn.shape[3])),]
        bricksStriped_dwTrain = [torch.empty((stripeIn.shape[0],0,stripeIn.shape[2],stripeIn.shape[3])),]

        # encoding
        for level, (brick_encoder, stripe_encoder) in enumerate( zip(self.brickGenerator.encoders, self.stripeGenerator.headEncoders) ):

            bricksI = torch.cat( [ bricks_dwTrain[-1], stripeBricked_dwTrain[-1].to(self.brickGenerator.device()) ], dim=1 )
            bricks_dwTrain.append( brick_encoder( bricksI ) )
            bricksStriped_dwTrain.append( bricks2stripe(bricks_dwTrain[-1]) )

            stripeI = torch.cat( [ stripe_dwTrain[-1], bricksStriped_dwTrain[-2].to(self.stripeGenerator.device()) ], dim=1 )
            stripe_dwTrain.append( stripe_encoder(stripeI))
            stripeBricked_dwTrain.append( stripe2bricks(stripe_dwTrain[-1]) )

        # bricks linear link
        bricks_mid = self.brickGenerator.fcLink(bricks_dwTrain[-1])

        # stripes tail
        tailed = self.procTail(stripe_dwTrain[-1], bricksStriped_dwTrain[-1],
                               bricks_dwTrain[-1], stripeBricked_dwTrain[-1], )

        bricks_upTrain = [bricks_mid,]
        stripe_upTrain = [tailed,]
        # decoding
        for level, (brick_decoder, stripe_decoder) in enumerate( zip(self.brickGenerator.decoders, self.stripeGenerator.headDecoders) ):

            bricksI = torch.cat( [ img.to(self.brickGenerator.device()) for img in (
                                    bricks_upTrain[-1],
                                    bricks_dwTrain[-1-level],
                                    stripe2bricks(stripe_upTrain[-1]),
                                    stripeBricked_dwTrain[-1-level]
                                ) ], dim=1)
            bricks_upTrain.append( brick_decoder(bricksI) )

            stripeI = torch.cat( [ img.to(self.stripeGenerator.device()) for img in (
                                    stripe_upTrain[-1],
                                    stripe_dwTrain[-1-level],
                                    bricks2stripe(bricks_upTrain[-2]),
                                    bricksStriped_dwTrain[-1-level],
                                ) ] , dim=1)
            stripe_upTrain.append( stripe_decoder(stripeI) )

        # last touches
        stripeI = torch.cat( [ img.to(self.stripeGenerator.device()) for img in (
                stripe_upTrain[-1],
                stripeIn,
                bricks2stripe(bricks_upTrain[-1]),
            ) ], dim=1 )
        stripe_results = self.stripeGenerator.lastTouch(stripeI) * self.stripeGenerator.amplitude
        stripe_results = reNormalizeImages(stripe_results, stripe_norms, stdOnly=True)

        bricksI = torch.cat( [ img.to(self.brickGenerator.device()) for img in (
                bricks_upTrain[-1],
                bricksIn,
                stripe2bricks(stripe_upTrain[-1]),
            ) ], dim=1 )
        bricks_results = self.brickGenerator.lastTouch(bricksI) * self.brickGenerator.amplitude
        bricks_results = reNormalizeImages(bricks_results, bricks_norms, stdOnly=True)
        bricks_results = bricks2stripe(bricks_results)

        # final result
        results = lrImages + bricks_results.to(lrImages.device) + stripe_results.to(lrImages.device)
        return results



generator = initIfNew('generator')
lowResGenerators = initIfNew('lowResGenerators', {})



class SubDiscriminatorTemplate(SubTemplate):

    def __init__(self, gapW, brick):
        super().__init__(gapW, brick)
        self.baseChannels = None
        self.baseChannelsOther = None
        self.inChannels = 1


    def createBody(self, encoders) :
        smpl = torch.zeros((1, 1, *self.cfg.sinoSh))
        for encoder in encoders :
            smpl = torch.zeros((1, encoder[0].in_channels, *smpl.shape[2:]))
            smpl = encoder(smpl)
        encSh = smpl.shape
        leftChannels = math.prod(encSh)
        layers = [nn.Flatten(),]
        while leftChannels > 1 :
            outChannels = max(leftChannels//4, 1)
            layers.append(nn.Linear(leftChannels, outChannels))
            layers.append(nn.LeakyReLU(0.2))
            leftChannels = outChannels
        return torch.nn.Sequential(*layers)

    def createMixer(self) :
        ratio = self.cfg.sinoSh[-2] // self.cfg.sinoSh[-1]
        inChans = 2*ratio
        return torch.nn.Sequential(
            nn.Linear(inChans, 1),
            nn.Sigmoid(),
        )

    def forward(self, images):
        raise Exception("this is not for direct use")



class DiscriminatorTemplate(SubDiscriminatorTemplate):

    def __init__(self, gapW):
        super().__init__(gapW, False)
        self.bricksDiscriminator = None
        self.stripeDiscriminator = None

    def procTail(self, stripe_starter, bricksStriped_starter,
                       bricks_starter, stripeBricked_starter) :
        stripe_dwTrainTail = [stripe_starter,]
        for encoder in self.stripeDiscriminator.tailEncoders  :
            stripe_dwTrainTail.append(encoder(stripe_dwTrainTail[-1]))
        # stripes linear link
        return self.stripeDiscriminator.lastTouch(stripe_dwTrainTail[-1])

    def forward(self, images):

        # preform inputs
        stripeIn = images.to(self.stripeDiscriminator.device())
        stripeIn = normalizeImages(stripeIn)[0]
        stripe_dwTrain = [stripeIn,]

        bricksIn = stripe2bricks(images).to(self.bricksDiscriminator.device())
        bricksIn = normalizeImages(bricksIn)[0]
        bricks_dwTrain = [bricksIn,]

        stripeBricked_dwTrain = [torch.empty((bricksIn.shape[0],0,bricksIn.shape[2],bricksIn.shape[3])),]
        bricksStriped_dwTrain = [torch.empty((stripeIn.shape[0],0,stripeIn.shape[2],stripeIn.shape[3])),]

        # encoding
        for level, (brick_encoder, stripe_encoder) in enumerate( zip(self.bricksDiscriminator.encoders, self.stripeDiscriminator.headEncoders) ):

            bricksI = torch.cat( [ bricks_dwTrain[-1], stripeBricked_dwTrain[-1].to(self.bricksDiscriminator.device()) ], dim=1 )
            bricks_dwTrain.append( brick_encoder( bricksI ) )
            bricksStriped_dwTrain.append( bricks2stripe(bricks_dwTrain[-1]) )

            stripeI = torch.cat( [ stripe_dwTrain[-1], bricksStriped_dwTrain[-2].to(self.stripeDiscriminator.device()) ], dim=1 )
            stripe_dwTrain.append( stripe_encoder(stripeI))
            stripeBricked_dwTrain.append( stripe2bricks(stripe_dwTrain[-1]) )

        # stripes tail
        stripe_res = self.procTail(stripe_dwTrain[-1], bricksStriped_dwTrain[-1],
                               bricks_dwTrain[-1], stripeBricked_dwTrain[-1], )
        # bricks linear link
        bricks_res = self.bricksDiscriminator.lastTouch(bricks_dwTrain[-1])
        bricks_res = bricks_res.view(stripe_res.shape[0],-1)

        results = torch.cat( [stripe_res, bricks_res], dim=1 )
        results = self.stripeDiscriminator.mixer(results)
        return results



discriminator = initIfNew('discriminator')


def createOptimizers() :
    raise Exception("Don't use this function, redefine it as needed.")
schedulers_G = []
schedulers_D = []
optimizers_G = []
optimizers_D = []


def adjustScheduler(scheduler, iniLr, target) :
    if scheduler is None :
        return ""
    gamma = scheduler.gamma
    curLR = scheduler.get_last_lr()[0] / iniLr
    if gamma < 1 and curLR > target \
    or gamma > 1 and curLR < target :
        scheduler.step()
    return f"LR : {scheduler.get_last_lr()[0]:.3e} ({curLR:.3e}). "












BCE = nn.BCELoss(reduction='none')
def loss_Adv(images, truth):
    if discriminator is None :
        raise Exception("Discriminator is not initialized for adversarial loss.")
    nofIm = images.shape[0]
    batchSplit = TCfg.batchSplit if TCfg.batchSplit > 1 else 1
    subBatchSize = TCfg.batchSize // batchSplit
    subBatches = max (nofIm // subBatchSize, 1)
    predictions = torch.empty((nofIm, 1), dtype=torch.float, device=images.device)
    for i in range(subBatches) :
        subRange = np.s_[i*subBatchSize:(i+1)*subBatchSize]
        predictions[subRange,...] = discriminator(images[subRange,...])
    labels = torch.full((nofIm, 1),  (1 - TCfg.labelSmoothFac ) if truth else TCfg.labelSmoothFac,
                        dtype=torch.float, device=images.device)
    BCE.to(images.device)
    return BCE(predictions, labels)[...,0], predictions

weightedAdversarialLoss = True
def loss_Adv_Gen(p_true, p_pred):
    global imer, weightedAdversarialLoss
    loss_pred, predictions_pred = loss_Adv(p_pred, False)
    if not weightedAdversarialLoss :
        return loss_pred
    loss_true, predictions_true = loss_Adv(p_true, True)
    advWeights = ( (predictions_true+1e-7) / (predictions_pred+1e-7) ) -  1
    writer.add_scalars("Aux", {'Adversiry': advWeights.mean()}, imer)
    return loss_pred , advWeights

def loss_Adv_Dis(p_true, p_pred):
    loss_true, predictions_true = loss_Adv(p_true, True)
    loss_pred, predictions_pred = loss_Adv(p_pred, False)
    return ( torch.cat((loss_true, loss_pred), dim=0),
             torch.cat((predictions_true, predictions_pred), dim=0) )

MSE = nn.MSELoss(reduction='none')
def loss_MSE(p_true, p_pred):
    MSE.to(p_pred.device)
    return MSE(p_true[DCfg.gapRng], p_pred[DCfg.gapRng]).sum(dim=(-1,-2,-3))

def loss_MSEM(p_true, p_pred):
    MSE.to(p_pred.device)
    toRet = torch.zeros((p_true.shape[0],), device=p_pred.device)
    current_true = p_true
    current_pred = p_pred
    while current_pred.shape[-2] > 2 :
        scale = p_true.shape[-2] / current_pred.shape[-2]
        toRet += loss_MSE(current_true, current_pred) * scale
        current_true = torch.nn.functional.interpolate(current_true, scale_factor=(0.5,1), mode='area')
        current_pred = torch.nn.functional.interpolate(current_pred, scale_factor=(0.5,1), mode='area')
    return toRet


avgKernel = torch.ones((1,1,3,3), dtype=torch.float32)
def loss_MSEC(p_true, p_pred):
    global avgKernel
    dev = p_pred.device
    MSE.to(dev)
    avgKernel = avgKernel.to(dev)
    p_true = p_true.to(dev)
    mseLoss = MSE(p_true[DCfg.gapRng], p_pred[DCfg.gapRng])
    diffLoss = p_true[DCfg.gapRng] - p_pred[DCfg.gapRng]
    convDiff = torch.nn.functional.conv2d( diffLoss, avgKernel, padding=1 )
    loss = mseLoss * convDiff**2
    return loss.sum(dim=(-1,-2,-3))

def loss_MSEA(p_true, p_pred):
    MSE.to(p_pred.device)
    #return MSE(p_true[DCfg.gapRng], p_pred[DCfg.gapRng]).sum(dim=(-1,-2,-3))
    gap_true = p_true[DCfg.gapRng]
    gap_pred = p_pred[DCfg.gapRng]
    allMSEs = torch.zeros((gap_true.shape[0],9,gap_true.shape[-2],gap_true.shape[-1]), device=p_pred.device) + 1e10
    for id2 in [-1,0,1] :
        for id3 in [-1,0,1] :
            allMSEs[:,[(id3+1)*3+(id2+1)],
                      max(0,id2) : gap_true.shape[2] + min(0,id2),
                      max(0,id3) : gap_true.shape[3] + min(0,id3)] = \
                MSE( gap_true[:,:,max(0,id2) : gap_true.shape[2] + min(0,id2),
                                  max(0,id3) : gap_true.shape[3] + min(0,id3)] ,
                     gap_pred[:,:,max(0,id2)-id2 : gap_true.shape[2] + min(0,id2)-id2,
                                  max(0,id3)-id3 : gap_true.shape[3] + min(0,id3)-id3] )
    minMSE = allMSEs.min(dim=1, keepdim=True)
    return minMSE.sum(dim=(-1,-2,-3))

def loss_MSEN(p_true, p_pred):
    rawLoss = loss_MSE(p_true, p_pred)
    stds = 1e-7 +  calculateNorm(p_true)[0].view([-1])
    return rawLoss / stds**2

def loss_MSENorm(p_true, p_pred):
    n_true, norms = normalizeImages(p_true)
    n_pred = (p_pred - norms[1]) / norms[2]
    return loss_MSE(n_true, n_pred)

def loss_MSEBrick(p_true, p_pred):
    b_true = stripe2bricks(p_true)
    b_pred = stripe2bricks(p_pred)
    n_true, norms = normalizeImages(b_true)
    n_pred = (b_pred - norms[2]) / norms[1]
    bLosses = loss_MSE(n_true, n_pred)
    sLosses = bLosses.view(p_true.shape[0], -1).mean(dim=1)
    return sLosses


def loss_MSEL(p_true, p_pred):
    l_true = p_true.view(-1,p_pred.shape[1],1,p_pred.shape[-1])
    l_pred = p_pred.view(-1,p_pred.shape[1],1,p_pred.shape[-1])
    n_true, norms = normalizeImages(l_true)
    n_pred = (l_pred - norms[2]) / norms[1]
    lLosses = loss_MSE(n_true, n_pred)
    sLosses = lLosses.view(p_true.shape[0], -1).mean(dim=1)
    return sLosses


def loss_MSECT(p_true, p_pred):
    e_true = - torch.log(torch.where(p_true>1e-07,p_true,1e-07))
    f_true = torch.fft.rfft(e_true, dim=-1)
    f_true *= 1 + torch.arange(f_true.shape[-1]).view(1,1,1,-1).to(f_true.device)
    e_true = torch.fft.irfft(f_true, dim=-1)
    e_pred = - torch.log(torch.where(p_pred>1e-07,p_pred,1e-07))
    f_pred = torch.fft.rfft(e_pred, dim=-1)
    f_pred *= 1 + torch.arange(f_pred.shape[-1]).view(1,1,1,-1).to(f_pred.device)
    e_pred = torch.fft.irfft(f_pred, dim=-1)
    return loss_MSE(e_true, e_pred)


def loss_SMSE(p_true, p_pred):
    mseLoss = MSE(p_true[DCfg.gapRng], p_pred[DCfg.gapRng])
    return (torch.square(mseLoss)).sum(dim=(-1,-2,-3))

L1L = nn.L1Loss(reduction='none')
def loss_L1L(p_true, p_pred):
    L1L.to(p_true.device)
    return L1L(p_true[DCfg.gapRng], p_pred[DCfg.gapRng]).sum(dim=(-1,-2,-3))

def loss_L1LL(p_true, p_pred):
    l_true = p_true.view(-1,p_pred.shape[1],1,p_pred.shape[-1])
    l_pred = p_pred.view(-1,p_pred.shape[1],1,p_pred.shape[-1])
    n_true, norms = normalizeImages(l_true)
    n_pred = (l_pred - norms[2]) / norms[1]
    lLosses = loss_L1L(n_true, n_pred)
    sLosses = lLosses.view(p_true.shape[0], -1).mean(dim=1)
    return sLosses



def loss_L1LN(p_true, p_pred):
    rawLoss = loss_L1L(p_true, p_pred)
    stds = 1e-7 + calculateNorm(p_true)[0].view([-1])
    return rawLoss / stds




def loss_COR(p_true, p_pred):
    d_true, _ = unsqeeze4dim(p_true[DCfg.gapRng])
    d_pred, _ = unsqeeze4dim(p_pred[DCfg.gapRng])
    means = torch.mean(d_true, dim=(-1,-2), keepdim=True)
    d_true = d_true - means
    d_pred = d_pred - means
    cor = (d_true*d_pred).sum(dim=(-1,-2))
    dist = 1 - cor / (d_true**2).sum(dim=(-1,-2))
    return torch.abs(dist.squeeze(1))
    #return 1 - cor / torch.sqrt( ( (p_true-means)**2 ).sum(dim=(-1,-2)) * ( (p_pred-means)**2 ).sum(dim=(-1,-2)) + 1e-7 )

def loss_STD(p_true, p_pred):
    p_true, _ = unsqeeze4dim(p_true[DCfg.gapRng])
    p_pred, _ = unsqeeze4dim(p_pred[DCfg.gapRng])
    return (p_true - p_pred).std(dim=(-1,-2,-3))

normalHist = torch.tensor([0.044058, 0.09185, 0.14988, 0.19145, 0.19146, 0.14987, 0.09184, 0.04405])
def loss_HIST(p_true, p_pred):
    global normalHist
    diffImages = p_true[DCfg.gapRng] - p_pred[DCfg.gapRng]
    normDiff, _ = normalizeImages(diffImages)
    normDiff = normDiff.view(normDiff.shape[0],-1)
    normalHist = normalHist.to(normDiff.device)
    toRet = []
    for imgn in range(normDiff.shape[0]) :
        img = normDiff[imgn,...]
        hist = torch.histc(img, bins=8, min=-2, max=2) / math.prod(img.shape)
        img_loss = torch.sum( (hist - normalHist)**2 )
        toRet.append( img_loss )
    return torch.stack(toRet)


def loss_LR(p_true, p_pred) :
    lr_pred = torch.nn.functional.interpolate(p_pred, scale_factor=0.5, mode='area')
    with torch.no_grad() :
        lr_true = torch.nn.functional.interpolate(p_true.to(p_pred.device), scale_factor=0.5, mode='area')
        lr_true = generator.lowResGenerator.generateImages(lr_true.detach())
    return MSE(lr_pred[generator.lowResGenerator.cfg.gapRng], lr_true[generator.lowResGenerator.cfg.gapRng]).sum(dim=(-1,-2,-3))


sobelKernelXY = torch.tensor([[[[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]]], # x
                              [[[ 1, 2, 1],
                                [ 0, 0, 0],
                                [-1,-2,-1]]], # y
                             ], dtype=torch.float32)

def loss_ABSGRD(p_true, p_pred):
    global sobelKernelXY
    p_true, _ = unsqeeze4dim(p_true)
    p_pred, _ = unsqeeze4dim(p_pred)
    sobelKernelXY = sobelKernelXY.to(p_pred.device)
    grad_true = torch.nn.functional.conv2d(p_true, sobelKernelXY, padding=1)
    grad_pred = torch.nn.functional.conv2d(p_pred, sobelKernelXY, padding=1)
    agrd_true = grad_true / (p_true+1e-7)
    agrd_pred = grad_pred / (p_pred+1e-7)
    return MSE(agrd_true[DCfg.gapRng], agrd_pred[DCfg.gapRng]).sum(dim=(-1,-2,-3))


@dataclass
class Metrics:
    calculate : callable
    norm : float # normalization factor - result of calculate on untrained model output
    weight : float # weight in the final loss function; zero means no loss contribution

metrices = {
    'Adv'    : Metrics(loss_Adv_Gen, 0, 0),
    'MSE'    : Metrics(loss_MSE,     1, 1),
    'MSEC'   : Metrics(loss_MSEC,     1, 1),
    #'SMSE'   : Metrics(loss_SMSE,    1, 1),
    'MSEN'   : Metrics(loss_MSEN,    1, 1),
    'L1L'    : Metrics(loss_L1L,     1, 1),
    'L1LN'   : Metrics(loss_L1LN,    1, 1),
    'STD'    : Metrics(loss_STD,     1, 1),
    'COR'    : Metrics(loss_COR,     1, 1),
    'HIST'   : Metrics(loss_HIST,    1, 1),
    'ABSGRD' : Metrics(loss_ABSGRD,  1, 1),
}


minMetrices = None
maxMetrices = None

def trackExtremes(track=True) :
    global minMetrices, maxMetrices
    toRet = (minMetrices, maxMetrices)
    if not track :
        minMetrices = None
        maxMetrices = None
    else :
        minMetrices = { key: None for key in metrices.keys() }
        minMetrices['loss'] = None
        maxMetrices = { key: None for key in metrices.keys() }
        maxMetrices['loss'] = None
    return toRet

def updateExtremes(vector, key, p_true, p_pred) :
    global minMetrices, maxMetrices
    if maxMetrices is not None :
        pos = vector.argmax()
        if maxMetrices[key] is None or vector[pos] > maxMetrices[key][0] :
            maxMetrices[key] = (vector[pos].item(),
                                p_true[pos,...].clone().detach(),
                                p_pred[pos,...].clone().detach() )
            #if key == 'loss' :
            #    print(pos.item())
            #    plotImage(maxMetrices['loss'][1][0].clone().detach().transpose(-1,-2).cpu().numpy())
            #    print()
    if minMetrices is not None :
        pos = vector.argmin()
        if minMetrices[key] is None or vector[pos] < minMetrices[key][0] :
            minMetrices[key] = (vector[pos].item(),
                                p_true[pos,...].clone().detach(),
                                p_pred[pos,...].clone().detach() )



def loss_Gen(p_true, p_pred):
    global metrices, minMetrices, maxMetrices
    myDev = p_pred.device
    p_true = p_true.to(myDev)
    losses = torch.tensor(0.0, requires_grad=True, device=myDev)
    sumweights = 0
    individualLosses = {}
    for key, metrics in metrices.items():
        if metrics.norm > 0 :
            #with torch.set_grad_enabled( metrics.weight > 0 ) :
            #    thisLoss = metrics.calculate(p_true, p_pred).to(myDev) / metrics.norm
            #losses = losses + thisLoss * metrics.weight
            #sumweights += metrics.weight
            with torch.set_grad_enabled( metrics.weight > 0 ) :
                returnFromLoss = metrics.calculate(p_true, p_pred)
                if isinstance(returnFromLoss, tuple) :
                    thisLoss = returnFromLoss[1].to(myDev) * returnFromLoss[0].to(myDev) / metrics.norm
                    weightModifiers = returnFromLoss[1]
                else :
                    thisLoss = returnFromLoss.to(myDev) / metrics.norm
                    weightModifiers = torch.tensor(1.0, requires_grad=False, device=myDev)
            losses = losses + thisLoss * weightModifiers * metrics.weight
            sumweights = sumweights + metrics.weight * weightModifiers.sum()
            individualLosses[key] = thisLoss.detach().sum().item()
            updateExtremes(thisLoss, key, p_true, p_pred)
        else :
            individualLosses[key] = p_true.shape[0]
    loss = losses / sumweights
    updateExtremes(loss, 'loss', p_true, p_pred)
    return loss.sum() , individualLosses


def loss_Dis(p_true, p_pred):
    myDev = p_pred.device
    p_true = p_true.to(myDev)
    advRes = loss_Adv_Dis(p_true, p_pred)
    return advRes[0].sum() / ( 2 * metrices['Adv'].norm ) , advRes[1]
















@dataclass(repr = False)
class TrainResClass:
    lossD : float = 0
    lossG : float = 0
    metrices : dict = field(default_factory=lambda: { key: 0 for key in metrices.keys() })
    predReal : float = 0
    #predPre : float = 0
    predFake : float = 0
    predGen : float = 0
    nofIm : int = 0

    def __repr__(self):
        if self.nofIm == 0 :
            return "No train results yet accumulated."
        return f"Images: {self.nofIm}. DIS: {self.lossD/self.nofIm:.3e}, GEN: {self.lossG/self.nofIm:.3e}. " + \
               f"Probs: True {self.predReal/self.nofIm:.3e}, Fake {self.predFake/self.nofIm:.3e}.\n" + \
               f"Individual losses: " + ' '.join([ f"{key}: {self.metrices[key]/self.nofIm:.3e} "
                                                   for key in self.metrices.keys()  ] ) +\
                "\n"

    def __add__(self, other):
        toRet = TrainResClass()
        for field in dataclasses.fields(TrainResClass):
            fn = field.name
            if fn == 'metrices' :
                for key in self.metrices.keys() :
                    toRet.metrices[key] = self.metrices[key] + other.metrices[key]
            else :
                setattr(toRet, fn, getattr(self, fn) + getattr(other, fn) )
        return toRet

    def __mul__(self, other):
        toRet = TrainResClass()
        for field in dataclasses.fields(TrainResClass):
            fn = field.name
            if fn == 'metrices' :
                for key in self.metrices.keys() :
                    toRet.metrices[key] = self.metrices[key] * other
            else :
                setattr(toRet, fn, getattr(self, fn) * other)
        return toRet

    __rmul__ = __mul__


def summarizeMe(toSumm, onPrep=True):
    global generator, discriminator, metrices

    if isinstance(toSumm, torch.utils.data.Dataset) :
        loader = createDataLoader(toSumm)
        return summarizeMe(loader, onPrep)

    sumAcc = TrainResClass()

    #generator.train()
    #generator.eval()
    #discriminator.eval()

    def summarizeImages(images) :
        nonlocal sumAcc

        images = unsqeeze4dim(images)[0]
        nofIm = images.shape[0]
        sumAcc.nofIm += nofIm

        batchSplit = TCfg.batchSplit if TCfg.batchSplit > 1 else 1
        subBatchSize = nofIm // batchSplit
        for i in range(batchSplit) :
            subRange = np.s_[i*subBatchSize:(i+1)*subBatchSize]
            subImages = preTransformImage(images[subRange,...])
            subFakeImages = generator.lowResProc(subImages) \
                                if onPrep else \
                            generator.generateImages(subImages)
            if 'Adv' in metrices and metrices['Adv'].weight > 0 :
                disLoss, probs = loss_Dis(subImages, subFakeImages)
                sumAcc.lossD += disLoss.item()
                sumAcc.predReal += probs[:subBatchSize,0].sum().item()
                sumAcc.predFake += probs[subBatchSize:,0].sum().item()
            genLoss, indLosses = loss_Gen(subImages, subFakeImages)
            sumAcc.lossG += genLoss.item()
            for key in indLosses.keys() :
                sumAcc.metrices[key] += indLosses[key]


    with torch.no_grad() :
        if isinstance(toSumm, torch.utils.data.DataLoader) :
            for it , data in tqdm.tqdm(enumerate(toSumm), total=int(len(toSumm))):
                summarizeImages(data[0])
        elif isinstance(toSumm, torch.Tensor) :
            summarizeImages(toSumm)
        else :
            raise Exception(f"Unknown type of input to summarizeMe: {type(toSumm)}.")

    print(sumAcc)
    return sumAcc



def generateDisplay(inp=None, boxes=None) :
    images = refImages if inp is None else inp
    images, orgDim = unsqeeze4dim(images)
    nofIm = images.shape[0]
    viewLen = DCfg.sinoSh[-1]
    imDevice = images.device

    genImages = images.clone()
    with torch.no_grad() :
        preImages = generator.lowResProc(images).to(imDevice)
        genPatches = generator.forward(images).to(imDevice)
        genImages[DCfg.gapRng] = genPatches[DCfg.gapRng]
    hGap = DCfg.gapW // 2

    if inp is None :
        boxes = refBoxes
    elif boxes is None : # find worst boxes
        diffImages = torch.abs(genImages - images)
        diffY = fn.conv2d(diffImages, torch.ones((1,1,diffImages.shape[-1],diffImages.shape[-1]), device=diffImages.device))
        boxes = diffY.squeeze(1,-1).argmax(dim=-1)

    views = torch.empty((nofIm, 4, viewLen, viewLen ), dtype=torch.float32, device=imDevice)
    for curim in range(nofIm) :
        rng = np.s_[curim, 0, boxes[curim] : boxes[curim] + viewLen, : ]
        views[curim,0,...] = images   [rng]
        views[curim,1,...] = preImages[rng]
        views[curim,2,...] = genImages[rng]
        views[curim,3,...] = 0
        views[curim,3,*DCfg.gapRng] = (genPatches - preImages)[DCfg.gapRng][rng]
        views[curim,3,:,hGap:hGap+DCfg.gapW] = (images - preImages)[DCfg.gapRng][rng]
        views[curim,3,:,-DCfg.gapW-hGap:-hGap] = (images - genPatches)[DCfg.gapRng][rng]
    return views, squeezeOrg(genImages, orgDim) , squeezeOrg(preImages, orgDim)


def displayImages(inp=None, boxes=None) :
    allImages = generateDisplay(inp, boxes)
    views, genImages, _ = allImages
    genImages = unsqeeze4dim(genImages)[0]
    views = views.detach().cpu().numpy()
    nofIm = views.shape[0]
    for curim in range(nofIm) :
        if not DCfg.brick :
            plotImage(genImages[curim,0,...].transpose(-1,-2).cpu().numpy())
        vmin = views[curim,0:3,...].min()
        vmax = views[curim,0:3,...].max()
        plt.figure(frameon=False)
        for id in range(3) :
            plt.subplot(1, 4, id + 1)
            plt.imshow(views[curim,id,...], cmap='gray', vmin=vmin, vmax=vmax)
            plt.axis("off")
        vmm = max( -views[curim,3,...].min(), views[curim,3,...].max() )
        plt.subplot(1, 4, 4)
        plt.imshow(views[curim,3,...], cmap='gray', vmin=-vmm, vmax=vmm)
        plt.axis("off")
        plt.show()
    return allImages



def calculateWeights(images) :
    return None

def calculateNorm(images) :
    noGapImages = torch.cat( (images[...,:DCfg.gapRngX.start], images[...,DCfg.gapRngX.stop:]), dim=-1)
    toRet = torch.std_mean( noGapImages, dim=(-1,-2), keepdim=True )
    return toRet[0], toRet[1]

epoch=initIfNew('epoch', 0)
imer = initIfNew('iter', 0)
minGEpoch = initIfNew('minGEpoch')
minGLoss = initIfNew('lastGLoss')
startFrom = initIfNew('startFrom', 0)


def saveCheckPoint(path, epoch=epoch, imer=imer, interimRes=TrainResClass()) :
    global  minGEpoch, minGLoss, generator, discriminator, optimizers_G, optimizers_D
    checkPoint = {}
    checkPoint['epoch'] = epoch
    checkPoint['iterations'] = imer
    checkPoint['minGEpoch'] = minGEpoch
    checkPoint['lastGLoss'] = minGLoss
    checkPoint['startFrom'] = startFrom
    checkPoint['generator'] = generator.state_dict()
    if discriminator is not None :
        checkPoint['discriminator'] = discriminator.state_dict()
    for idx, optim in enumerate(optimizers_G) :
        checkPoint[f"optimizerGen_{idx}"] = optim.state_dict()
    for idx, optim in enumerate(optimizers_D) :
        checkPoint[f"optimizerDis_{idx}"] = optim.state_dict()
    checkPoint['resAcc'] = interimRes
    torch.save(checkPoint, path)


def loadCheckPoint(path) :
    global minGEpoch, minGLoss, generator, discriminator, optimizers_G, optimizers_D
    checkPoint = torch.load(path, map_location='cpu', weights_only=False)
    epoch = checkPoint['epoch']
    iterations = checkPoint['iterations']
    minGEpoch = checkPoint['minGEpoch']
    minGLoss = checkPoint['lastGLoss']
    startFrom = checkPoint['startFrom'] if 'startFrom' in checkPoint else 0
    generator.load_state_dict(checkPoint['generator'])
    if discriminator is not None :
        discriminator.load_state_dict(checkPoint['discriminator'])
    for idx, optim in enumerate(optimizers_G) :
        key = f"optimizerGen_{idx}"
        if key in checkPoint :
            try : optim.load_state_dict(checkPoint[key])
            except :
                print("Failed to load optimizer " + key)
                continue
        else :
            print("No optimizer " + key + " in chekpoint")
    for idx, optim in enumerate(optimizers_D) :
        key = f"optimizerDis_{idx}"
        if key in checkPoint :
            try : optim.load_state_dict(checkPoint[key])
            except :
                print("Failed to load optimizer " + key)
                continue
        else :
            print("No optimizer " + key + " in chekpoint")
    interimRes = checkPoint['resAcc'] if 'resAcc' in checkPoint else TrainResClass()
    return epoch, iterations, minGEpoch, minGLoss, startFrom, interimRes


def restoreCheckpoint(path=None, logDir=None) :
    if logDir is None :
        logDir = TCfg.logDir
    if path is None :
        if os.path.exists(logDir) :
            raise Exception(f"Starting new experiment with existing log directory \"{logDir}\"."
                            " Remove it .")
        try : os.remove(TCfg.historyHDF)
        except : pass
        return 0, 0, 0, None, 0, TrainResClass()
    else :
        return loadCheckPoint(path)


def saveModels(path="") :
    save_model(generator, model_path = ( path if path else f"model_{TCfg.exec}" ) + "_gen.pt" )
    if discriminator is not None :
        save_model(discriminator, model_path = ( path if path else f"model_{TCfg.exec}" ) + "_dis.pt"  )




















repeatDis = 1
repeatGen = 1

def doTrainDis(locals) :
    return True

def doTrainGen(locals) :
    return True



def criteriaToFollow() :
    image = refImages[[2],...]
    box = refBoxes[2]
    rng = np.s_[0, 0, box : box + DCfg.sinoSh[-1], DCfg.gapRngX]
    #rng = np.s_[0, 0, box : box + DCfg.sinoSh[-1], :]
    with torch.no_grad() :
        genImage = generator.forward(image).to(image.device)
        crit = L1L(genImage[rng], image[rng]).sum().item() / metrices['L1L'].norm
    return crit

@dataclass
class Follower:
    index : tuple = ()
    deltaScore : float = 0
    ratioScore : float = 0
    tests : int = 0
    def deltaAverage(self) : return self.deltaScore / self.tests if self.tests else 0
    def ratioAverage(self) : return self.ratioScore / self.tests if self.tests else 0

followers = None
mixedInFollowers = 0

def mixInFollowers(data) :
    global followers, mixedInFollowers
    if followers is None or not mixedInFollowers or not len(followers):
        return data
    return data

def dealWithTheFollowers(images, indeces, criteriaBefore, criteriaAfter) :
    if followers is None :
        return
    return



lrCoeff = 1
save_minimal = True
minimal_criteria = None
saved_criteria = None
last_criteria = None
def updateCriteria(saveMe=True) :
    global minimal_criteria, last_criteria, saved_criteria
    last_criteria = criteriaToFollow()
    writer.add_scalars("Aux", {'Crit': last_criteria}, imer)
    if minimal_criteria is None or (last_criteria < minimal_criteria) :
        print(f"New best criteria: {last_criteria:.6e}.")
        minimal_criteria = last_criteria
        if saveMe and (saved_criteria is None or minimal_criteria < 0.998 * saved_criteria) :
            image = refImages[[2],...]
            with torch.no_grad() :
                genImage = generator.forward(image).to(image.device)
                preImage = generator.lowResProc(image).to(image.device)
                svImage = torch.cat( [ normalizeImages(img)[0].detach().cpu() for img in
                                  ( image, preImage, genImage, genImage-preImage, image - genImage ) ] , dim=-1 )
            saveCheckPoint(f"checkPoint_{TCfg.exec}_mini.pth", epoch=epoch-1, imer=imer)
            save_model(generator, f"model_{TCfg.exec}_gen_mini.pt")
            tifffile.imwrite(f"mini_{TCfg.exec}.tif", svImage[0,0,...].transpose(-1,-2).numpy())
            saved_criteria = minimal_criteria
    return last_criteria



def train_step(allImages):
    global skipGen, skipDis, followers, save_minimal

    trainRes = TrainResClass()
    allImages, _ = unsqeeze4dim(allImages)
    allImages.requires_grad_(True)
    nofAllIm = allImages.shape[0]

    while trainRes.nofIm < nofAllIm :

        images = allImages[ trainRes.nofIm : trainRes.nofIm + TCfg.batchSize , ... ]
        nofIm = images.shape[0]
        trainRes.nofIm += nofIm
        batchSplit = TCfg.batchSplit if TCfg.batchSplit > 1 else 1
        subBatchSize = nofIm // batchSplit

        # train discriminator
        if 'Adv' in metrices and  metrices['Adv'].weight > 0 :
            if repeatDis :
                for _ in range(repeatDis) :
                    for optim in optimizers_D :
                        optim.zero_grad(set_to_none=False)
                    for i in range(batchSplit) :
                        subRange = np.s_[i*subBatchSize:(i+1)*subBatchSize]
                        subImages = images[subRange,...]
                        with torch.no_grad() : # create fake images to be descriminated
                            subFakeImages = generator.generateImages(subImages)
                        #subImages.requires_grad = True
                        #subFakeImages.requires_grad = True
                        disLoss, probs = loss_Dis(subImages, subFakeImages)
                        trainRes.predReal += probs[:subBatchSize,0].sum().item() / repeatDis
                        trainRes.predFake += probs[subBatchSize:,0].sum().item() / repeatDis
                        trainRes.lossD += disLoss.item() / repeatDis
                        if doTrainDis(locals()) :
                            disLoss = disLoss / subBatchSize
                            disLoss.backward()
                    for optim in optimizers_D :
                        optim.step()
                        optim.zero_grad(set_to_none=True)
            else :
                with torch.no_grad() : # create fake images to be descriminated
                    for i in range(batchSplit) :
                        subRange = np.s_[i*subBatchSize:(i+1)*subBatchSize]
                        subImages = images[subRange,...]#.clone().detach()
                        subFakeImages = generator.generateImages(subImages)
                        disLoss, probs = loss_Dis(subImages, subFakeImages.detach())
                        trainRes.predReal += probs[:subBatchSize,0].sum().item()
                        trainRes.predFake += probs[subBatchSize:,0].sum().item()
                        trainRes.lossD += disLoss.item()

        # train generator
        for _ in range(repeatGen) :
            for optim in optimizers_G :
                optim.zero_grad(set_to_none=False)
            for i in range(batchSplit) :
                subRange = np.s_[i*subBatchSize:(i+1)*subBatchSize]
                subImages = images[subRange,...]
                #with torch.no_grad():
                #    subFakeImages = fillTheGap(subImages, torch.zeros_like(subImages[:,[0],*generator.cfg.gapRng]))
                subFakeImages = generator.generateImages(subImages)
                genLoss, indLosses = loss_Gen(subImages, subFakeImages)
                trainRes.lossG += genLoss.item() / repeatGen
                for key in indLosses.keys() :
                    trainRes.metrices[key] += indLosses[key] / repeatGen
                genLoss = genLoss / subBatchSize
                if doTrainGen(locals()) :
                    genLoss.backward()
            for optim in optimizers_G :
                optim.step()
                optim.zero_grad(set_to_none=True)



    if minimal_criteria is not None :
        updateCriteria(save_minimal)

    del genLoss
    del indLosses
    return trainRes



def beforeEachEpoch(locals) :
    return

def afterEachEpoch(locals) :
    return

def beforeReport(locals) :
    return

def afterReport(locals) :
    return

def preTransformImage(images):
    return images.to(generator.device())

trainLoader=None
testLoader=None
resAcc = TrainResClass()
revert_minimal_criteria = None
correlatedCriteriaFile = None

def train(savedCheckPoint, epochSize=None):
    global epoch, minGLoss, minGEpoch, startFrom, imer, resAcc
    global minimal_criteria, lrCoeff, followers
    global trainLoader, testLoader, testSet, refImages, minMetrices, maxMetrices
    lastGLoss = minGLoss

    lastUpdateTime = time.time()
    lastSaveTime = time.time()
    initLRs = { sched : sched.get_last_lr()[0] / TCfg.learningRateG for sched in schedulers_G }

    while TCfg.nofEpochs is None or epoch <= TCfg.nofEpochs :
        epoch += 1
        beforeEachEpoch(epoch)

        trainLoader = createDataLoader(trainSet, num_workers=TCfg.num_workers)
        testLoader = createDataLoader(testSet, num_workers=TCfg.num_workers)
        #generator.train()
        #discriminator.train()
        resAcc = TrainResClass()
        updAcc = TrainResClass()
        #_ = trackExtremes()

        localMinima = 1e100
        prev_data = None
        repCounter = 0
        total = len(trainLoader)
        if epochSize is not None :
            total = min(total,epochSize)
        for it , data in tqdm.tqdm(enumerate(trainLoader), total=total):
            if epochSize is not None and resAcc.nofIm >= epochSize * TCfg.batchSize :
                break

            if followers is not None:
                pass
            elif prev_data is not None :
                data = prev_data
                criteriaBefore = criteriaAfter
            else :
                criteriaBefore = criteriaToFollow()

            #if followers is not None :
            #    data = mixInFollowers(data)
            #    criteriaBefore = criteriaToFollow()
            images = data[0]
            images = preTransformImage(images)
            imer += images.shape[0]

            # acrtual training
            trainRes = train_step(images)
            resAcc += trainRes
            updAcc += trainRes

            if followers is not None or repCounter > 32 :
                prev_data = None
                repCounter = 0
            else :
                criteriaAfter = criteriaToFollow()
                if criteriaAfter < criteriaBefore  :
                    if repCounter == 0 :
                        prev_data = data
                        initLRs = { sched : sched.get_last_lr()[0] / TCfg.learningRateG for sched in schedulers_G }
                        for sched in schedulers_G :
                            sched.gamma = 1-0.01
                            torch.optim.lr_scheduler.LambdaLR(sched.optimizer, lambda epoch: initLRs[sched]/2).step()
                    repCounter += 1
                    localMinima = min( criteriaAfter, localMinima )
                elif repCounter and ( criteriaAfter < localMinima * 1.01 ) :
                    repCounter += 1
                else :
                    prev_data = None
                    if repCounter > 0 :
                        localMinima = 1e100
                        for sched in schedulers_G :
                            sched.gamma = 1-0.002
                            torch.optim.lr_scheduler.LambdaLR(sched.optimizer, lambda epoch: initLRs[sched]).step()
                    repCounter = 0


            #if followers is not None :
            #    criteriaAfter = criteriaToFollow()
            #    dealWithTheFollowers(images, data[1], criteriaBefore, criteriaAfter)


            # Per minute update:
            if time.time() - lastUpdateTime > 60  or imer == images.shape[0]:

                # generate previews
                #generator.eval()
                if None in [ minMetrices, maxMetrices ] or \
                   not 'loss' in minMetrices or not 'loss' in maxMetrices or \
                   None in [ minMetrices['loss'], maxMetrices['loss'] ] :
                    extImages = images[random.sample(range(images.shape[0]), 2),...]
                else :
                    extImages = torch.stack((maxMetrices['loss'][1],minMetrices['loss'][1]))
                extViews, extGen, _ = generateDisplay(extImages)
                extViews = extViews.detach().cpu().numpy()
                refViews, genImages, _ = generateDisplay()
                refViews = refViews.detach().cpu().numpy()
                rndIndeces = random.sample(range(images.shape[0]), 2)
                rndViews, rndGen, _ = generateDisplay(images[rndIndeces,...])
                rndViews = rndViews.detach().cpu().numpy()
                #generator.train()

                IPython.display.clear_output(wait=True)
                beforeReport(locals())
                print(f"Epoch: {epoch} ({minGEpoch}). ", end=' ')
                print(updAcc)
                updAcc *= 1/updAcc.nofIm
                for key in updAcc.metrices.keys() :
                    if metrices[key].norm > 0 :
                        writer.add_scalars("Metrices per iter", {key : updAcc.metrices[key],}, imer )
                writer.add_scalars("Losses per iter", {'Gen': updAcc.lossG}, imer )
                if discriminator is not None :
                    writer.add_scalars("Losses per iter", {'Dis': updAcc.lossD}, imer )
                    writer.add_scalars("Probs per iter",
                                       {'Ref':updAcc.predReal
                                       ,'Gen':updAcc.predFake
                                       #,'Pre':trainRes.predGen
                                       }, imer )

                subLay = (1,1)
                def addSubplot(pos, img, sym=True) :
                    nonlocal subLay
                    plt.subplot(*subLay, pos) ;
                    vmm = max( -img.min(), img.max() )
                    plt.imshow(img, cmap='gray', vmin = -vmm if sym else None,
                                                 vmax =  vmm if sym else None)
                    plt.axis("off")

                if not DCfg.brick :
                    subLay = (3,1)
                    plt.figure(frameon=True, layout='compressed', facecolor=(0,0.3,0.5))
                    #plt.subplots_adjust(hspace=0.5, wspace=0)
                    addSubplot(1, rndGen[0,0,...].transpose(-1,-2).detach().cpu().numpy(), False)
                    addSubplot(2, genImages[0,0,...].transpose(-1,-2).detach().cpu().numpy(), False)
                    addSubplot(3, extGen[0,0,...].transpose(-1,-2).detach().cpu().numpy(), False)
                    plt.show()

                subLay = (2,5)
                plt.figure(frameon=True, layout='compressed', facecolor=(0,0.3,0.5))
                #plt.subplots_adjust(hspace = 0.1, wspace=0.1)
                addSubplot( 1, extViews[0,2],False)
                addSubplot( 2, rndViews[0,0],False)
                addSubplot( 3, rndViews[0,1],False)
                addSubplot( 4, refViews[3,3])
                addSubplot( 5, refViews[2,3])
                addSubplot( 6, extViews[0,3])
                addSubplot( 7, rndViews[0,2],False)
                addSubplot( 8, rndViews[0,3])
                addSubplot( 9, refViews[1,3])
                addSubplot(10, refViews[0,3])
                plt.show()


                afterReport(locals())
                lastUpdateTime = time.time()
                updAcc = TrainResClass()
                #_ = trackExtremes()

            # Per hour save:
            if time.time() - lastSaveTime > 3600 :

                lastSaveTime = time.time()
                saveCheckPoint(savedCheckPoint+"_hourly.pth", epoch=epoch-1, imer=imer, interimRes=resAcc)
                saveModels(f"model_{TCfg.exec}_hourly")

                if minimal_criteria is not None :
                    if revert_minimal_criteria is not None :
                        sv_lr = schedulers_G[0].get_last_lr()[0] / TCfg.learningRateG
                        #createOptimizers()
                        #freeGPUmem()
                        _ = restoreCheckpoint(f"checkPoint_{TCfg.exec}_mini.pth")
                        #freeGPUmem()
                        for optim in optimizers_G :
                            torch.optim.lr_scheduler.LambdaLR(optim, lambda epoch:sv_lr).step()
                        #freeGPUmem()
                    #elif revert_minimal_criteria is not None :
                    #    revert_minimal_criteria = minimal_criteria

        # Per epoch update:
        if True :
            # summary of the epoch
            print(resAcc)
            resAcc *= 1/resAcc.nofIm
            for key in resAcc.metrices.keys() :
                if metrices[key].norm > 0 :
                    writer.add_scalars("Metrices per epoch", {key : resAcc.metrices[key],}, epoch )
            writer.add_scalars("Losses per epoch",{'Gen': resAcc.lossG,}, epoch )
            if discriminator is not None :
                writer.add_scalars("Losses per epoch",{'Dis': resAcc.lossD}, epoch )
                writer.add_scalars("Probs per epoch",
                                   {'Ref':resAcc.predReal
                                   ,'Gen':resAcc.predFake
                                   #,'Pre':trainRes.predGen
                                   }, epoch )

            # reference views after epoch
            def overview():
                with torch.no_grad() :
                    if 'Adv' in metrices and  metrices['Adv'].weight > 0 :
                        probs_ref = []
                        progs_gen = []
                        for idx in range(refImages.shape[0]) :
                            probs_ref.append(discriminator.forward(refImages[[idx],...]).view(-1).item())
                            progs_gen.append(discriminator.forward(generator.generateImages(refImages[[idx],...])).view(-1).item())
                        print(probs_ref)
                        print(progs_gen)
                    displayImages()
            try :
                generator.train()
                print("Reference images in train mode:")
                overview()
                generator.eval()
                print("Reference images in eval mode:")
                overview()
            except Exception as e:
                print(e)
                continue


            # save current models
            saveModels()
            os.system(f"mv {savedCheckPoint}.pth {savedCheckPoint}_previous.pth")
            saveCheckPoint(savedCheckPoint+".pth", epoch=epoch, imer=imer)

            # post-epoch test summary
            try :
                resTest = summarizeMe(testLoader, False)
                resTest *= 1/resTest.nofIm
                for key in resTest.metrices.keys() :
                    if metrices[key].norm > 0 :
                        writer.add_scalars("Metrices epoch test", {key : resTest.metrices[key],}, epoch )
                writer.add_scalars("Losses epoch test",{'Gen': resTest.lossG}, epoch )
                if discriminator is not None :
                    writer.add_scalars("Losses epoch test",{'Dis': resTest.lossD}, epoch )
                    writer.add_scalars("Probs epoch test",
                        {'Ref':resTest.predReal
                        ,'Gen':resTest.predFake
                        }, epoch )
            except Exception as e:
                print(e)
                continue

            # saving if test is the best so far
            lastGLoss = resTest.lossG # Rec_test
            if minGLoss is None or minGLoss == 0 or lastGLoss < minGLoss :
                minGLoss = lastGLoss
                minGEpoch = epoch
                os.system(f"cp {savedCheckPoint}.pth {savedCheckPoint}_best.pth")
                os.system(f"cp {savedCheckPoint}_previous.pth {savedCheckPoint}_beforebest.pth")
                #saveModels(f"model_{TCfg.exec}_B")

        # reset and other post-epoch actions
        resAcc = TrainResClass()
        afterEachEpoch(locals())
        print("Epoch completed.\n")






def freeGPUmem() :
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()






