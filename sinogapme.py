#!/usr/bin/env python3

#import sys
import os
#import random
#import time
#from dataclasses import dataclass
#from enum import Enum

#import statistics
import numpy as np
import torch
from torchvision import transforms
#import torch.nn as nn
#import torch.nn.functional as fn
#from torch import optim
#from torchinfo import summary

import argparse
import h5py
import tifffile
import tqdm

myPath = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description=
    'Fill sinograms with sinogap NN.')
parser.add_argument('input', type=str, default="",
                    help='Input stack of CT projections to fill.')
parser.add_argument('output', type=str, default="",
                    help='Output filled stack.')
parser.add_argument('-m', '--mask', type=str, default="",
                    help='Mask of the input stack.')
parser.add_argument('-M', '--model', type=str, default="",
                    help='Mask of the input stack.')
parser.add_argument('-v', '--verbose', action='store_true', default=False,
                    help='Plot results.')
args = parser.parse_args()

if args.model in ["", "default", "3", "mse"] :
    from models.model2 import loadMe as model
elif args.model in ["2", "adv"] :
    from models.model3 import loadMe as model
else :
    raise Exception(f"Unknown model \"{args.model}\" given via -M/--model option.", )






device = torch.device('cuda:0')
maxBatchSize = 0
try:
    localCfgDict = dict()
    exec(open(os.path.join(myPath, ".local.cfg")).read(),localCfgDict)
    if 'torchdevice' in localCfgDict :
        device = torch.device(localCfgDict['torchdevice'])
    if 'maxBatchSize' in localCfgDict :
        maxBatchSize = localCfgDict['maxBatchSize']
except KeyError:
    raise
except:
    pass

#%% FUNCS




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
    dataN = np.empty(data.shape, dtype=np.float32)
    if args.verbose :
        print("Reading input ... ", end="", flush=True)
    data.read_direct(dataN)
    if args.verbose :
        print("Done.")
    return dataN


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


class OutputWrapper:

    def __init__(self, outputString, shape):
        if len(shape) != 3 :
            raise Exception(f"Not appropriate output array size {shape}.")
        nameSplit = outputString.split(':')
        self.trgH5F = None
        if len(nameSplit) == 1 : # tiff image
            if shape[1] != 1 :
                raise Exception(f"Cannot save 3D data {shape} from input to a tiff file.")
            self.TiffName = nameSplit[0]
            return
        self.TiffName = None
        if len(nameSplit) != 2 :
            raise Exception(f"String \"{outputString}\" does not represent an HDF5 format \"fileName:container\".")
        hdfName = nameSplit[0]
        hdfVolume = nameSplit[1]
        try :
            self.trgH5F =  h5py.File(hdfName,'w')
        except :
            raise Exception(f"Failed to open HDF file '{hdfName}'.")
        if  hdfVolume not in self.trgH5F.keys():
            self.dset = self.trgH5F.create_dataset(hdfVolume, shape, dtype='f')
        else :
            self.dset = self.trgH5F[hdfVolume]
            csh = self.dset.shape
            if csh[0] < shape[0] or csh[1] != shape[1] or csh[2] != shape[2] :
                raise Exception(f"Shape mismatch: input {shape}, output HDF {self.dset.shape}.")

    def __del__(self):
        if self.trgH5F is not None :
            self.trgH5F.close()

    def put(self, data, slice):
        if len(data.shape) != 2 :
            raise Exception(f"Output accepts 2D data. Got shape {data.shape} instead.")
        if self.TiffName is not None :
            tifffile.imwrite(self.TiffName, data)
            if slice != 0 :
                Warning(f"Output is a tiff file, but non-zero slice {slice} is saved.")
        elif self.trgH5F is not None :
            self.dset[:,slice,:] = data
        else :
            raise Exception(f"Output is not defined. Should never happen.")

#%% MODELS

generator = model.createGenerator(device)

#%% EXEC

def fillSinogram(sinogram) :

    def unsqeeze4dim(tens):
        orgDims = tens.dim()
        if tens.dim() == 2 :
            tens = tens.unsqueeze(0)
        if tens.dim() == 3 :
            tens = tens.unsqueeze(1)
        return tens, orgDims

    gapW = model.DCfg.gapW
    gapSh = model.DCfg.gapSh
    ssh = model.DCfg.sinoSh

    sinoW = sinogram.shape[-1]
    sinoL = sinogram.shape[-2]
    if sinoW % 5 :
        raise Exception(f"Sinogram width {sinoW} is not devisable bny 5.")
    blockW = sinoW // 5
    sinogram, _ = unsqeeze4dim(sinogram)
    sinogram = sinogram.to(device)
    resizedSino = torch.zeros(( 1 , 1 , sinoL , ssh[1] ), device=device)
    resizedSino[ ... , : 2*gapW ] = torch.nn.functional.interpolate(
        sinogram[ ... , : 2*blockW ], size=( sinoL , 2*gapW ), mode='bilinear')
    resizedSino[ ... , 2*gapW : 3*gapW ] = torch.nn.functional.interpolate(
        sinogram[ ... , 2*blockW : 3*blockW ], size=( sinoL , gapW ), mode='bilinear')
    resizedSino[ ... , 3*gapW:] = torch.nn.functional.interpolate(
        sinogram[ ... , 3*blockW : ], size=( sinoL , 2*gapW ), mode='bilinear')

    blockH = ssh[0]
    sinoCutStep = gapW
    lastStart = sinoL - blockH
    nofBlocks, lastBlock = divmod(lastStart, sinoCutStep)
    totBlocks = nofBlocks + bool(lastBlock)
    modelIn = torch.empty( ( totBlocks , 1 , *ssh ), device=device )
    for block in range(nofBlocks) :
        modelIn[ block, 0, ... ] = resizedSino[0 , 0, block * sinoCutStep : block * sinoCutStep + blockH , : ]
    if lastBlock :
        modelIn[ -1, 0, ... ] = resizedSino[0,0, -blockH : , : ]

    mytransforms = transforms.Compose([
        transforms.Normalize(mean=(0.5), std=(1))
    ])
    modelIn = mytransforms(modelIn)

    modelIn[ -1, 0, ... ] = modelIn[ -1, 0, ... ].flip(dims=(-2,)) # to get rid of the deffect in the end
    results = torch.zeros( ( totBlocks , 1 , *gapSh ), device=device )
    batchSize = min(maxBatchSize, totBlocks) if maxBatchSize else totBlocks
    with torch.no_grad() :
        for batch in range(0, totBlocks, batchSize) :
            results[ batch:batch+batchSize, ... ] = generator.generatePatches(modelIn[batch:batch+batchSize,...])
        #results = generator.generatePatches(modelIn)
    results[ -1, 0, ... ] = results[ -1, 0, ... ].flip(dims=(-2,)) # to flip back

    if lastBlock :
        newLast = torch.zeros(gapSh, device=device)
        tail = sinoCutStep - lastBlock
        newLast[:-tail,:] = results[-1,0,tail:,:]
        results[-1,0,...] = newLast
    preBlocks = torch.zeros((4,1,*gapSh), device=device)
    pstBlocks = torch.zeros((4,1,*gapSh), device=device)
    for curs in range(4) :
        preBlocks[ -curs-1 , 0 , sinoCutStep*(curs+1) :  , : ] \
            = results[ 0 , 0 , : -sinoCutStep*(curs+1) , : ]
        pstBlocks[ curs , 0 , : (-sinoCutStep*curs) if curs else (blockH+1) , : ] \
            = results[ -1 , 0 , sinoCutStep*curs : , : ]
    resultsPatched = torch.cat( (preBlocks, results, pstBlocks), dim=0 )

    blockCut = blockH / 5
    profileWeight = torch.empty( (blockH,), device=device )
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
    stitchedGap = torch.zeros( ( (resultsProfiled.shape[0]-1) * sinoCutStep + blockH, gapW ), device=model.TCfg.device )
    for curblock in range(resultsProfiled.shape[0]) :
        stitchedGap[ curblock*sinoCutStep : curblock*sinoCutStep + blockH , : ] += resultsProfiled[curblock,0,...]
    stitchedGap = stitchedGap.unsqueeze(0).unsqueeze(0)
    resizedGap = torch.nn.functional.interpolate(
        stitchedGap, size=( stitchedGap.shape[-2] ,  blockW), mode='bilinear')

    sinogram[..., 2*blockW : 3*blockW ] = resizedGap[0,0, sinoCutStep*4 : sinoCutStep*4 + sinoL, : ] / 2
    return sinogram





inData = getInData(args.input)
fsh = inData.shape[1:]
mask = loadImage(args.mask, fsh) if len(args.mask) else None
leftMask = np.ones(fsh, dtype=np.uint8)
outWrapper = OutputWrapper(args.output, inData.shape)


pbar = tqdm.tqdm(total=fsh[-2]) if args.verbose else None
for curSl in range(fsh[-2]):

    inSinogram = torch.tensor(inData[:,curSl,:], device=model.TCfg.device)
    sinoMask =  torch.where( inSinogram.sum(dim=0) == 0 , 0, 1 ) if mask is None else mask[curSl,:]

    gaps = []
    clmn=0
    gapStart=-1
    while clmn < fsh[-1] :
        value = sinoMask[clmn]
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
    outWrapper.put(inSinogram.cpu().numpy(), curSl)
    if pbar is not None:
        pbar.update(1)


leftMask4fill = leftMask.copy()
leftMask4stitch = leftMask.copy()
for row in range(fsh[0]) :
    if np.all(leftMask[row,:]==0) :
        leftMask4fill[row,:] = 1
        leftMask4stitch[row,:] = 0
    else :
        leftMask4fill[row,:] = leftMask[row,:]
        leftMask4stitch[row,:] = 1
leftMask4fill *= 255
leftMask4stitch *= 255
#leftMask *= 255
leftMaskName = ".".join(args.output.split(".")[:-1])+"_mask"
tifffile.imwrite(leftMaskName + ".tif", leftMask4stitch)
if not np.all(leftMask4fill) :
    tifffile.imwrite(leftMaskName + "_left.tif", leftMask4fill)

if pbar is not None:
    pbar.close()
    print("Done")



# %%
