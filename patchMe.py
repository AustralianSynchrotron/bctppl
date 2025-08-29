#!/usr/bin/env python3

import os, sys
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import argparse
import numpy as np
import torch
import math
import tqdm
import re
from torchvision import transforms

import commonsource as cs
from models.shiftpatch_generic import loadMe as model

jitterAmplitude = 6
searchAmplitude = 100
myPath = os.path.dirname(os.path.realpath(__file__))
device = torch.device('cuda:0')
maxBatchSize = model.batchsize
try:
    localCfgDict = dict()
    exec(open(os.path.join(myPath, ".local.cfg")).read(),localCfgDict)
    if 'torchdevice' in localCfgDict :
        device = torch.device(localCfgDict['torchdevice'])
    if 'shiftpatch_maxBatchSize' in localCfgDict :
        maxBatchSize = localCfgDict['shiftpatch_maxBatchSize']
except KeyError:
    raise
except:
    pass
cs.device = device
#_ = model.process(torch.zeros((1),device=cs.device))

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description=
    'Fill and stitch shift-in-scan projections.')
parser.add_argument('original', type=str, default="",
                    help='Input stack of frames in original position.')
parser.add_argument('shifted', type=str, nargs='?', default="",
                    help='Input stack of frames in shifted position.')
parser.add_argument('-s', '--stitch', default="", type=str,
                    help='Output of previously executed pairShift.py.')
parser.add_argument('-m', '--mask', default=[], action='append',
                    help='Mask in the original and (if given) shifted position.')
parser.add_argument('-o', '--output', type=str, default="",
                    help='Output HDF file with the results cropped to ovelapping region.')
parser.add_argument('-w', '--wide', default="", type=str,
                    help='Filename for the results which fit as much as possible informative pixels.')
#parser.add_argument('-c', '--crop', default="", type=str,
#                    help='Additionally crop final image.')
parser.add_argument('-v', '--verbose', action='store_true', default=False,
                    help='Be verbose.')
args = parser.parse_args()

if not len(args.output) :
    raise Exception("Error! No output file name given via -o/--output option.")

# prepare inputs
dataO = cs.getInData(args.original)
dataS = cs.getInData(args.shifted) if len(args.shifted) else dataO
face = dataO.shape[-2:]
if dataS.shape[-2:] != face :
    raise Exception( "Error! Input stacks have different sizes."
                    f" {args.original}: {dataO.shape} and {args.shifted} :{dataS.shape}.")
maskO = cs.loadImage(args.mask[0], face) if len(args.mask) else np.ones(face)
maskO = cs.stretchImage(maskO)
maskS = cs.loadImage(args.mask[1], face) if len(args.mask) > 1 else maskO
maskS = cs.stretchImage(maskS)
maskO = torch.tensor(maskO, device=cs.device)
maskS = torch.tensor(maskS, device=cs.device)
maskF = maskS.flip(dims=(-1,))

# read pairs info
if len(args.stitch) :
    sys.stdin = open(args.stitch, 'r')
else :
    raise Exception("Error! No input data name given via -s/--stitch option.")
pairs = []
maxDiv = (0,0)
try:
    while True:
        strg = input().strip()
        if strg[0] == "#" :
            continue
        rs = strg.split()
        if len(rs) < 5 :
            raise Exception(f"Error! Failed to parse line \"{strg}\".")
        pairs.append( ( (int(rs[0]), int(rs[1])),
                        (int(rs[2]), int(rs[3])),
                        bool(int(rs[4])) ) )
        maxDiv =  ( max( maxDiv[0], abs(int(rs[2])) ) ,
                    max( maxDiv[1], abs(int(rs[3])) ) )
except EOFError:
    pass
except :
    raise
if not len(pairs) :
    raise Exception(f"Error! Empty or corrupt data in file {args.stitch}.")


# prepare output
finalSh = tuple( face[dim] - 2*maxDiv[dim] for dim in (0,1) )
orgPos = maxDiv
toOut = torch.empty(finalSh, device=cs.device)
outPut = cs.OutputWrapper(args.output, (len(pairs), *finalSh) )
if len(args.wide) :
    orgPosW = tuple( maxDiv[dim] + jitterAmplitude for dim in (0,1) )
    finalShW = tuple( face[dim] + 2*orgPosW[dim] for dim in (0,1) )
    toOutW = torch.empty(finalShW, device=cs.device)
    outPutW = cs.OutputWrapper(args.wide, (len(pairs), *finalShW) )
    outWghts = torch.empty(finalShW, device=cs.device)
    mesh = np.mgrid[0:face[0], 0:face[1]]
    inWghts = ( face[0] - abs(2*mesh[0]-face[0]+1) )*( face[1] - abs(2*mesh[1]-face[1]+1) )
    inWghts = torch.tensor(inWghts, device=cs.device)


# load model and create auxilary data
msh = (60,60) # model.inShape
mesh = np.mgrid[0:msh[0], 0:msh[1]].astype(float)
mesh[0,...] = mesh[0,...]/(msh[0]-1)
mesh[1,...] = mesh[1,...]/(msh[1]-1)
#gnW = np.exp( -4 * ((1-2*mesh[1]/(msh[1]-1))**2 + (1-2*mesh[0]/(msh[0]-1))**2 ))
gnW = np.sin( math.pi * mesh[0] )  * np.sin( math.pi * mesh[1] )
genWghts = np.zeros(model.inShape)
genWghts[10:-10, 10:-10] = gnW
genWghts += 1e-6 # unmask edges
genWghts = torch.tensor(genWghts, device=cs.device)[None,...]

mesh = 1 - mesh
fftWR = ( mesh[0] + mesh[1] ) / 2
fsh = tuple(msh[dim]*2 for dim in (0,1) )
fftWghts = np.zeros(fsh)
fftWghts[ :msh[0], :msh[1]] = fftWR
fftWghts[ :msh[0], msh[1]:2*msh[1]] = np.flip(fftWR, axis=-1)
fftWghts[msh[0]:2*msh[0]:, :msh[1]] = np.flip(fftWR, axis=-2)
fftWghts[msh[0]:2*msh[0], msh[1]:2*msh[1]] = np.flip(fftWR, axis=(-1,-2))
#fftWghts = 1 - fftWghts
fftWghts = torch.tensor(fftWghts, device=cs.device)[None,...]


# gap filling function
btSh = model.inShape
batch = torch.empty( ( maxBatchSize, 4, *btSh), device = cs.device )
def fillInGaps(inO, inS, mskO, mskS) :

    ioStack = torch.stack((inO, inS, mskO, mskS), dim=0).to(cs.device)
    inSh = ioStack.shape[-2:]
    #outStack = inStack.clone()
    ioStack[0:2,...] *= ioStack[2:4,...]
    batchCounter = 0
    batchRanges = []
    nofPatches = 0

    def procBatch(force=False) :
        nonlocal batchCounter, batchRanges, nofPatches
        if batchCounter < maxBatchSize and not force :
            return
        inps = batch[:batchCounter,...]
        masks = batch[:batchCounter,2:4,...]
        pmasks = (1-masks[:,[0,1],...]) * masks[:,[1,0],...] * genWghts
        with torch.no_grad() :
            results = model.process(inps)
            results *= pmasks
            #sumr = results.sum(dim=(-1,-2))
            #coef = torch.sqrt( torch.where( sumr[:,0] * sumr[:,1] != 0, sumr[:,0] / sumr[:,1], 1 ) ).view(-1,1,1)
            #results[:,0,...] /= coef
            #results[:,1,...] *= coef
            #fftIn = torch.empty(results.shape[0],2, *fsh, device=cs.device)
            #fftIn[...,  :msh[0], :msh[1]] = results
            #fftIn[...,  :msh[0], msh[1]:2*msh[1]] = results.flip(dims=(-1,))
            #fftIn[..., msh[0]:2*msh[0]:, :msh[1]] = results.flip(dims=(-2,))
            #fftIn[..., msh[0]:2*msh[0], msh[1]:2*msh[1]] = results.flip(dims=(-1,-2))
            #fftFwd = torch.fft.fft2(fftIn, norm="forward")
            #fftMid = fftFwd[:,[0,1],...] * fftWghts + fftFwd[:,[1,0],...] * (1-fftWghts)
            #results_fft = torch.fft.fft2(fftMid, norm="backward").real.flip(dims=(-2,-1))
            #results_fft[:,0,...] *= coef
            #results_fft[:,1,...] /= coef
            #results = results_fft[...,0:msh[0], 0:msh[1]]
        for flIdx in range(batchCounter) :
            ranges = batchRanges[flIdx]
            ioStack[0:2,*ranges] += results[flIdx,...]
            ioStack[2:4,*ranges] += pmasks[flIdx,...]
        nofPatches += batchCounter
        batchCounter = 0
        batchRanges = []

    def procBrick(roi) :
        nonlocal batchCounter, batchRanges
        if torch.any(mskO[*roi]<1) or torch.any(mskS[*roi]<1) :
            batchRanges.append(roi)
            batch[batchCounter,...] = ioStack[:,*roi]
            batchCounter += 1
            procBatch()

    #step = tuple( btSh[dim]//2 - 1 for dim in (0,1) )
    step = (30,30)
    nofSteps = tuple( 1 + (inSh[dim] - btSh[dim]) // step[dim] for dim in (-2,-1) )
    for yidx in range( nofSteps[-2] ) :
        yrng = np.s_[ yidx * step[0] : yidx * step[0]+ btSh[0] ]
        for xidx in range( nofSteps[-1] ) :
            xrng = np.s_[ xidx * step[1] : xidx * step[1]+ btSh[1] ]
            procBrick((yrng,xrng))
    for xidx in range( nofSteps[-1] ) :
        procBrick(np.s_[ -btSh[-2] : , xidx * step[-1] : xidx * step[-1]+ btSh[-1] ])
    for yidx in range( nofSteps[-2] ) :
        procBrick(np.s_[ yidx * step[-2] : yidx * step[-2]+ btSh[-2] , -btSh[-1] : ])
    procBatch(True)
    ioStack[0:2] = torch.where( ioStack[2:4] > 0 , ioStack[0:2]/ioStack[2:4] , 0 )
    #print(nofPatches, math.ceil(nofPatches / btSz))

    return ioStack[0:2]


# actual stitching
if args.verbose :
    print(f"Patching:")
pbar = tqdm.tqdm(total=len(pairs)) if args.verbose else None
for idx, pair in enumerate(pairs) :

    #if idx < 1450 : # for debug only
    #    continue

    # find relative shift
    indeces, shift, flip = pair
    inO = torch.tensor( dataO[indeces[0]], device=cs.device )
    inS = torch.tensor( dataS[indeces[1]], device=cs.device )
    if flip :
        inS = inS.flip(dims=(-1,))
    mskO = maskO
    mskS = maskF if flip else maskS

    # fill the patches
    subO =  tuple( np.s_[ max(0,  shift[dim]) : face[dim] + min( shift[dim], 0) ] for dim in (0,1)  )
    subS =  tuple( np.s_[ max(0, -shift[dim]) : face[dim] + min(-shift[dim], 0) ] for dim in (0,1)  )
    filledPair = fillInGaps( inO[*subO], inS[*subS], mskO[*subO], mskS[*subS])

    # stitch tight
    stitched = torch.clamp(filledPair[0,...]+filledPair[1,...], min=0 ) / 2
    eps = stitched[stitched>0].min() / 100
    stitched += torch.where( mskO[*subO] + mskS[*subS] > 0 , eps , 0 ) # zero is the mask
    cornerPos = tuple( orgPos[dim] - subO[dim].start for dim in (0,1) )
    startRes =  tuple( max(0,  cornerPos[dim]) for dim in (0,1) )
    startOut =  tuple( max(0, -cornerPos[dim]) for dim in (0,1) )
    roiSh =  tuple( min( stitched.shape[dim] - startRes[dim], finalSh[dim] - startOut[dim] ) for dim in (0,1) )
    subRes =  tuple( np.s_[ startRes[dim] : startRes[dim] +roiSh[dim] ] for dim in (0,1) )
    subOut =  tuple( np.s_[ startOut[dim] : startOut[dim] +roiSh[dim] ] for dim in (0,1) )
    toOut[...] = 0
    toOut[subOut] = stitched[subRes]
    outPut.put(toOut.cpu().numpy(), idx)

    # stitch wide
    if len(args.wide) :
        inO[*subO] = filledPair[0]
        inS[*subS] = filledPair[1]
        inO *= inWghts
        inS *= inWghts
        subOf =  tuple( np.s_[ orgPosW[dim] : orgPosW[dim] + face[dim] ] for dim in (0,1)  )
        subSf =  tuple( np.s_[ orgPosW[dim] + shift[dim] : orgPosW[dim] + shift[dim] + face[dim] ] for dim in (0,1)  )
        toOutW[...] = 0
        toOutW[subOf] += inO
        toOutW[subSf] += inS
        maskW = torch.zeros_like(toOutW)
        maskW[subOf] += mskO
        maskW[subSf] += mskS
        toOutW = torch.clamp(toOutW, min=0)
        eps = toOutW[toOutW>0].min()
        toOutW += torch.where( maskW > 0 , eps , 0 ) # zero is the mask
        outWghts[...] = 0
        outWghts[subOf] += inWghts
        outWghts[subSf] += inWghts
        toOutW *= torch.where( outWghts > 0 , 1/outWghts , 0 )
        outPutW.put(toOutW.cpu().numpy(), idx)

    if pbar is not None :
        pbar.update()

if pbar is not None :
    del pbar
if args.verbose :
    print("Done!")
exit(0)


