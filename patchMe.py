#!/usr/bin/env python3

import os, sys
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import argparse
import numpy as np
import torch
import math
import tqdm

import commonsource as cs
from models.shiftpatch_mini import loadMe as model

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

torch.set_grad_enabled(False)


parser = argparse.ArgumentParser(description=
    'Fill and stitch shift-in-scan projections.')
parser.add_argument('original', type=str, default="",
                    help='Input stack of frames in original position.')
parser.add_argument('shifted', type=str, nargs='?', default="",
                    help='Input stack of frames in shifted position.')
parser.add_argument('output', type=str, nargs='?', default="",
                    help='Output HDF file.')
parser.add_argument('-s', '--stitch', default="", type=str,
                    help='Output of stitch.sh script run with no arguments.'
                         ' Reads from stdin if omited')
parser.add_argument('-m', '--mask', default=[], action='append',
                    help='Mask in the original and (if given) shifted position.')
parser.add_argument('-c', '--crop', default="", type=str,
                    help='Additionally crop final image.')
parser.add_argument('-v', '--verbose', action='store_true', default=False,
                    help='Plot results.')
args = parser.parse_args()


dataO = cs.getInData(args.original)
dataS = cs.getInData(args.shifted) if len(args.shifted) else dataO
face = dataO.shape[-2:]
if dataS.shape[-2:] != face :
    print("Error! Input stacks have different sizes."
          f" {args.original}: {dataO.shape} and {args.shifted} :{dataS.shape}.")
    exit(1)
maskO = cs.loadImage(args.mask[0]) if len(args.mask) else np.ones(face)
maskO = cs.stretchImage(maskO)
maskS = cs.loadImage(args.mask[1]) if len(args.mask) > 1 else maskO
maskS = cs.stretchImage(maskS)

if len(args.stitch) :
    sys.stdin = open(args.stitch, 'r')
directPair = None
flipedPair = None
pairs = []
try:
    while True:
        strg = input()
        indeces = tuple([int(el) for el in strg.split()[0:2]])
        flip = "-f" in strg
        pairs.append( indeces + (flip,) )
        if directPair is None and not flip:
            directPair = indeces
        if flipedPair is None and flip:
            flipedPair = indeces
        #if directPair is not None and flipedPair is not None:
        #    break
except EOFError:
    pass
if directPair is None or flipedPair is None:
    raise Exception("Failed to find direct and/or flipped pairs in the input stream.")

verbose = int(args.verbose) * 2
if args.verbose :
    print("Estimating shift.")
shifts = cs.findShift(dataO[directPair[0]], dataS[directPair[1]],
                      maskO, maskS, amplitude=100, verbose=int(args.verbose)*2)
shiftDir =  tuple( round(np.median( [ sft[0][0][dim] for sft in shifts] )) for dim in (-2,-1) )
#shiftDir = (-73, 23)
exit(0)
if args.verbose :
    print(f"Shift estimation: {shiftDir}")
    print("Estimating rotation centre.")
#shifts = cs.findShift(dataO[flipedPair[0]], np.flip(dataS[flipedPair[1]], axis=-1).copy(),
#                      maskO, np.flip(maskS, axis=-1).copy(),
#                      amplitude=(6,100), start=shiftDir, verbose=int(args.verbose)*2)
#shiftFlp =  tuple( shiftDir[dim] + round(np.median( [ sft[0][0][dim] for sft in shifts] )) for dim in (-2,-1) )
shiftFlp = (-71, -32)
rotCent = round( ( shiftDir[-1] + shiftFlp[-1] ) / 2 )
if args.verbose :
    print(f"Rotation cenre estimation: {rotCent} from shift {shiftFlp}.")
maxDiv = max( abs(shiftDir[1]), abs(shiftFlp[1]) )
finalSh = ( face[0] - abs(shiftDir[0]) , face[1] - 2*maxDiv )
orgPos = ( max(shiftDir[0],0), maxDiv )

outPut = cs.OutputWrapper(args.output, (len(pairs), *finalSh) )

generator = model.createGenerator(cs.device)
msh = model.inShape
genWghts = torch.empty( msh, device = cs.device )
for d0 in range(msh[0])  :
    for d1 in range(msh[1]) :
        genWghts[d0,d1] = math.exp( -4 * ((1-2*d1/(msh[1]-1))**2 + (1-2*d0/(msh[0]-1))**2 ))
genWghts = genWghts[None,...]


btSh = model.inShape
batch = torch.empty( ( maxBatchSize, 4, *btSh), device = cs.device )
def fillInGaps(inO, inS, mskO, mskS) :

    inStack = torch.stack((inO, inS, mskO, mskS), dim=0).to(cs.device)
    inSh = inStack.shape[-2:]
    outStack = inStack.clone()
    outStack[0:2,...] *= outStack[2:4,...]
    batchCounter = 0
    batchRanges = []
    nofPatches = 0

    def procBatch(force=False) :
        nonlocal batchCounter, batchRanges, nofPatches
        if batchCounter < maxBatchSize and not force :
            return
        with torch.no_grad() :
            results = generator.forward((batch[:batchCounter,...],None))
        masks = batch[:batchCounter,2:4,...]
        for flIdx in range(batchCounter) :
            ranges = batchRanges[flIdx]
            outStack[0:2,*ranges] += results[flIdx,...] * (1-masks[flIdx,...]) * masks[flIdx,[1,0],...] * genWghts
            outStack[2:4,*ranges] += (1-masks[flIdx,...]) * masks[flIdx,[1,0],...] * genWghts
        nofPatches += batchCounter
        batchCounter = 0
        batchRanges = []

    def procBrick(roi) :
        nonlocal batchCounter, batchRanges
        if torch.any(mskO[*roi]<1) or torch.any(mskS[*roi]<1) :
            batchRanges.append(roi)
            batch[batchCounter,...] = inStack[:,*roi]
            batchCounter += 1
            procBatch()

    step = tuple( btSh[dim]//2 - 1 for dim in (0,1) )
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
    outStack[0:2] *= torch.where( outStack[2:4] > 0 , 1/outStack[2:4] , 0 )
    #print(nofPatches, math.ceil(nofPatches / btSz))
    return outStack[0:2]


pbar = tqdm.tqdm(total=len(pairs)) if verbose else None
for idx, pair in enumerate(pairs) :

    #if idx < 9365 : # for debug only
    #    continue

    # find relative shift
    indeces = pair[0:2]
    flip = pair[2]
    inO = torch.tensor( dataO[indeces[0]] )
    inS = torch.tensor( np.flip(dataS[indeces[1]], axis=-1).copy() if flip else dataS[indeces[1]] )
    mskO = torch.tensor(maskO)
    mskS = torch.tensor( np.flip(maskS, axis=-1).copy() if flip else maskS )
    shift = shiftFlp if flip else shiftDir
    corr = cs.findShift(inO, inS, mskO, mskS, amplitude=6, start=shift, verbose=False )

    # fill the patches
    corr =  tuple( round(np.median( [ cr[0][0][dim] for cr in corr] )) for dim in (-2,-1) )
    shift =  tuple( shift[dim] + corr[dim] for dim in (0,1) )
    subO =  tuple( np.s_[ max(0,  shift[dim]) : face[dim] + min( shift[dim], 0) ] for dim in (0,1)  )
    subS =  tuple( np.s_[ max(0, -shift[dim]) : face[dim] + min(-shift[dim], 0) ] for dim in (0,1)  )
    filledPair = fillInGaps( inO[*subO], inS[*subS], mskO[*subO], mskS[*subS])
    stitched = ( filledPair[0,...] + filledPair[1,...] ) / 2

    # populate output
    cornerPos = tuple( orgPos[dim] - subO[dim].start for dim in (0,1) )
    startRes =  tuple( max(0,  cornerPos[dim]) for dim in (0,1) )
    startOut =  tuple( max(0, -cornerPos[dim]) for dim in (0,1) )
    roiSh =  tuple( min( stitched.shape[dim] - startRes[dim], finalSh[dim] - startOut[dim] ) for dim in (0,1) )
    subRes =  tuple( np.s_[ startRes[dim] : startRes[dim] +roiSh[dim] ] for dim in (0,1) )
    subOut =  tuple( np.s_[ startOut[dim] : startOut[dim] +roiSh[dim] ] for dim in (0,1) )
    toOut = torch.zeros(finalSh, device=cs.device)
    toOut[subOut] = stitched[subRes]
    outPut.put(toOut.cpu().numpy(), idx)

    if pbar is not None :
        pbar.update()





