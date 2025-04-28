#!/usr/bin/env python3

import os, sys
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import argparse
import numpy as np
import torch
import math

import commonsource as cs
from models.shiftpatch import loadMe as model

torch.set_grad_enabled(False)

myPath = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description=
    'Fill sinograms with sinogap NN.')
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
#shifts = cs.findShift(dataO[directPair[0]],
#                      dataS[directPair[1]],
#                      maskO, maskS,
#                      amplitude=100, verbose=int(args.verbose)*2)
#shiftDir = ( round(np.median( [ sft[0][0][-2] for sft in shifts] )) ,
#             round(np.median( [ sft[0][0][-1] for sft in shifts] )) )
shiftDir = (-73, 23)
if args.verbose :
    print(f"Shift estimation: {shiftDir}")
    print("Estimating rotation centre.")
#shifts = cs.findShift(dataO[flipedPair[0]],
#                      np.flip(dataS[flipedPair[1]], axis=-1).copy(),
#                      maskO, np.flip(maskS, axis=-1).copy(),
#                      amplitude=(6,100), start=shiftDir, verbose=int(args.verbose)*2)
#shiftFlp = ( shiftDir[-2] + round(np.median( [ sft[0][0][-2] for sft in shifts] )) ,
#             shiftDir[-1] + round(np.median( [ sft[0][0][-1] for sft in shifts] )) )
shiftFlp = (-71, -32)
rotCent = ( shiftDir[-1] + shiftFlp[-1] ) / 2
if args.verbose :
    print(f"Rotation cenre estimation: {rotCent} from shift {shiftFlp}.")

#exit(0)


generator = model.createGenerator(cs.device)
msh = model.inShape
genWghts = torch.empty( msh, device = cs.device )
for d0 in range(msh[0])  :
    for d1 in range(msh[1]) :
        genWghts[d0,d1] = math.exp( -9 * ((1-2*d1/(msh[1]-1))**2 + (1-2*d0/(msh[0]-1))**2 ))
genWghts = genWghts[None,...]


def fillInGaps(inO, inS, mskO, mskS) :
    #outO = inO * mskO + inS * (1-mskO)
    #outS = inS * mskS + inO * (1-mskS)
    #return outO, outS
    inStack = torch.stack((inO, inS, mskO, mskS), dim=0).to(cs.device)
    inSh = inStack.shape[-2:]
    outStack = inStack.clone()
    outStack[0:2,...] *= outStack[2:4,...]

    btSh = model.inShape
    btSz = model.batchsize
    batch = torch.empty( ( btSz, 4, *btSh), device = cs.device )
    batchCounter = 0
    batchRanges = []

    def procBatch(force=False) :
        nonlocal batchCounter, batchRanges
        if batchCounter < btSz and not force :
            return
        results = generator.forward((batch[:batchCounter,...],None))
        masks = batch[:batchCounter,2:4,...]
        for flIdx in range(batchCounter) :
            ranges = batchRanges[flIdx]
            outStack[0:2,*ranges] += results[flIdx,...] * (1-masks[flIdx,...]) * masks[flIdx,[1,0],...] * genWghts
            outStack[2:4,*ranges] += (1-masks[flIdx,...]) * masks[flIdx,[1,0],...] * genWghts
        batchCounter = 0
        batchRanges = []

    def procBrick(roi) :
        nonlocal batchCounter, batchRanges
        if torch.all(mskO[*roi]>0) and torch.all(mskS[*roi]>0) :
            return
        batchRanges.append(roi)
        batch[batchCounter,...] = inStack[:,*roi]
        batchCounter += 1
        procBatch()

    step = (btSh[0]//5, btSh[1]//5)
    nofSteps = ( (inSh[-2] - btSh[-2]) // step[-2], (inSh[-1] - btSh[-1]) // step[-1] )
    for yidx in range( nofSteps[-2] ) :
        for xidx in range( nofSteps[-1] ) :
            procBrick(np.s_[ yidx * step[-2] : yidx * step[-2]+ btSh[-2] ,
                             xidx * step[-1] : xidx * step[-1]+ btSh[-1] ])
    if inSh[-2] % btSh[-2] :
        for xidx in range( nofSteps[-1] ) :
            procBrick(np.s_[ -btSh[-2] : , xidx * step[-1] : xidx * step[-1]+ btSh[-1] ])
    if inSh[-1] % btSh[-1] :
        for yidx in range( inSh[-2] // btSh[-2] ) :
            procBrick(np.s_[ yidx * step[-2] : yidx * step[-2]+ btSh[-2] , -btSh[-1] : ])
    procBatch(True)
    outStack[0:2] *= torch.where( outStack[2:4] > 0 , 1/outStack[2:4] , 0 )
    return outStack[0:2]



for pair in pairs :

    indeces = pair[0:2]
    flip = pair[2]
    inO = torch.tensor( dataO[indeces[0]] )
    inS = torch.tensor( np.flip(dataS[indeces[1]], axis=-1) if flip else dataS[indeces[1]] )
    mskO = torch.tensor(maskO)
    mskS = torch.tensor(np.flip(maskS, axis=-1) if flip else maskS)
    shift = shiftFlp if flip else shiftDir
    corr = cs.findShift(inO, inS, mskO, mskS, amplitude=6, start=shift,
                        verbose=int(args.verbose)*2 )
    corr = ( round(np.median( [ cr[0][0][-2] for cr in corr] )) ,
             round(np.median( [ cr[0][0][-1] for cr in corr] )) )
    shift = ( shift[0] + corr[0], shift[1] + corr[1])

    subF = np.s_[ max(0,  shift[0]) : face[-2] + min( shift[0], 0) ,
                  max(0,  shift[1]) : face[-1] + min( shift[1], 0) ]
    subS = np.s_[ max(0, -shift[0]) : face[-2] + min(-shift[0], 0) ,
                  max(0, -shift[1]) : face[-1] + min(-shift[1], 0) ]

    filledPair = fillInGaps( inO[*subF], inS[*subS], mskO[*subF], mskS[*subS])
    stitched = ( filledPair[0,...] + filledPair[1,...] ) / 2

    print(shift)




