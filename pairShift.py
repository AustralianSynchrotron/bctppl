#!/usr/bin/env python3

import os, sys
import argparse
import numpy as np
import torch
import tqdm

import commonsource as cs

jitterAmplitude = 6
searchAmplitude = 100
myPath = os.path.dirname(os.path.realpath(__file__))
device = torch.device('cuda:0')
try:
    localCfgDict = dict()
    exec(open(os.path.join(myPath, ".local.cfg")).read(),localCfgDict)
    if 'torchdevice' in localCfgDict :
        device = torch.device(localCfgDict['torchdevice'])
except KeyError:
    raise
except:
    pass
cs.device = device

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description=
    'Find shifts between matching pairs of frames in original and shiftes stacks.')
parser.add_argument('original', type=str, default="",
                    help='Input stack of frames in original position.')
parser.add_argument('shifted', type=str, nargs='?', default="",
                    help='Input stack of frames in shifted position.')
parser.add_argument('-o', '--output', type=str, default="",
                    help='Filename for output.')
parser.add_argument('-s', '--stitch', default="", type=str,
                    help='Output of stitch.sh script run with no arguments.'
                         ' Reads from stdin if omited')
parser.add_argument('-m', '--mask', default=[], action='append',
                    help='Mask in the original and (if second given) shifted position.')
parser.add_argument('-A', '--search-amplitude', default=searchAmplitude, type=int,
                    help='Amplitude to search for shift in the first pair.')
parser.add_argument('-a', '--jitter-amplitude', default=jitterAmplitude, type=int,
                    help='Maximum difference of the shifts from the one found in the first pair.')
parser.add_argument('-v', '--verbose', action='store_true', default=False,
                    help='Plot results.')
args = parser.parse_args()
jitterAmplitude = abs(args.jitter_amplitude)
searchAmplitude = abs(args.search_amplitude)

if not len(args.output) :
    raise Exception("Error! No output file name given via -o/--output option.")

# prepare inputs
dataO = cs.getInData(args.original)
dataS = cs.getInData(args.shifted) if len(args.shifted) else dataO
face = dataO.shape[-2:]
if dataS.shape[-2:] != face :
    raise Exception( "Error! Input stacks have different sizes."
                    f" {args.original}: {dataO.shape} and {args.shifted} :{dataS.shape}.")
maskO = cs.loadImage(args.mask[0]) if len(args.mask) else np.ones(face)
maskO = cs.stretchImage(maskO)
maskS = cs.loadImage(args.mask[1]) if len(args.mask) > 1 else maskO
maskS = cs.stretchImage(maskS)

# prepare output
try :
    outPut = open(args.output, 'w')
except :
    raise Exception(f"Failed to open output file {args.output} for writing.")


# find direct and flipped pairs for first estimation of shift and rotation centre
directPair = None
flipedPair = None
pairs = []
if len(args.stitch) :
    sys.stdin = open(args.stitch, 'r')
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
    if directPair is None:
        raise Exception("Failed to find pairs in the input stream.")
    if flipedPair is None:
        flipedPair = directPair
else :
    directPair = flipedPair = (0,0)
    for idx in range( min(dataO.shape[0], dataS.shape[0]) ) :
        pairs.append( (idx, idx, False) )


# estimating shift and rotation centre
verbose = int(args.verbose) * 2
if args.verbose :
    print("Estimating first shift:")
searchCrop = np.s_[searchAmplitude:-searchAmplitude, searchAmplitude:-searchAmplitude]
shifts = cs.findShift(dataO[directPair[0], *searchCrop], dataS[directPair[1], *searchCrop],
                      maskO[searchCrop], maskS[searchCrop], amplitude=searchAmplitude, verbose=int(args.verbose)*2)
shiftDir = shifts[2][0]
#shiftDir = (-73, 23) # for debug only
if args.verbose :
    print(f"Shift estimation: {shiftDir}")
    print("Estimating rotation centre:")
searchCrop = np.s_[jitterAmplitude:-jitterAmplitude, searchAmplitude:-searchAmplitude]
shifts = cs.findShift(dataO[flipedPair[0], *searchCrop], np.flip(dataS[flipedPair[1], *searchCrop], axis=-1).copy(),
                      maskO[searchCrop], np.flip(maskS, axis=-1)[searchCrop].copy(),
                      amplitude=(jitterAmplitude,searchAmplitude), start=shiftDir, verbose=int(args.verbose)*2)
#shiftFlp =  tuple( shiftDir[dim] + round(np.median( [ sft[0][0][dim] for sft in shifts] )) for dim in (-2,-1) )
shiftFlp =  tuple( shiftDir[dim] + shifts[2][0][dim] for dim in (-2,-1) )
#shiftFlp = (-71, -32) # for debug only
rotCent = round( ( shiftDir[-1] + shiftFlp[-1] ) / 2 )
if args.verbose :
    print(f"Rotation cenre estimation: {rotCent} from shift {shiftFlp}.")

try :
    outPut.write(f"# Face {face}\n")
    outPut.write(f"# Shift direct {shiftDir}\n")
    outPut.write(f"# Shift flip   {shiftFlp}\n")
    outPut.write(f"# Rotation centre {rotCent}\n")
    outPut.write(f"# Columns:\n")
    outPut.write(f"# Index_org  Index_sft Shift_Y  Shift_X  Flip\n")
    outPut.flush()
except :
    raise Exception(f"Failed to open log file {args.output} for writing.", file=sys.stderr)

results = []
blocks = []
prevShift = shiftDir if pairs[0][2] else shiftFlp
curBlock = pairs[0][2]
blockStarted = 0

# find shifts
if args.verbose :
    print(f"Finding cross-pair shifts:")
pbar = tqdm.tqdm(total=len(pairs)) if verbose else None
for idx, pair in enumerate(pairs) :

    indeces = pair[0:2]
    flip = pair[2]
    if flip != curBlock :
        blocks.append((blockStarted, idx, curBlock))
        blockStarted = idx
        curBlock = flip

    # find relative shift
    inO = torch.tensor( dataO[indeces[0]], device=cs.device )
    inS = torch.tensor( np.flip(dataS[indeces[1]], axis=-1).copy() if flip else dataS[indeces[1]], device=cs.device )
    mskO = torch.tensor(maskO, device=cs.device)
    mskS = torch.tensor( np.flip(maskS, axis=-1).copy() if flip else maskS, device=cs.device )
    if indeces == directPair :
        prevShift = shiftDir
    elif indeces == flipedPair :
        prevShift = shiftFlp
    searchCrop = tuple( np.s_[ jitterAmplitude + abs(prevShift[dim]) : -jitterAmplitude - abs(prevShift[dim]) ] \
                        for dim in (0,1)  )
    corrs = cs.findShift(inO[searchCrop], inS[searchCrop], mskO[searchCrop], mskS[searchCrop],
                        amplitude=2, start=prevShift, verbose=False )
    #corr =  tuple( round(np.median( [ cr[0][0][dim] for cr in corrs] )) for dim in (-2,-1) )
    corr = corrs[2][0]
    shift =  tuple( prevShift[dim] + corr[dim] for dim in (0,1) )
    if max(  tuple(abs(shift[dim]-prevShift[dim]) for dim in (0,1) )  ) > 2:
         print(f"Warning! in projection {idx} {indeces}: shift {shift} is more than 2 pixels"
               f" away from previous {prevShift}. Will ignore it as a mistake.", file=sys.stderr)
         shift = prevShift
    else :
        prevShift = shift # tuple( int( shift[dim] / abs(shift[dim]) ) if shift[dim] else 0  for dim in (0,1) )
    results.append( [ *indeces, *shift, flip ] )
    if pbar is not None :
        pbar.update()
blocks.append((blockStarted, idx+1, curBlock))
if pbar is not None :
    del pbar

# clean results
results = np.array(results)
for block in blocks :
    if block[1] - block[0] < 5 :
        continue
    def idxVal(inp, pos) :
        return inp[0] if len(set(np.delete(inp, pos))) == 1 else inp[pos]
    for dim in (2,3) :
        results[block[0]  , dim] = idxVal( results[block[0]:block[0]+3, dim],  0)
        results[block[0]+1, dim] = idxVal( results[block[0]:block[0]+4, dim],  1)
        results[block[1]-2, dim] = idxVal( results[block[1]-4:block[1], dim], -2)
        results[block[1]-1, dim] = idxVal( results[block[1]-3:block[1], dim], -1)
        for idx in range(block[0]+2, block[1]-2) :
            results[idx, dim] = idxVal(results[idx-2:idx+3, dim], 2)

# write output
for idx in range(len(results)) :
    outPut.write(' '.join(map(str, results[idx,:])) + '\n')
outPut.close()

if args.verbose :
    print("Done!")

exit(0)


