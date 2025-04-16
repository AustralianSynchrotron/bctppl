#!/usr/bin/env python3

import os, sys
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import argparse
import numpy as np

import commonsource as cs

myPath = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description=
    'Fill sinograms with sinogap NN.')
parser.add_argument('original', type=str, default="",
                    help='Input stack of frames in original position.')
parser.add_argument('shifted', type=str, nargs='?', default="",
                    help='Input stack of frames in shifted position.')
parser.add_argument('-s', '--stitch', default="", type=str,
                    help='Output of stitch.sh script run with no arguments.'
                         ' Reads from stdin if omited')
parser.add_argument('-m', '--mask', default=[], action='append',
                    help='Mask in the original and (if given) shifted position.')
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
try:
    while True:
        strg = input()
        if directPair is None and "-g" in strg:
            directPair = tuple([int(el) for el in strg.split()[0:2]])
        if flipedPair is None and "-f" in strg:
            flipedPair = tuple([int(el) for el in strg.split()[0:2]])
        if directPair is not None and flipedPair is not None:
            break
except EOFError:
    pass
if directPair is None or flipedPair is None:
    raise Exception("Failed to find both direct and flipped pairs in the input stream.")

verbose = int(args.verbose) * 2
shift = cs.findShift(dataO[directPair[0]],
                     dataS[directPair[1]],
                     maskO, maskS,
                     amplitude=100, verbose=verbose)
print(shift)
shiftCent = cs.findShift(dataO[flipedPair[0]],
                         np.flip(dataS[flipedPair[1]], axis=-1).copy(),
                         maskO, np.flip(maskS, axis=-1).copy(),
                         amplitude=100, verbose=verbose)
print(shiftCent)
