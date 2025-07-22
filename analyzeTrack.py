#!/usr/bin/env python3

import numpy as np
import argparse
import math
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser(description=
    'Analyzes results of ball tracking to find stich parameters.')
parser.add_argument('torg', type=str, default="",
                    help='Results of tracking in original position.')
parser.add_argument('tsft', type=str, default="", nargs='?',
                    help='Results of tracking in shifted position.')
parser.add_argument('-a', '--ark', type=int, default="",
                    help='Number of steps to cover 180deg ark.')
parser.add_argument('-s', '--sft', type=int, default=0,
                    help='Distance between first frames in shifted and original positions.'
                         ' Ignored for single input.')
parser.add_argument('-w', '--iwidth', type=int, default="",
                    help='Width of the image where the ball was tracked.')
parser.add_argument('-W', '--kwidth', type=int, default="",
                    help='Width of the the ball image.')
args = parser.parse_args()




def fit_as_sin(dat, xdat) :
    def sin_func(x, a, b, c, d):
        return a + b * np.sin(x * math.pi / (c-1) + d)
    delta = dat.max() - dat.min()
    if delta == 0 :
        return dat
    xsize = xdat[-1]-xdat[0]
    x_norm = xdat / xsize
    meanDat = dat.mean()
    dat_norm = (dat - meanDat) / delta # normalize for fitting
    popt, _ = curve_fit(sin_func, x_norm, dat_norm,
                        p0 = [0, 0.5, args.ark, 0],
                        bounds=([-1 , 0, args.ark / 1.1,         0],
                                [ 1 , 1, args.ark * 1.1, 2*math.pi]))
    popt[0] = popt[0] * delta + meanDat
    popt[1] *= delta
    return popt


resOrg = np.loadtxt( args.torg, dtype=int).astype(float)
vOrg = resOrg[:,2]
hOrg = resOrg[:,3]
hOrg += ( args.kwidth - 1 ) / 2
frameNofsOrg = np.linspace(0, resOrg.shape[0]-1, resOrg.shape[0])
poptOrg = fit_as_sin(hOrg, frameNofsOrg)

if not args.tsft :
    print(
       round( resOrg[:,1].max() - resOrg[:,1].min()),
       round( resOrg[:,0].max() - resOrg[:,0].min()),
       0,
       0,
       poptOrg[0] - (args.iwidth - 1) / 2,
       poptOrg[2] )
    exit(0)


resSft = np.loadtxt( args.tsft, dtype=int).astype(float)
vSft = resSft[:,2]
hSft = resSft[:,3]
hSft += ( args.kwidth - 1 ) / 2
frameNofsSft = np.linspace(args.sft, args.sft+resSft.shape[0]-1, resSft.shape[0])
f_pos = np.concatenate((frameNofsOrg, frameNofsSft))
poptSft = fit_as_sin(hSft, frameNofsSft)

sftFrame = ( args.ark + args.sft ) / 2
def h_func(x, cent, ampl, carc, phs, sft):
    shift = [ ( 0 if xc < sftFrame else sft ) for xc in x ]
    return cent + shift + ampl * np.sin( x * math.pi / (carc-1) + phs)
h_pos = np.concatenate((hOrg, hSft))
cent0 = poptOrg[0]
sft0 = poptSft[0]-poptOrg[0]
h_popt, _ = curve_fit(h_func, f_pos, h_pos,
                      p0     =  [cent0,
                                 ( poptSft[1] + poptOrg[1] ) / 2,
                                 args.ark,
                                 ( poptSft[3] + poptOrg[3] ) / 2,
                                 sft0
                                ],
                      bounds = ([cent0 / 1.1,
                                 min(poptSft[1], poptOrg[1]) / 1.1,
                                 args.ark / 1.1,
                                 0,
                                 sft0 - abs(sft0) * 0.1
                                ],
                                [cent0 * 1.1,
                                 max(poptSft[1], poptOrg[1]) * 1.1,
                                 args.ark * 1.1,
                                 2*math.pi,
                                 sft0 + abs(sft0) * 0.1
                                ]
                               )
                     )
dat_fit = h_func( f_pos , *h_popt)
#print(h_popt)
#plotData( [dat_fit, h_pos ] , dataX=f_pos )

def v_func(x, posOrg, vsft):
    return [ ( posOrg if xc < sftFrame else posOrg + vsft ) for xc in x ]
v_pos = np.concatenate((resOrg[:,2], resSft[:,2]))
v_popt, _ = curve_fit(v_func, f_pos, v_pos,
                      p0 = [ resOrg[:,2].mean(), resSft[:,2].mean() - resOrg[:,2].mean() ] )
#print(v_popt)

print( round(max ( resOrg[:,1].max(), resSft[:,1].max() ) - min( resOrg[:,1].min(), resSft[:,1].min() )),
       round(max ( resOrg[:,0].max(), resSft[:,0].max() ) - min( resOrg[:,0].min(), resSft[:,0].min() )),
       -h_popt[4],
       -v_popt[1],
       h_popt[0] - (args.iwidth - 1) / 2,
       h_popt[2]
       )



