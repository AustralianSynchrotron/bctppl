#!/usr/bin/env python3

import numpy as np
import sys
import re
from scipy.optimize import curve_fit

if len(sys.argv) < 2 :
    print("Error! No input stream provided.", file=sys.stderr)
    exit(1)


data = []
badline=None
with open(sys.argv[1]) as file:
    for line in file:
        line = line.strip()
        if len(line) :
            lres = re.search('FrameNumber=\"([^\"]*)\".*Angle=\"([^\"]*)\"', line)
            if not lres:
                badline=line
            else :
                data.append([int(lres.group(1)), float(lres.group(2))])
if badline :
    print(f"Warning! Failed to parse line(s) from PPS stream; f.e.: \"{badline}\".")
data = np.array(data)

firstFrame = np.where( np.logical_and (data[:,0] != data[0,0], data[:,0]>0) )[0][0] - 1
lastFrame = np.where(data[:,0] == data[-1,0])[0][0] + 1
data = data[firstFrame:lastFrame,:]
points = data.shape[0]
uniqData = []
prFrame=-1
for idx in range(points) :
    frame = data[idx,0]
    if frame != prFrame :
        uniqData.append(idx)
        prFrame = frame
    else :
        pass
uniqData = np.array(uniqData)
data = data[uniqData,:]


def lin_func(x, a, b):
    return a + b * x
def lin_fit(xdat, ydat) :
    popt, pcov = curve_fit(lin_func, xdat, ydat)
    ftData = lin_func(xdat, *popt)
    dfData = ydat - ftData
    return dfData, popt, pcov, dfData.std()

while True :
    df, ppt, pcv, stdev = lin_fit( data[:,0], data[:,1] )
    if stdev < ppt[1] :
        break
    goodPoints = np.where( abs(df) < stdev )[0]
    if goodPoints.shape[0] == data.shape[0] :
        break
    data = data[goodPoints,:]
    if points > 10 * data.shape[0] :
        print("Error! Failed to estimate rotation ark: no convergence.", file=sys.stderr)
        exit(1)

if ppt[1] < 0.001 :
    print(f"Error! Impossible estimated rotation step {ppt[1]} < 0.001.", file=sys.stderr)
    exit(1)

print(abs(int(180/ppt[1])))