#!/usr/bin/env python3

import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave
import numpy as np
import cv2
import h5py
import tifffile
import tqdm
import sys
import time
import IPython
import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
import math
import gc
import argparse
import tqdm
from scipy.optimize import curve_fit
from multiprocessing import Pool
import commonsource as cs

myPath = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description=
    'Tracks the ball in the BCT experiment.')
parser.add_argument('images', type=str, default="",
                    help='Input stack to track the ball in.')
parser.add_argument('-m', '--mask', type=str, default="",
                    help='Mask of the input stack.')
parser.add_argument('-o', '--out', type=str, default="",
                    help='Output results. Stdout by default.')
parser.add_argument('-J', '--only', action='store_true', default=False,
                    help='Will analyze only Y coordinate with X assumed perfect.')
parser.add_argument('-v', '--verbose', action='store_true', default=False,
                    help='Be verbose and save plot results.')
args = parser.parse_args()

device = torch.device('cuda:0')
try:
    localCfgDict = dict()
    exec(open(os.path.join(myPath, ".local.cfg")).read(),localCfgDict)
    device = torch.device(localCfgDict['torchdevice'])
except KeyError:
    raise
except:
    pass



if args.verbose :
    print("Reading input ...", end="", flush=True)

kernelImage = cs.loadImage(os.path.dirname(__file__) + "/ball.tif")
ksh = kernelImage.shape
kernel = torch.tensor(kernelImage, device=device).unsqueeze(0).unsqueeze(0)
st, mn = torch.std_mean(kernel)
kernel = ( kernel - mn ) / st
kernelBin = torch.where(kernel>0, 0, 1).to(torch.float32).to(device)

data = cs.getInData(args.images, args.verbose, preread=True)
dsh = data.shape[1:]
nofF = data.shape[0]

maskImage = cs.loadImage(args.mask, dsh) if args.mask else np.ones(dsh)
maskImage /= maskImage.max()
maskPad = torch.zeros( (1, 1, dsh[-2] + 2*ksh[-2] - 2, dsh[-1] + 2*ksh[-1] - 2 ) )
maskPad[..., ksh[-2]-1 : -ksh[-2]+1, ksh[-1]-1 : -ksh[-1]+1 ] = torch.from_numpy(maskImage).unsqueeze(0).unsqueeze(0)
maskPad = maskPad.to(device)
maskCount = fn.conv2d(maskPad, torch.ones_like(kernel, device=device))
maskCount = torch.where(maskCount>0, 1/maskCount, 0)
maskBall = fn.conv2d(maskPad, kernelBin)
minArea = math.prod(ksh) // 56
maskCorr = torch.where( maskBall > minArea, 1, 0).squeeze().cpu().numpy()

if args.verbose :
    print(" Read.")
    print("Tracking the ball.")


def selectVisually() :


    def getFrame(frame) :
        return np.where(maskImage == 1, data[frame,...], 0 )

    currentIdx = 0
    currentFrame = None
    roi = None
    currentPos = (0,0)
    thresholds = [0,100]
    clip = [0,0]
    currentMatch = None

    def onMouse(event, x, y, flags, *userdata) :
        global roi, currentPos
        if currentFrame is None:
            return
        currentPos = (x,y)
        if  event == cv2.EVENT_RBUTTONDOWN:
            roi = (x,y, None, None)
            updateFrame()
        elif event == cv2.EVENT_RBUTTONUP and not roi is None:
            if currentPos == (roi[0], roi[1]) :
                roi = None
            elif roi[2] is None:
                x = 0 if x < 0 else currentFrame.shape[1]-1 if x >= currentFrame.shape[1] else x
                y = 0 if y < 0 else currentFrame.shape[0]-1 if y >= currentFrame.shape[0] else y
                roi = ( min(roi[0],x), min(roi[1],y), abs(roi[0]-x), abs(roi[1]-y) )
            updateFrame()
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_RBUTTON :
            updateFrame()


    def updateFrame(index = None) :
        global currentIdx, currentFrame
        if not index is None:
            currentIdx = index
            currentFrame = getFrame(index)
            cv2.setTrackbarPos(frameTrackbarName, windowName, index)
        global clip
        if currentFrame is None :
            return
        minV = currentFrame.min()
        maxV = currentFrame.max()
        # here I use second to the max value to avoid special values in some detectors
        maxV =  np.where(currentFrame == maxV, 0, currentFrame ).max()
        delta = maxV - minV
        clip[0] = minV + delta * thresholds[0] / 100
        clip[1] = maxV - delta * (1 - thresholds[1] / 100)
        if (clip[1] - clip[0]) < delta / 100 :
            shImage = np.where(currentFrame < clip[0], 0.0, 1.0)
            clip[1] = clip[0] + delta / 100
        else :
            shImage = ( np.clip(currentFrame, a_min=clip[0], a_max=clip[1]) - clip[0] ) / \
                        ( clip[1]-clip[0] if clip[1] != clip[0] else 1.0)
        shImage = np.repeat( np.expand_dims(shImage, 2), 3, axis=2 )
        if not roi is None:
            plotRoi = roi
            if roi[2] is None :
                plotRoi = (min(roi[0],currentPos[0]), min(roi[1],currentPos[1]),
                           abs(roi[0]-currentPos[0]), abs(roi[1]-currentPos[1]) )
            cv2.rectangle(shImage, plotRoi, color=(0,0,255), thickness=2)
        if not currentMatch is None:
            cv2.rectangle(shImage, currentMatch, color=(0,255,255), thickness=2)
        cv2.imshow(windowName, shImage)
        return True

    def showImage(*args):
        global currentFrame, currentIdx, currentMatch
        currentMatch = None
        currentIdx = args[0]
        currentFrame = getFrame(currentIdx)
        updateFrame()

    def updateThresholds():
        updateFrame()
        cv2.setTrackbarPos(loThresholdTrackbarName, windowName, thresholds[0])
        cv2.setTrackbarPos(hiThresholdTrackbarName, windowName, thresholds[1])

    def onLoThreshold(*args):
        thresholds[0] = args[0]
        if thresholds[1] < thresholds[0] :
            thresholds[1] = thresholds[0]
        updateThresholds()

    def onHiThreshold(*args):
        thresholds[1] = args[0]
        if thresholds[1] < thresholds[0] :
            thresholds[0] = thresholds[1]
        updateThresholds()


    windowName = "tracker"
    cv2.namedWindow(windowName, cv2.WINDOW_GUI_NORMAL)
    cv2.setWindowProperty(windowName, cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback(windowName, onMouse)
    frameTrackbarName = "Frame"
    cv2.createTrackbar(frameTrackbarName, windowName, 0, nofF, showImage)
    loThresholdTrackbarName = "Lo threshold"
    cv2.createTrackbar(loThresholdTrackbarName, windowName, 0, 100, onLoThreshold)
    hiThresholdTrackbarName = "Hi threshold"
    cv2.createTrackbar(hiThresholdTrackbarName, windowName, 100, 100, onHiThreshold)
    onLoThreshold((0))
    onHiThreshold((100))

    def backToWindow() :
        showImage(currentIdx)
        while True :
            c = cv2.waitKey(0)
            if c == 27 : # Esc
                roi = None
                updateFrame()
            elif c == 32 : #space
                return True
            elif c == 225 or c == 233: #modifiers
                continue
            else :
                print(c)
                #cv2.destroyAllWindows()
                return False





def normalizeWithMask(ten, msk) :
    ten, odim = cs.unsqeeze4dim(ten)
    maskSum = torch.count_nonzero(msk)
    ten *= msk
    mn = ten.sum(dim=(-2,-1)).view(-1,1,1,1) / maskSum
    ten -= mn
    ten *= msk
    st = ten.norm(p=2, dim=(-2,-1)).view(-1,1,1,1) / maskSum
    ten /= st
    return cs.squeezeOrg(ten,odim)


def removeBorders(img, mask) :

  sh = img.shape
  # BCT ONLY: the ball is never in the upper part of the image:
  #mask[:sh[0]//4,:] = 0

  def cleaLine(ln, ms, pos) :
    str = 0 if pos else -1
    mul = 1 if pos else -1
    idx = str
    prev = ln[idx]+1
    lnl = ln.shape[0]
    upcounter = 0
    while abs(idx) < lnl+abs(str)-1 :
      if ms[idx] == 0.0 :
        prev = ln[idx+mul]
      elif ln[idx] > prev :
        upcounter += 1
        if upcounter > 1 :
          break
        else :
          ms[idx] = 0
      else :
        upcounter = 0
        ms[idx] = 0
        prev = ln[idx]
      idx += mul

  for idy in range(sh[0]) :
    ln = img[idy,...]
    ms = mask[idy,...]
    cleaLine(ln, ms, True)
    cleaLine(ln, ms, False)
  for idx in range(sh[1]) :
    ln = img[...,idx]
    ms = mask[...,idx]
    cleaLine(ln, ms, True)
    cleaLine(ln, ms, False)
  # apply BCT mask: the ball should never be there
  mask[:2*ksh[0],:] = 0
  mask[:,:2*ksh[1]] = 0
  mask[:,-2*ksh[1]:] = 0


# I can move removeNorders to GPU and use it instead of Pool multiprocessing.
# But on our 50-100 cores it will save max 5 minutes per dataset - not worth the efforts.
def getPosInPool(img) :
    global maskCorr
    borderMask = maskCorr.copy()
    removeBorders(img, borderMask)
    img *= borderMask
    return np.array(np.unravel_index(np.argmax(img), img.shape))


def trackIt() :
    global maskPad, maskCount

    torch.no_grad()

    #results=torch.empty( (0,2), device=device )
    results = np.empty((0,2))
    btPerIm = 4 * ( math.prod(maskPad.shape) + math.prod(maskImage.shape) )
    startIndex=0
    if args.verbose :
        pbar = tqdm.tqdm(total=nofF)
    while True :
        gc.collect()
        torch.cuda.empty_cache()
        maxNofF = int ( 0.5   * torch.cuda.mem_get_info(device)[0] / btPerIm ) # 0.5 for contingency
        #maxNofF = 10
        stopIndex=min(startIndex+maxNofF, nofF)
        fRange = np.s_[startIndex:stopIndex]
        nofR = stopIndex-startIndex
        dataPad = torch.zeros( (nofR, 1, dsh[-2] + 2*ksh[-2] - 2, dsh[-1] + 2*ksh[-1] - 2 ) )
        dataPad[ ... , ksh[-2]-1 : -ksh[-2]+1, ksh[-1]-1 : -ksh[-1]+1 ] = \
            torch.from_numpy(data[fRange,...]).unsqueeze(1)
        dataPad = dataPad.to(device)
        dataPad = normalizeWithMask(dataPad, maskPad)
        dataCorr = fn.conv2d(dataPad, kernel) * maskCount
        psh = dataCorr.shape
        dataNP = dataCorr.cpu().numpy()
        dataInPool = [ dataNP[cursl,0,...] for cursl in range(nofR) ]
        with Pool() as p: # CPU load
            resultsR = np.array(p.map(getPosInPool, dataInPool))
            results = np.concatenate((results,resultsR),axis=0)
        del dataPad
        del dataCorr

        if args.verbose :
            pbar.update(nofR)
        startIndex = stopIndex
        if stopIndex >= nofF:
            break

    results = results - ksh + 1 # to correct for padding
    return results


def trackItFine(poses) :
    # area around expected position +/- 5 pixels and ksh on all sides
    neib=5
    dataBuf = torch.zeros( (dsh[0], ksh[0]+2*neib, ksh[1]+2*neib) )
    maskBuf = torch.zeros( (dsh[0], ksh[0]+2*neib, ksh[1]+2*neib) )
    for cursl in range(dsh[0]) :
        pos = poses[cursl,...]
        imFrom = ( max(0, pos[0] - neib ) , max(0, pos[1] - neib  ) )
        imTo = ( min(dsh[0], pos[0] + neib + ksh[0]) , min(dsh[1], pos[1] + neib + ksh[1]) )
        arSz = ( imTo[0] - imFrom[0], imTo[1] - imFrom[1])
        dstFrom = (imFrom[0] - pos[0] + neib , imFrom[1] - pos[1] + neib  )
        dataBuf[cursl, dstFrom[0] : dstFrom[0]+arSz[0] , dstFrom[1] : dstFrom[1]+arSz[1] ] = \
            torch.from_numpy(data[cursl , imFrom[0]:imTo[0], imFrom[1]:imTo[1] ])
        maskBuf[cursl, dstFrom[0] : dstFrom[0]+arSz[0] , dstFrom[1] : dstFrom[1]+arSz[1] ] = \
            torch.from_numpy(maskImage)[imFrom[0]:imTo[0], imFrom[1]:imTo[1] ]
        dataBuf[cursl,...] = normalizeWithMask( dataBuf[cursl,...], maskBuf[cursl,...] )
    dataBuf = dataBuf.unsqueeze(1).to(device)
    dataBuf = fn.conv2d(dataBuf, kernel).squeeze()
    maskBuf = maskBuf.unsqueeze(1).to(device)
    maskBuf = fn.conv2d(maskBuf, torch.ones_like(kernel, device=device)).squeeze()
    dataBuf /= maskBuf
    bufSh = ( dataBuf.shape[-2], dataBuf.shape[-1] )
    for cursl in range(dsh[0]) :
        pos = poses[cursl,...]
        cpos = np.unravel_index(torch.argmax(dataBuf[cursl,...]).item(), bufSh)
        pos[0] +=  cpos[0] - neib
        pos[1] +=  cpos[1] - neib

    return poses




def analyzeResults(analyzeme) :

    def fit_as_sin(dat, xdat, xxdat=None) :

        def sin_func(x, a, b, c, d):
            return a + b * np.sin(c*x+d)

        delta = dat.max() - dat.min()
        if delta == 0 :
            return dat
        xsize = xdat[-1]-xdat[0]
        x_norm = xdat / xsize
        meanDat = dat.mean()
        dat_norm = (dat - meanDat) / delta # normalize for fitting
        #popt, _ = curve_fit(sin_func, x_norm, dat_norm)

        popt, _ = curve_fit(sin_func, x_norm, dat_norm,
                            #p0 = [0, 0, math.pi, 0],
                            bounds=([-1 , 0, 0,         0],
                                    [ 1 , 1, 2*math.pi, 2*math.pi]))
        dat_fit = meanDat + delta * sin_func(x_norm if xxdat is None else xxdat / xsize , *popt)
        popt[0] = popt[0] * delta + meanDat
        popt[1] *= delta

        return dat_fit, popt

    # first stage of cleaning: based on Y position which should not change more than 3 pixels between frames
    def firstStageClean(rawRes) :
        toRet = np.empty((0,3))
        for curF in range(1,rawRes.shape[0]-1) :
            if  abs(rawRes[curF,0]-rawRes[curF-1,0]) <= 3 \
            and abs(rawRes[curF,0]-rawRes[curF+1,0]) <= 3 :
                toRet = np.concatenate((toRet,rawRes[[curF],:]),axis=0)
        if  abs(rawRes[0,0]-toRet[0,0]) <= 2 :
            toRet = np.concatenate((rawRes[[0],:],toRet),axis=0)
        if  abs(rawRes[-1,0]-toRet[-1,0]) <= 2 :
            toRet = np.concatenate((toRet,rawRes[[-1],:]),axis=0)
        return toRet

    cleanResults = firstStageClean(analyzeme)

    # second stage of cleaning: based on Y position which should not change more than 6 pixels away from median
    def secondStageClean(rawRes) :
        med = np.median(rawRes[:,0])
        toRet = np.empty((0,3))
        for curF in range(rawRes.shape[0]) :
            if  abs(rawRes[curF,0]-med) <= 6 :
                toRet = np.concatenate((toRet,rawRes[[curF],:]),axis=0)
        return toRet

    cleanResults = secondStageClean(cleanResults)

    # third stage of cleaning: based on both X and Y tracks,
    # which should not be more than 3 pixels away from fitted curves
    def thirdStageClean(rawRes, fit) :
        toRet = np.empty((0,3))
        for curF in range(rawRes.shape[0]) :
            if  abs( fit[curF,0] - rawRes[curF,0] ) <= 3 \
            and abs( fit[curF,1] - rawRes[curF,1] ) <= 3 :
                toRet = np.concatenate((toRet,rawRes[[curF],:]),axis=0)
        return toRet

    #res_fit0, _ = fit_as_sin(cleanResults[:,0], cleanResults[:,-1], frameNumbers)
    res_fit0 = np.full(frameNumbers.shape, np.median(cleanResults[:,0]))
    res_fit1, _ = fit_as_sin(cleanResults[:,1], cleanResults[:,-1], frameNumbers)
    res_fit = np.concatenate((res_fit0, res_fit1), axis=1)
    cleanResults = thirdStageClean(analyzeme, res_fit)


    # fill the gaps in data cleaned earlier
    def fillCleaned(rawRes, frameNumbers) :
        inter0 = np.interp(frameNumbers, rawRes[:,-1], rawRes[:,0])
        inter1 = np.interp(frameNumbers, rawRes[:,-1], rawRes[:,1])
        filled = np.concatenate((inter0, inter1, frameNumbers), axis=1)
        return filled

    posResults = fillCleaned(cleanResults, frameNumbers)
    #plotData( finalResults[:,0], dataYR=finalResults[:,1] , dataX=finalResults[:,-1])

    # make shifts from positions
    shiftResults = posResults.copy()
    shiftResults[:,0] = shiftResults[:,0] - res_fit0[:,0]
    shiftResults[:,1] = shiftResults[:,1] - res_fit1[:,0]
    #shiftResults[:,1] -= np.round(np.mean(shiftResults[:,1]))

    # remove peaks
    def shiftsClean(rawRes) :
        while True :
            interim = rawRes.copy()
            somethingChanged = False
            for curF in range(1,rawRes.shape[0]-1) :
                avNeib = 0.5 * (rawRes[curF-1] + rawRes[curF+1])
                if abs( rawRes[curF-1] - rawRes[curF+1] ) < 1 \
                and abs ( avNeib - rawRes[curF] ) >= 1 :
                    interim[curF] = avNeib
                    somethingChanged = True
            if somethingChanged :
                rawRes[()] = interim
            else:
                break
        # deal with ends
        delta = rawRes[2] - rawRes[1]
        if abs(delta) < 1 :
            rawRes[0] = rawRes[1] - delta
        delta = rawRes[-3] - rawRes[-2]
        if abs(delta) < 1 :
            rawRes[-1] = rawRes[-2] - delta
        return rawRes

    shiftsClean(shiftResults[:,0])
    #shiftResults[:,0] -= np.round( (shiftResults[:,0].min() + shiftResults[:,0].max()) / 2 )
    shiftsClean(shiftResults[:,1])
    #shiftResults[:,1] -= np.round( (shiftResults[:,1].min() + shiftResults[:,1].max()) / 2 )

    allResults = np.concatenate((shiftResults[:,:2], posResults[:,:2]), axis=1)

    return allResults


trackResults = trackIt()
frameNumbers = np.expand_dims( np.linspace(0, nofF-1, nofF), 1)
results = np.concatenate((trackResults, frameNumbers), axis=1)
#np.savetxt(".rawtracking.dat", trackResults, fmt='%i')
results = analyzeResults(results)
results = trackItFine( np.round(results[:,2:4]).astype(int) )
results = np.concatenate((results, frameNumbers), axis=1)
results = np.round(analyzeResults(results)).astype(int)
if args.only :
    results[:,3] -= results[:,1]
    results[:,1] = 0

np.savetxt(args.out if args.out else sys.stdout.buffer, results, fmt='%i')
if args.verbose :
    plotName = os.path.splitext(args.out)[0] + "_plot.png"
    cs.plotData( (results[:,0], results[:,1]),
              dataYR=(results[:,2], results[:,3]),
              saveTo = plotName, show = False)





