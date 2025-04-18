
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as fn
import h5py
import tifffile
import matplotlib.pyplot as plt
import ssim
import tqdm
import math

device = torch.device('cuda:0')



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
    if preread :
        dataN = np.empty(data.shape, dtype=np.float32)
        if verbose :
            print("Reading input ... ", end="", flush=True)
        data.read_direct(dataN)
        if verbose :
            print("Done.")
        data = dataN
    return data


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
    if type(dataY) is not tuple and type(dataY) is not list:
        raise Exception(f"Unknown data type to plot: {type(dataY)}.")
    if type(dataYR) is not tuple and dataYR is not None:
        raise Exception(f"Unknown data type to plot: {type(dataYR)}.")

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
        raise Exception(f"Bad data type on plotData input rangeP: {type(rangeP)}")
    rangeP = np.s_[ max(0, rangeP[0]) : min(last, rangeP[1]) ]
    if dataX is None :
        dataX = np.arange(rangeP.start, rangeP.stop)

    plt.style.use('default')
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
    colors = [ plt.colors.hsv_to_rgb((hv/nofPlots, 1, 1)) for hv in range(nofPlots) ]
    for idx , data in enumerate(dataY):
        ax1.plot(dataX, data[rangeP], linestyle='-',  color=colors[idx])

    if dataYR is not None : # right Y axis
        ax2 = ax1.twinx()
        ax2.yaxis.grid(True, 'both', linestyle='dotted')
        nofPlots = len(dataYR)
        if rangeYR is not None:
            ax2.set_ylim(rangeYR)
        colors = [ plt.colors.hsv_to_rgb((hv/nofPlots, 1, 1)) for hv in range(nofPlots) ]
        for idx , data in enumerate(dataYR):
            ax2.plot(dataX, data[rangeP], linestyle='dashed',  color=colors[idx])

    if saveTo:
        fig.savefig(saveTo)
    if not show:
        plt.close(fig)
    else :
        plt.show()


def plotImage(image) :
    plt.imshow(image, cmap='gray')
    plt.axis("off")
    plt.show()


def plotImages(images) :
    for i, img in enumerate(images) :
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.axis("off")
    plt.show()


def unsqeeze4dim(tens):
    orgDims = tens.dim()
    if tens.dim() == 2 :
        tens = tens.unsqueeze(0)
    if tens.dim() == 3 :
        tens = tens.unsqueeze(1)
    return tens, orgDims


def squeezeOrg(tens, orgDims):
    if orgDims == tens.dim():
        return tens
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


def stretchImage(image) :
    mn = image.min()
    mx = image.max()
    if mn < mx :
        image = (image - mn) / (mx - mn)
    else :
        image = mn
    return image


SSIM_loss = ssim.SSIM(data_range=2.0, size_average=False, channel=1)


def findShift(inF, inS, maskF=None, maskS=None, amplitude=0, start=(0,0), verbose=False) :

    dims = len(inF.shape)
    if dims < 2 :
        raise Exception(f"Input must have 2 or 3 dimensions. Got {dims}.")
    face = inF.shape[-2:]
    if inF.shape != inS.shape :
        raise Exception(f"Input tensors must be of the same shape. Got: {inF.shape} and {inS.shape}.")
    inF = torch.tensor(inF, device=device, requires_grad=False)
    inS = torch.tensor(inS, device=device, requires_grad=False)
    if dims == 2 :
        inF = inF[None,:,:]
        inS = inS[None,:,:]
    nofSl = inF.shape[0]
    if maskF is not None :
        maskF = torch.tensor(maskF, device=device, requires_grad=False)
        if maskF.dim == 2:
            maskF = maskF[None,:,:]
        inF *= maskF
    if maskS is not None :
        maskS = torch.tensor(maskS, device=device, requires_grad=False)
        if maskS.dim == 2:
            maskS = maskS[None,:,:]
        inS *= maskS

    def individualNorm(inData, msk=None) :
        #im_mean = inData.sum(dim=(-1,-2)) / \
        #    (math.prod(face) if msk is None else msk.sum(dim=(-1,-2)))
        im_mean = inData.mean(dim=(-1,-2)) if msk is None else inData[:,(msk>0)].mean(-1)
        im_std = inData.std(dim=(-1,-2)) if msk is None else inData[:,(msk>0)].std(-1)
        return (1 if msk is None else msk) * (inData - im_mean) /\
            ( 1 if im_std == 0 else im_std )
    inFn = individualNorm(inF, maskF)
    inSn = individualNorm(inS, maskS)
    #convolution

    results = torch.empty( (3, nofSl, amplitude*2+1, amplitude*2+1 ),
                           dtype=torch.float32, device=device)
    if verbose > 1 :
        pbar = tqdm.tqdm(total=(2*amplitude+1)**2)
    elif verbose == 1:
        pbar = tqdm.tqdm(total=2*amplitude+1)
    else :
        pbar = None

    for shiftX in range(-amplitude, amplitude+1) :
        for shiftY in range(-amplitude, amplitude+1) :

            pos = (shiftY+amplitude, shiftX+amplitude)
            subF = np.s_[ max(0, start[0]+shiftY) : face[-2] + min(start[0]+shiftY, 0),
                          max(0, start[1]+shiftX) : face[-1] + min(start[1]+shiftX, 0)]
            subS = np.s_[ max(0, start[0]-shiftY) : face[-2] + min(start[0]-shiftY, 0),
                          max(0, start[1]-shiftX) : face[-1] + min(start[1]-shiftX, 0)]

            if maskF is None and maskS is None :
                convNorm =  math.prod(inFn[0,*subF].shape) * torch.ones((nofSl,1), device=device)
            elif maskF is None :
                convNorm = maskS[...,*subS].sum(dim=(-1,-2))
            elif maskS is None :
                convNorm = maskF[...,*subF].sum(dim=(-1,-2))
            else :
                convNorm = (maskF[...,*subF] * maskS[...,*subS]).sum(dim=(-1,-2))
            convNorm = torch.where(convNorm > 0, 1/convNorm, 0)

            results[0,:,*pos] = convNorm * \
                (inFn[:,*subF] * inSn[:,*subS]).sum(dim=(-1,-2))

            results[1,:,*pos] = - convNorm * fn.mse_loss( # negate to search for max
                inFn[:,*subF], inSn[:,*subS], reduction='none').sum(dim=(-1,-2))

            results[2,:,*pos] = SSIM_loss( torch.clamp(inFn[:,None,*subF]+1, 0, 2)*maskF[...,*subF] ,
                                           torch.clamp(inSn[:,None,*subS]+1, 0, 2)*maskS[...,*subS] )

            if verbose > 1:
                pbar.update(1)
        if verbose == 1:
            pbar.update(1)

    def shift(conved) :
        flatres = conved.view(nofSl, -1)
        indeces = torch.argmax(flatres, dim=-1)
        sfts = []
        vals = []
        for i in range(nofSl) :
            pos = divmod(indeces[i].item(), 2*amplitude + 1)
            sfts.append( (pos[0]-amplitude, pos[1]-amplitude) )
            vals.append( conved[i, *pos].item() )
        return sfts, vals
    #def shift(conved) :
    #    toRet = []
    #    for i in range(nofSl) :
    #        inarr = conved[i,...].clone()
    #        inarr *= inarr
    #        pos = sp.center_of_mass(inarr.cpu().numpy())
    #        toRet.append( (float(pos[0])-amplitude, float(pos[1])-amplitude) )
    #    return toRet
    toRet = []
    for meth in range(results.shape[0]) :
        sfts, vals = shift(results[meth,...])
        toRet.append((sfts, vals))
    return toRet
