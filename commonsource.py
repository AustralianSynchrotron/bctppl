
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as fn
import h5py
from h5py import h5d
import tifffile
import matplotlib
import matplotlib.pyplot as plt
import ssim
import tqdm
import math
import os
import mmap

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


def residesInMemory(hdfName) :
    mmapPrefixes = ["/dev/shm",]
    if "CTAS_MMAP_PATH" in os.environ :
        mmapPrefixes.append[os.environ["CTAS_MMAP_PATH"].split(':')]
    hdfName = os.path.realpath(hdfName)
    for mmapPrefix in mmapPrefixes :
        if hdfName.startswith(mmapPrefix) :
            return True
    return False

def goodForMmap(trgH5F, data) :
    fileSize = trgH5F.id.get_filesize()
    offset = data.id.get_offset()
    plist = data.id.get_create_plist()
    if offset < 0 \
    or not plist.get_layout() in (h5d.CONTIGUOUS, h5d.COMPACT) \
    or plist.get_external_count() \
    or plist.get_nfilters() \
    or fileSize - offset < math.prod(data.shape) * data.dtype.itemsize :
        return None, None
    else :
        return offset, data.id.dtype


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
        trgH5F =  h5py.File(hdfName,'r', swmr=True)
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
    try : # try to mmap hdf5 if it is in memory
        if not residesInMemory(hdfName) :
            raise Exception()
        fileSize = trgH5F.id.get_filesize()
        offset = data.id.get_offset()
        dtype = data.id.dtype
        plist = data.id.get_create_plist()
        if offset < 0 \
        or not plist.get_layout() in (h5d.CONTIGUOUS, h5d.COMPACT) \
        or plist.get_external_count() \
        or plist.get_nfilters() \
        or fileSize - offset < math.prod(sh) * data.dtype.itemsize :
            raise Exception()
        # now all is ready
        dataN = np.memmap(hdfName, shape=sh, dtype=dtype, mode='r', offset=offset)
        data = dataN
        trgH5F.close()
        #plist = trgH5F.id.get_access_plist()
        #fileno = trgH5F.id.get_vfd_handle(plist)
        #dataM = mmap.mmap(fileno, fileSize, offset=offset, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
    except :
        if preread :
            dataN = np.empty(data.shape, dtype=np.float32)
            if verbose :
                print("Reading input ... ", end="", flush=True)
            data.read_direct(dataN)
            if verbose :
                print("Done.")
            data = dataN
            trgH5F.close()
    return data


def getOutData(outputString, shape, dtype='f') :
    if len(shape) == 2 :
        shape = (1,*shape)
    if len(shape) != 3 :
        raise Exception(f"Not appropriate output array size {shape}.")
    nameSplit = outputString.split(':')
    hdfName = nameSplit[0]
    hdfVolume = nameSplit[1]
    if len(nameSplit) != 2 :
        raise Exception(f"String \"{outputString}\" does not represent an HDF5 format \"fileName:container\".")
    try :
        trgH5F =  h5py.File(hdfName,'w', libver='latest')
    except :
        raise Exception(f"Failed to open HDF file '{hdfName}'.")
    trgH5F.require_dataset(hdfVolume, shape, dtype=dtype, exact=True)
    dset = trgH5F[hdfVolume]
#    if  hdfVolume in trgH5F.keys() and trgH5F[hdfVolume].shape == shape : # existing dataset
#    if residesInMemory(hdfName) :
#        #spaceid = h5py.h5s.create_simple((numRows, numCols))
#        plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
#        plist.set_fill_time(h5d.FILL_TIME_NEVER)
#        plist.set_alloc_time(h5d.ALLOC_TIME_EARLY)
#        plist.set_layout(h5d.CONTIGUOUS)
#        datasetid = h5py.h5d.create(fout.id, "rows", h5py.h5t.NATIVE_DOUBLE, spaceid, plist)
#    else :
    trgH5F.swmr_mode = True
    return dset, trgH5F


def ten2hdf(outputString, data) :
    nameSplit = outputString.split(':')
    hdfName = nameSplit[0]
    hdfVolume = nameSplit[1]
    if len(nameSplit) != 2 :
        raise Exception(f"String \"{outputString}\" does not represent an HDF5 format \"fileName:container\".")
    try :
        trgH5F =  h5py.File(hdfName,'w', libver='latest')
    except :
        raise Exception(f"Failed to open HDF file '{hdfName}'.")
    trgH5F.require_dataset(hdfVolume, data.shape, dtype=data.dtype, exact=True)
    trgH5F[hdfVolume][...] = data
    trgH5F.close()


class OutputWrapper:

    def __init__(self, outputString, shape):
        if len(shape) != 3 :
            raise Exception(f"Not appropriate output array size {shape}.")
        nameSplit = outputString.split(':')
        self.trgH5F = None
        if len(nameSplit) == 1 : # tiff image
            if shape[0] != 1 :
                raise Exception(f"Cannot save 3D data {shape} from input to a tiff file.")
            self.name = nameSplit[0]
            return
        if len(nameSplit) != 2 :
            raise Exception(f"String \"{outputString}\" does not represent an HDF5 format \"fileName:container\".")
        self.name = nameSplit[0]
        hdfVolume = nameSplit[1]
        try :
            self.trgH5F =  h5py.File(self.name,'w', libver='latest')
        except :
            raise Exception(f"Failed to open HDF file '{self.name}'.")
        if  hdfVolume not in self.trgH5F.keys():
            self.dset = self.trgH5F.create_dataset(hdfVolume, shape, dtype='f')
        else :
            self.dset = self.trgH5F[hdfVolume]
            csh = self.dset.shape
            if csh[0] < shape[0] or csh[1] != shape[1] or csh[2] != shape[2] :
                raise Exception(f"Shape mismatch: input {shape}, output HDF {csh}.")
        self.trgH5F.swmr_mode = True

    #def __del__(self):
    #    if self.trgH5F is not None :
    #        self.trgH5F.close()

    def put(self, data, slice, flush=True) :
        if len(data.shape) != 2 :
            raise Exception(f"Output accepts 2D data. Got shape {data.shape} instead.")
        if self.trgH5F is None : # tiff output
            tifffile.imwrite(self.name, data)
            if slice != 0 :
                Warning(f"Output is a tiff file, but non-zero slice {slice} is saved.")
        elif self.dset is not None : # hdf output
            if isinstance(slice, int) :
                slice = np.s_[slice,...]
            self.dset[slice] = data
            if flush :
                self.dset.flush()
        else :
            raise Exception(f"Output is not defined. Should never happen.")



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
    colors = [ matplotlib.colors.hsv_to_rgb((hv/nofPlots, 1, 1)) for hv in range(nofPlots) ]
    for idx , data in enumerate(dataY):
        ax1.plot(dataX, data[rangeP], linestyle='-',  color=colors[idx])

    if dataYR is not None : # right Y axis
        ax2 = ax1.twinx()
        ax2.yaxis.grid(True, 'both', linestyle='dotted')
        nofPlots = len(dataYR)
        if rangeYR is not None:
            ax2.set_ylim(rangeYR)
        colors = [ matplotlib.colors.hsv_to_rgb((hv/nofPlots, 1, 1)) for hv in range(nofPlots) ]
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




def findShift(inF, inS, maskF=None, maskS=None, amplitude=0, start=(0,0), verbose=False) :

    SSIM_loss = ssim.MS_SSIM(data_range=2.0, size_average=False, channel=1, win_size=11)

    dims = len(inF.shape)
    if dims < 2 :
        raise Exception(f"Input must have 2 or 3 dimensions. Got {dims}.")
    face = inF.shape[-2:]
    if inF.shape != inS.shape :
        raise Exception(f"Input tensors must be of the same shape. Got: {inF.shape} and {inS.shape}.")
    if not isinstance(inF, torch.Tensor) :
        inF = torch.tensor(inF, device=device, requires_grad=False)
    if not isinstance(inS, torch.Tensor) :
        inS = torch.tensor(inS, device=device, requires_grad=False)
    if dims == 2 :
        inF = inF[None,:,:]
        inS = inS[None,:,:]
    nofSl = inF.shape[0]
    if maskF is not None :
        if not isinstance(maskF, torch.Tensor) :
            maskF = torch.tensor(maskF, device=device, requires_grad=False)
        if maskF.dim == 2:
            maskF = maskF[None,:,:]
        inF *= maskF
    if maskS is not None :
        if not isinstance(maskS, torch.Tensor) :
            maskS = torch.tensor(maskS, device=device, requires_grad=False)
        if maskS.dim == 2:
            maskS = maskS[None,:,:]
        inS *= maskS
    if isinstance(amplitude, int) :
        amplitude = (amplitude, amplitude)

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

    results = torch.empty( (3, nofSl, amplitude[-2]*2+1, amplitude[-1]*2+1 ),
                           dtype=torch.float32, device=device)
    if verbose > 1 :
        pbar = tqdm.tqdm(total=(2*amplitude[-2]+1)*(2*amplitude[-1]+1))
    elif verbose == 1:
        pbar = tqdm.tqdm(total=2*amplitude[-1]+1)
    else :
        pbar = None

    for shiftX in range(-amplitude[-1], amplitude[-1]+1) :
        for shiftY in range(-amplitude[-2], amplitude[-2]+1) :

            pos = (shiftY+amplitude[-2], shiftX+amplitude[-1])
            subF = np.s_[ max(0,  start[0]+shiftY) : face[-2] + min( start[0]+shiftY, 0),
                          max(0,  start[1]+shiftX) : face[-1] + min( start[1]+shiftX, 0)]
            subS = np.s_[ max(0, -start[0]-shiftY) : face[-2] + min(-start[0]-shiftY, 0),
                          max(0, -start[1]-shiftX) : face[-1] + min(-start[1]-shiftX, 0)]

            if maskF is None and maskS is None :
                convNorm =  math.prod(inFn[0,*subF].shape) * torch.ones((nofSl,1), device=device)
            elif maskF is None :
                convNorm = maskS[...,*subS].sum(dim=(-1,-2))
            elif maskS is None :
                convNorm = maskF[...,*subF].sum(dim=(-1,-2))
            else :
                convNorm = (maskF[...,*subF] * maskS[...,*subS]).sum(dim=(-1,-2))
            convNorm = torch.where(convNorm > 0, 1/convNorm, 0)

            #results[0,:,*pos] = convNorm * \
            #    (inFn[:,*subF] * inSn[:,*subS]).sum(dim=(-1,-2))

            #results[1,:,*pos] = - convNorm * fn.mse_loss( # negate to search for max
            #    inFn[:,*subF], inSn[:,*subS], reduction='none').sum(dim=(-1,-2))

            commonMask = maskS[...,*subS]*maskF[...,*subF]
            results[2,:,*pos] = SSIM_loss(
                torch.clamp(inFn[:,None,*subF]+1, 0, 2)*commonMask ,
                torch.clamp(inSn[:,None,*subS]+1, 0, 2)*commonMask \
                    )#, mask = commonMask[None,None,...] > 0 )

            if verbose > 1:
                pbar.update(1)
        if verbose == 1:
            pbar.update(1)

    def shift(conved) :
        flatres = conved.view(nofSl, -1)
        indeces = torch.argmax(flatres, dim=-1)
        sfts = []
        #vals = []
        for i in range(nofSl) :
            pos = divmod(indeces[i].item(), 2*amplitude[-1] + 1)
            sfts.append( (pos[0]-amplitude[-2], pos[1]-amplitude[-1]) )
            #vals.append( conved[i, *pos].item() )
        return sfts#, vals
    toRet = []
    onlyRes = shift(results[2,...])
    for meth in range(results.shape[0]) :
        #toRet.append(shift(results[meth,...]))
        toRet.append(onlyRes)
    return toRet
