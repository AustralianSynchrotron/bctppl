#import shiftpatch_module as sg
import torch
import torch.nn as nn
from torchvision import transforms
import math
import os
import pytorch_amfill


latentDim = 64
inShape = (80,80)
batchsize = 2**8
save_interim = False


def inverseElements(arr, zval = 0) :
    arr = torch.where(arr != 0, 1/arr, zval)
    return arr


def inverseElements_(arr, zval = 0) :
    arr[...] = torch.where(arr != 0, 1/arr, zval)
    return arr


def normAndFillImages(images) :
    fImages = pytorch_amfill.ops.amfill(images, images > 0)
    sval, mval = torch.std_mean(fImages, dim=(-1,-2), keepdim=True)
    return (fImages - mval) * inverseElements(sval) , sval, mval


def denorm(images, sval, mval) :
    return torch.addcmul(mval, images, sval)


class Generator(nn.Module):


    def __init__(self):
        super(Generator, self).__init__()
        self.latentChannels = 1
        self.inputChannels = 2
        self.baseChannels = 16
        self.amplitude = 1
        self.batchNormOpt = {}

        self.noise2latent = self.createLatent()

        self.encoders =  nn.ModuleList([
            self.encblock( (self.inputChannels+abs(self.latentChannels)) /self.baseChannels,
                               1, 3, norm=False,),
            self.encblock( 1,  1, 3, norm=True, dopadding=True),
            self.encblock( 1,  2, 3, norm=True, stride=2),
            self.encblock( 2,  2, 3, norm=True, dopadding=True),
            self.encblock( 2,  4, 3, norm=True, stride=2),
            self.encblock( 4,  4, 3, norm=True, dopadding=True),
            self.encblock( 4,  8, 3, norm=True, stride=2),
            self.encblock( 8,  8, 3, norm=True, dopadding=True),
            self.encblock( 8, 16, 3, norm=True, stride=2),
            self.encblock(16, 16, 3, norm=True, dopadding=True),
            ])

        self.fcLink = self.createFClink()

        self.decoders = nn.ModuleList([
            self.decblock(32, 16, 3, norm=True, dopadding=True),
            self.decblock(32,  8, 4, norm=True, stride=2),
            self.decblock(16,  8, 3, norm=True, dopadding=True),
            self.decblock(16,  4, 4, norm=True, stride=2),
            self.decblock( 8,  4, 3, norm=True, dopadding=True),
            self.decblock( 8,  2, 4, norm=True, stride=2),
            self.decblock( 4,  2, 3, norm=True, dopadding=True),
            self.decblock( 4,  1, 4, norm=True, stride=2),
            self.decblock( 2,  1, 3, norm=True, dopadding=True),
            self.decblock( 2,  1, 3, norm=False),
            ])

        self.lastTouch = self.createLastTouch()



    def preProc(self, input):
        images, noises = input
        with torch.no_grad() :

            orgDims = images.dim()
            if orgDims == 3 :
                images = images.view(1, *images.shape)
            masks = images[:,2:4,...]
            images = images[:,0:2,...] * masks

            presentInBoth = ( masks[:,[0],...] * masks[:,[1],...] > 0 )
            pImages = pytorch_amfill.ops.amfill(images, presentInBoth)
            rImages = inverseElements(pImages)
            procImages = torch.where( masks > 0 , images , images[:,[1,0],...] * pImages * rImages[:,[1,0],...] )
            procImages, sval, mval = normAndFillImages(procImages)

            noises = noises if noises is not None else None if not self.latentChannels else \
                torch.randn( (images.shape[0], latentDim) , device = images.device)

        return (procImages, noises), (sval, mval, orgDims)


    def postProc(self, images, cfg):
        sval, mval, orgDims = cfg
        pImages = denorm(images, sval, mval)
        if orgDims == 3 :
            pImages = pImages.view(1, *images.shape)
        return pImages




    def forward(self, input):
        global save_interim

        if not save_interim is None :
            save_interim = {}
        def saveToInterim(key, data) :
            if not save_interim is None :
                save_interim[key] = data.clone().detach()

        masks = input[0][:,2:4,...]
        input, procInf = self.preProc(input)
        images, noises = input
        saveToInterim('input', images)
        #return self.postProc(images[:,[1,0],...], procInf)

        dwTrain = [images,] if noises is None else [torch.cat((images, self.noise2latent(noises)), dim=1),]
        for encoder in self.encoders :
            dwTrain.append(encoder(dwTrain[-1]))
        mid = self.fcLink(dwTrain[-1])
        upTrain = [mid]
        for level, decoder in enumerate(self.decoders) :
            upTrain.append( decoder( torch.cat( (upTrain[-1], dwTrain[-1-level]), dim=1 ) ) )
        res = images[:,[1,0],...] + self.amplitude * self.lastTouch(torch.cat( (upTrain[-1], images ), dim=1 ))

        res = torch.where ( masks > 0 , images, self.postProc(res, procInf) )
        saveToInterim('output', res)
        return res


    def patchMe(self, input) :
        res = self.forward(input)
        images = input[0]
        masks = images[:,2:,...]
        #mn = torch.where( masks[:,0:2,...] > 0 , images[:,0:2,...], images.max() ).amin(dim=(2,3))
        res[:,0:2,...] = masks[:,0:2,...] * images[:,0:2,...] + \
                          ( 1-masks[:,0:2,...] ) * masks[:,[1,0],...] * res[:,0:2,...]
        return res



    def createLatent(self) :
        if self.latentChannels == 0 :
            return None
        realLatent = abs(self.latentChannels)
        toRet =  nn.Sequential(
            nn.Linear(latentDim, math.prod(inShape) * realLatent),
            nn.ReLU(),
            nn.Unflatten( 1, (realLatent,) + inShape )
        )
        return toRet


    def encblock(self, chIn, chOut, kernel, stride=1, norm=False, dopadding=False) :
        chIn = int(chIn*self.baseChannels)
        chOut = int(chOut*self.baseChannels)
        layers = []
        layers.append( nn.Conv2d(chIn, chOut, kernel, stride=stride, bias = not norm,
                                padding='same', padding_mode='reflect') \
                                if stride == 1 and dopadding else \
                                nn.Conv2d(chIn, chOut, kernel, stride=stride, bias= not norm )
                     )
        if norm :
            layers.append(nn.BatchNorm2d(chOut, **self.batchNormOpt))
        layers.append(nn.LeakyReLU(0.2))
        return torch.nn.Sequential(*layers)


    def decblock(self, chIn, chOut, kernel, stride=1, norm=False, dopadding=False) :
        chIn = int(chIn*self.baseChannels)
        chOut = int(chOut*self.baseChannels)
        layers = []
        layers.append( nn.ConvTranspose2d(chIn, chOut, kernel, stride, bias=not norm,
                                          padding=1) \
                       if stride == 1 and dopadding else \
                       nn.ConvTranspose2d(chIn, chOut, kernel, stride, bias = not norm)
                     )
        if norm :
            layers.append(nn.BatchNorm2d(chOut, **self.batchNormOpt))
        layers.append(nn.LeakyReLU(0.2))
        return torch.nn.Sequential(*layers)


    def createFClink(self) :
        smpl = torch.zeros((1, self.inputChannels + abs(self.latentChannels), *inShape))
        with torch.no_grad() :
            for encoder in self.encoders :
                smpl = encoder(smpl)
        encSh = smpl.shape
        linChannels = math.prod(encSh)
        toRet = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linChannels, linChannels),
            nn.LeakyReLU(0.2),
            nn.Linear(linChannels, linChannels),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, encSh[1:]),
        )
        return toRet


    def createLastTouch(self) :
        toRet = nn.Sequential(
            nn.Conv2d(self.baseChannels+self.inputChannels, 2, 1),
            nn.Tanh(),
        )
        return toRet




#sg.optimizer_G = sg.createOptimizer(sg.generator, sg.TCfg.learningRateG)
#model_summary = summary(sg.generator, input_data=[ [sg.refImages[[0],0:4,...], sg.refNoises[[0],...]] ] ).__str__()
#print(model_summary)

#_ = sg.testMe(sg.refImages)


dataMeanNorm = (0.5,0.5,0,0,0,0) # masks not to be normalized
imageFwdTransforms = transforms.Normalize(mean=(0.5,0.5,0,0), std=(1))
imageInvTransforms = transforms.Normalize(mean=(2.0,2.0,0,0), std=(1))

modelPath = os.path.join( os.path.dirname(os.path.realpath(__file__)), "model_gen.pt")

print(modelPath)
generator = None

def process(images) :
    global generator
    if generator is None :
        generator = Generator()
        generator.load_state_dict(torch.load(modelPath, map_location=images.device))
        generator = generator.to(images.device)
        generator = generator.eval()
    with torch.no_grad() :
        #ppres, ppinfo = generator.preProc(((images,None)))
        #ppres = generator.postProc(ppres[0], ppinfo)
        #res = ppres
        nnres = generator.forward((images,None))
        res=nnres
        #resRat = 2/3
        #res = resRat * nnres + (1-resRat) * ppres
    return res



