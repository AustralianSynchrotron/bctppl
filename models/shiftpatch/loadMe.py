#import shiftpatch_module as sg
import torch
import torch.nn as nn
import math
import os


latentDim = 64
inShape = (80,80)
batchsize = 2**8
save_interim = False

class Generator(nn.Module):

    def createLatent(self) :
        if self.latentChannels == 0 :
            return None
        realLatent = abs(self.latentChannels)
        toRet =  nn.Sequential(
            nn.Linear(latentDim, math.prod(inShape) * realLatent),
            nn.ReLU(),
            nn.Unflatten( 1, (realLatent,) +inShape )
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
            layers.append(nn.BatchNorm2d(chOut))
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
            layers.append(nn.BatchNorm2d(chOut))
        layers.append(nn.LeakyReLU(0.2))
        return torch.nn.Sequential(*layers)


    def createFClink(self) :
        smpl = torch.zeros((1, 4+abs(self.latentChannels), *inShape))
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
            nn.Conv2d(self.baseChannels+4, 2, 1),
            nn.Tanh(),
        )
        return toRet


    def __init__(self, latentChannels=0):
        super(Generator, self).__init__()
        self.latentChannels = latentChannels
        self.baseChannels = 64
        self.amplitude = 4

        self.noise2latent = self.createLatent()

        self.encoders =  nn.ModuleList([
            self.encblock( (4+abs(self.latentChannels)) /self.baseChannels,
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

        #sg.load_model(self, model_path="model_0_gen.pt" )


    def preProc(self, images):
        with torch.no_grad() :
            orgDims = images.dim()
            if orgDims == 3 :
                images = images.view(1, *images.shape)
            masks = images[:,2:,...]
            presentInBoth = masks[:,0,...] * masks[:,1,...]
            sumOrg = torch.where(presentInBoth > 0, 0.5 + images[:,0,...], 0).sum(dim=(-1,-2))
            sumSft = torch.where(presentInBoth > 0, 0.5 + images[:,1,...], 0).sum(dim=(-1,-2))
            procImages = images[:,0:4,...].clone().detach()
            #blurImages = blurTransform(procImages)
            #blurImages[:,0:2,...] /= torch.where ( blurImages[:,2:4,...] > 0, blurImages[:,2:4,...], 1 )
            coef = torch.sqrt( torch.where( sumOrg * sumSft != 0, sumOrg / sumSft, 1 ) ).view(-1,1,1)
            procImages[:,0,...] = masks[:,0,...] * \
                ( procImages[:,0,...] / coef - 0.5 * (1 - 1/coef) )
            procImages[:,1,...] = masks[:,1,...] * \
                ( procImages[:,1,...] * coef - 0.5 * (1 - coef) )
        return procImages, (coef, orgDims)


    def postProc(self, images, cfg):
        coef, orgDims = cfg
        pimages0 = images[:,0,...] * coef + 0.5 * (coef - 1)
        pimages1 = images[:,1,...] / coef + 0.5 * (1/coef - 1)
        pimages = torch.stack((pimages0, pimages1), dim=1)
        if orgDims == 3 :
            pimages = pimages.view(1, *images.shape)
        return pimages


    def forward(self, input):
        global save_interim

        if not save_interim is None :
            save_interim = {}
        def saveToInterim(key, data) :
            if not save_interim is None :
                save_interim[key] = data.clone().detach()

        images, noises = input
        saveToInterim('input', images )
        images, procInf = self.preProc(images)
        #return self.postProc(images[:,[1,0],...], procInf)

        if self.latentChannels :
            latent = self.noise2latent(noises) \
                if self.latentChannels > 0 else \
                     2 * torch.rand((images.shape[0], -self.latentChannels, *inShape),
                                device = images.device) - 1
            dwTrain = [torch.cat((images, latent), dim=1),]
        else :
            dwTrain = [images,]
        for encoder in self.encoders :
            dwTrain.append(encoder(dwTrain[-1]))
        mid = self.fcLink(dwTrain[-1])
        upTrain = [mid]
        for level, decoder in enumerate(self.decoders) :
            upTrain.append( decoder( torch.cat( (upTrain[-1], dwTrain[-1-level]), dim=1 ) ) )
        res = self.lastTouch(torch.cat( (upTrain[-1], images ), dim=1 ))

        res = self.postProc(res, procInf)
        saveToInterim('output', res)
        return res


#sg.optimizer_G = sg.createOptimizer(sg.generator, sg.TCfg.learningRateG)
#model_summary = summary(sg.generator, input_data=[ [sg.refImages[[0],0:4,...], sg.refNoises[[0],...]] ] ).__str__()
#print(model_summary)

#_ = sg.testMe(sg.refImages)


def createGenerator(device):
    generator = Generator(-1)
    modelDir = os.path.dirname(os.path.abspath(__file__))
    modelPath = os.path.join(modelDir, "model_gen.pt")
    generator.load_state_dict(torch.load(modelPath, map_location=device))
    generator = generator.to(device).eval()
    return generator

