import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
class VAE(nn.Module):
    def __init__(self,config,spkSet):
        super(VAE, self).__init__()
        # Get Feat Dim
        inputDim=config["VAE"]["INPUT_DIM"]
        # Define Encoder
        encCellSize=config["VAE"]["ENCODER_CELL"]
        encLayerNum=config["VAE"]["ENCODER_LAYER"]
        #self.enc_bn = nn.ModuleList([])
        self.encoder=nn.ModuleList([])
        currDim=inputDim
        for i in range(encLayerNum):
            self.encoder.append(nn.Linear(currDim,encCellSize))
            self.encoder[i].weight=nn.Parameter(torch.Tensor(encCellSize,currDim).uniform_(0,0.00001))
            self.encoder[i].bias=nn.Parameter(torch.zeros(encCellSize))
            #self.enc_bn.append(nn.BatchNorm1d(encCellSize,momentum=0.05))
            currDim=encCellSize
        # Define Latent
        latCellSize=config["VAE"]["LATENT_DIM"]
        self.latMu=nn.Linear(currDim,latCellSize)
        self.latMu.weight=nn.Parameter(torch.Tensor(latCellSize,currDim).uniform_(0,0.00001))
        self.latMu.bias=nn.Parameter(torch.zeros(latCellSize))
        self.latVar=nn.Linear(currDim,latCellSize)
        self.latVar.weight=nn.Parameter(torch.Tensor(latCellSize,currDim).uniform_(0,0.00001))
        self.latVar.bias=nn.Parameter(torch.zeros(latCellSize))
        # Define Y
        self.yType=config["VAE"]["SPK_INFO"]
        spkCellSize=config["VAE"]["SPEAKER_DIM"]
        if self.yType=="one-hot":
            yDim=len(spkSet)
            self.Y=np.zeros(yDim)
            self.spkDict={spkId:spkIdx for spkIdx, spkId in enumerate(spkSet)}
        else:
            print("Wrong Y Type")
            return 0
        self.spkEmb=nn.Linear(yDim,spkCellSize)
        self.spkEmb.weight=nn.Parameter(torch.Tensor(spkCellSize,yDim).uniform_(0,0.00001))
        self.spkEmb.bias=nn.Parameter(torch.zeros(spkCellSize))
        # Define Decoder
        decCellSize=config["VAE"]["DECODER_CELL"]
        decLayerNum=config["VAE"]["DECODER_LAYER"]
        self.decoder=nn.ModuleList([])
        #self.dec_bn = nn.ModuleList([])
        currDim=spkCellSize+latCellSize
        for i in range(decLayerNum):
            self.decoder.append(nn.Linear(currDim,decCellSize))
            self.decoder[i].weight=nn.Parameter(torch.Tensor(decCellSize,currDim).uniform_(0,0.00001))
            self.decoder[i].bias=nn.Parameter(torch.zeros(decCellSize))
            #self.dec_bn.append(nn.BatchNorm1d(encCellSize,momentum=0.05))
            currDim=decCellSize
        # Define Output
        '''
        self.outX=nn.Linear(currDim,inputDim)
        self.outX.weight=nn.Parameter(torch.ones(inputDim,currDim))
        self.outX.bias=nn.Parameter(torch.zeros(inputDim))
        '''
        # Define Output
        self.outMu=nn.Linear(currDim,inputDim)
        self.outMu.weight=nn.Parameter(torch.Tensor(inputDim,currDim).uniform_(0,0.00001))
        self.outMu.bias=nn.Parameter(torch.zeros(inputDim))

        self.outVar=nn.Linear(currDim,inputDim)
        self.outVar.weight=nn.Parameter(torch.Tensor(inputDim,currDim).uniform_(0,0.00001))
        self.outVar.bias=nn.Parameter(torch.zeros(inputDim))
        # Define Activation Function
        self.act=config["VAE"]["ACTIVATION"]
        if self.act=="relu":
            self.act=nn.ReLU()
        

    def encode(self, x):
        h=x
        for i, encLayer in enumerate(self.encoder):
            h=encLayer(h)
            h=self.act(h)
        
        mu = self.latMu
        logvar = self.latVar
        return mu,logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h=z
        for i, decLayer in enumerate(self.decoder):
            h=decLayer(h)
            h=self.act(h)
            #h=self.dec_bn[i](h)

        pred=self.outMu(h)
        return pred
    def decode_gauss(self, z):
        h=z
        for i, decLayer in enumerate(self.decoder):
            h=decLayer(h)
            h=self.act(h)
            #h=self.dec_bn[i](h)

        predMu=self.outMu(h)
        predVar=self.outVar(h)
        return predMu, predVar
    def getSpkInfo(self):
        return self.spkDict
    def setSpkInfo(self,spkDict):
        self.spkDict=spkDict
    def forward(self, x, spkIds, useGPU=0, printLog=False):
        #print("X",x)
        mu, logvar = self.encode(x)
        if printLog:
            print(mu)
        z = self.reparameterize(mu, logvar)
        spkIdxs=[self.spkDict[spkId] for spkId in spkIds]
        if self.yType=="one-hot":
            y = []
            for spkIdx in spkIdxs:
                curY=self.Y.copy()
                curY[spkIdx]=1
                y.append(curY)
            
            y=torch.Tensor(y).float()
            if useGPU==1:
                y=y.cuda()
        spkY=self.spkEmb(y)
        
        decInp = torch.cat((z,spkY),1)
        output = self.decode(decInp)
        
        #outMu,outVar = self.decode_gauss(decInp)
        #print("X_",output)
        #return outMu,outVar, mu, logvar
        return output, mu, logvar
    def transform(self, x, spkId, useGPU=0):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if self.yType=="one-hot":
            y_ = self.Y.copy()
            spkIdx=self.spkDict[spkId]
            y_[spkIdx]=1.0
            y=[y_ for count in range(len(x))]
            y=torch.Tensor(y).float()
            if useGPU==1:
                y=y.cuda()
        spkY=self.spkEmb(y)
        decInp=torch.cat((z,spkY),1)
        return self.decode(decInp)
# Reconstruction + KL divergence losses summed over all elements and batch
def VAE_LOSS(recon_x, x, mu, logvar, beta):
    BCE = F.mse_loss(recon_x, x)#,reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta * KLD

class VQ_VAE(nn.Module):
    def __init__(self,config,spkSet):
        super(VQ_VAE, self).__init__()
        # Encoder
        encInputDim=config["VQ_VAE"]["ENCODER"]["INPUT_DIM"]
        encLayerNum=config["VQ_VAE"]["ENCODER"]["LAYER_NUM"]
        encKernel=config["VQ_VAE"]["ENCODER"]["KERNEL"]
        encStride=config["VQ_VAE"]["ENCODER"]["STRIDE"]
        encPadding=config["VQ_VAE"]["ENCODER"]["PADDING"]
        encChannel=config["VQ_VAE"]["ENCODER"]["CHANNEL"]
        self.encoder=nn.ModuleList([])
        inChannel=1
        for i in range(encLayerNum):
            curLayer=nn.Sequential(
            nn.Conv1d(inChannel,encChannel[i],encKernel[i],encStride[i],encPadding[i]),
            nn.ReLU())
            #nn.MaxPool1d(encKernel[i],encStride[i]))
            self.encoder.append(curLayer)
            inChannel=encChannel[i]
        # Latent Codebook
        codeDim=config["VQ_VAE"]["CODEBOOK"]["CODE_DIM"]
        codeNum=config["VQ_VAE"]["CODEBOOK"]["CODE_NUM"]        
        self.codebook=nn.ParameterList([])
        for i in range(codeNum):
            self.codebook.append(nn.Parameter(torch.Tensor(codeDim).uniform_(0,0.00001)))
        # Speaker Embedding
        decInputDim=config["VQ_VAE"]["DECODER"]["INPUT_DIM"]

        yDim=len(spkSet)
        self.Y=np.zeros(yDim)
        self.spkDict={spkId:spkIdx for spkIdx, spkId in enumerate(spkSet)}
        
        self.spkEmb=nn.Linear(yDim+codeDim,decInputDim)
        self.spkEmb.weight=nn.Parameter(torch.Tensor(decInputDim,yDim+codeDim).uniform_(0,0.00001))
        self.spkEmb.bias=nn.Parameter(torch.zeros(decInputDim))
        # Decoder
        decLayerNum=config["VQ_VAE"]["DECODER"]["LAYER_NUM"]
        decKernel=config["VQ_VAE"]["DECODER"]["KERNEL"]
        decStride=config["VQ_VAE"]["DECODER"]["STRIDE"]
        decPadding=config["VQ_VAE"]["DECODER"]["PADDING"]
        decChannel=config["VQ_VAE"]["DECODER"]["CHANNEL"]

        self.decoder=nn.ModuleList([])
        inChannel=1
        for i in range(decLayerNum):
            curLayer=nn.Sequential(
            nn.ConvTranspose1d(inChannel,decChannel[i],decKernel[i],decStride[i],decPadding[i]),
            nn.ReLU())
            self.decoder.append(curLayer)
            inChannel=decChannel[i]
        # Define Output
        self.outMu=nn.Linear(encInputDim,encInputDim)
        self.outMu.weight=nn.Parameter(torch.Tensor(encInputDim,encInputDim).uniform_(0,0.00001))
        self.outMu.bias=nn.Parameter(torch.zeros(encInputDim))
    def encode(self, x):
        h=x
        for encLayer in self.encoder:
            h=encLayer(h)
        return h
    
    def find_codebook(self,z):
        minDist=9999999999999999
        batchSize=z.shape[0]
        codeDim=z.shape[2]
        minDists=[minDist for count in range(batchSize)]
        curCode=[[] for count in range(batchSize)]
        z=z.view(batchSize,codeDim)
        for code in self.codebook:
            curDists=torch.sum(((z-code) ** 2)/2,dim=1)
            for i, curDist in enumerate(curDists):
                if curDist < minDists[i]:
                    curCode[i]=code
                    minDists[i]=curDist
        curCodeSet=torch.Tensor().float()
        curCodeSet=curCodeSet.cuda()
        for code in curCode:
            curCodeSet=torch.cat((curCodeSet,code))
        curCode=curCodeSet.view(batchSize,codeDim)
        return curCode

    def decode(self, x):
        h=x
        for decLayer in self.decoder:
            h=decLayer(h)
        pred=self.outMu(h)
        return pred

    def forward(self, x, spkIds, useGPU=0, printLog=False):
        x=x.view(x.shape[0],1,x.shape[1])
        z = self.encode(x)
        code=self.find_codebook(z)
        spkIdxs=[self.spkDict[spkId] for spkId in spkIds]
        y = []
        for spkIdx in spkIdxs:
            curY=self.Y.copy()
            curY[spkIdx]=1
            y.append(curY)
        
        y=torch.Tensor(y).float()
        if useGPU==1:
            y=y.cuda()
        embInp = torch.cat((code,y),1)
        decInp=self.spkEmb(embInp)
        decInp=decInp.view(decInp.shape[0],1,decInp.shape[1])
        output = self.decode(decInp)
        output=output.view(output.shape[0],output.shape[2])
        z=z.view(z.shape[0],z.shape[2])
        return z, code, output
    def getSpkInfo(self):
        return self.spkDict
    def setSpkInfo(self,spkDict):
        self.spkDict=spkDict

def VQ_VAE_LOSS(originX,reconX,z,code,beta=0.25):
    BCE = F.mse_loss(reconX, originX)
    sgz=z.detach()
    sgcode=code.detach()
    KLD1 = F.mse_loss(sgz, code)
    KLD2 = F.mse_loss(z, sgcode) * beta

    #loss=encLoss+codeLoss+decLoss
    return BCE, KLD1, KLD2
    #return loss

        