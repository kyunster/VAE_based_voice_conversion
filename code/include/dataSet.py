from torch.utils.data.dataset import Dataset
import scipy.io
import pickle as pk
import numpy as np
import queue

class matLabDataSet(Dataset):
    def __init__(self,config,dset):
        super(matLabDataSet, self).__init__()
        
        # data_path
        dataPath=config["EXP"][dset+"_PATH"]
        print("Process ",dset)
        # Open SP & F0
        matDictPath=dataPath+"/mat.pk"
        self.SPs=dict(); self.F0=dict()
        with open(matDictPath,'rb') as f:
            matDict=pk.load(f)
        for uttId, matPath in matDict.items():
            curMat=scipy.io.loadmat(matPath)
            self.SPs[uttId]=curMat['sp']
            self.F0[uttId]=np.reshape(curMat['f0'],(-1,1))

        '''
        # Open SP
        # sp & f0 path
        spPath=dataPath+"/sp.pk"
        f0Path=dataPath+"/f0.pk"
        with open(spPath,'rb') as f:
            self.SPs=pk.load(f)
        # Open F0
        with open(f0Path,'rb') as f:
            self.F0=pk.load(f)
        '''
        # Get SPK List
        self.X=[]
        self.Y=[]
        self.spkSet=set()
        self.ESet=dict()
        for uttId, SP in self.SPs.items():
            SP=SP.T
            SP=np.log(SP)
            #E=np.reshape(np.sum(SP,axis=1),(-1,1))
            #E=np.reshape(np.mean(SP**2,axis=1),(-1,1))
            #SP=SP/E
            
            #self.ESet[uttId]=E
            self.SPs[uttId]=SP
            spkId=uttId.split("_")[0]
            self.X.extend(SP)
            self.Y.extend([spkId for count in range(len(SP))])
            self.spkSet.add(spkId)
    def __getitem__(self,index):
        curX=self.X[index]
        curY=self.Y[index]
        return (curX,curY)
    def __len__(self):
        return len(self.X)
    def get_spk_set(self):
        return self.spkSet
    def get_SPs(self):
        return self.SPs
    def get_E(self,uttId):
        return self.ESet[uttId]
    def get_F0(self):
        return self.F0

class matLabDataSetMemoryPreserve(Dataset):
    def __init__(self,config,dset):
        super(matLabDataSetMemoryPreserve, self).__init__()
        
        # data_path
        dataPath=config["EXP"][dset+"_PATH"]
        print("Process",dset)
        '''
        # Open SP & F0
        #matDictPath=dataPath+"/mat.pk"
        self.uttSet=[]
        self.spkSet=set()
        self.xInfo=[]
        self.yInfo=dict()
        #with open(matDictPath,'rb') as f:
        #    matDict=pk.load(f)
        spPath=dataPath+"/sp.pk"
        with open(spPath,'rb') as f:
            spDict=pk.load(f)
        for uttId, spPath in spDict.items():
            curSP=np.load(spPath)
            frameLen=len(curSP.T)
            for frameCount in range(frameLen):
                curInfo=[uttId,frameCount]
                self.xInfo.append(curInfo)
            spkId=uttId.split("_")[0]
            self.yInfo[uttId]=spkId
            self.uttSet.append(uttId)
            self.spkSet.add(spkId)
        
        # SP Cache
        cacheMaxSize=config["CACHE"]["MAX_UTT"]
        self.SP_Cache=uttCache(cacheMaxSize,spDict)
        '''
        spPath=dataPath+"/sp.pk"
        f0Path=dataPath+"/f0.pk"
        with open(spPath,'rb') as f:
            spDict=pk.load(f)
        
        with open(f0Path,'rb') as f:
            f0Dict=pk.load(f)
        self.X=[]
        self.Y=[]
        self.ESet=dict()
        self.spkSet=set()
        for uttId, spPath in spDict.items():
            curSP=np.load(spPath)
            curSP=curSP.T
            E1=np.reshape(np.sum(curSP,axis=1),(-1,1))
            #curSP=curSP/E1
            curSP=np.log(curSP)
            #mE2=np.reshape(np.mean(curSP,axis=0),(1,-1))
            #vE2=np.reshape(np.std(curSP,axis=0),(1,-1))
            #curSP=(curSP-mE2)/vE2
            #self.ESet[uttId]=[E1,mE2,vE2]
            self.ESet[uttId]=E1
            
            self.X.extend(curSP)
            spkId=uttId.split("_")[0]
            self.Y.extend([spkId for count in range(len(curSP))])
            self.spkSet.add(spkId)
        self.F0=dict()
        for uttId, f0Path in f0Dict.items():
            curF0=np.load(f0Path)
            self.F0[uttId]=np.reshape(curF0,(-1,1))

    def __getitem__(self,index):
        '''
        curInfo=self.xInfo[index]
        uttId=curInfo[0]
        curFrame=curInfo[1]
        curX=self.SP_Cache.getFrame(uttId,curFrame)
        curY=self.yInfo[uttId]
        '''
        curX=self.X[index]
        curY=self.Y[index]
        return (curX,curY)
    def __len__(self):
        #return len(self.xInfo)
        return len(self.X)
    def get_spk_set(self):
        return self.spkSet
    def get_F0(self):
        return self.F0
    #def flush_cache(self):
    #    self.SP_Cache.flush()
    def get_E(self,uttId):
        return self.ESet[uttId]

class uttCache():
    def __init__(self,maxSize,matDict):
        self.maxSize=maxSize
        self.curSize=0
        self.matDict=matDict
        self.uttStore=dict()
        self.curUtt=queue.Queue()
    def flush(self):
        self.curUtt=queue.Queue()
        self.uttStore=dict()
    def isFull(self):
        if self.curSize >= self.maxSize:
            return True
        else:
            return False

    def loadUtt(self,uttId):
        matPath=self.matDict[uttId]
        #curMat=scipy.io.loadmat(matPath)
        curX=np.load(matPath)
        #curX=curMat['sp'].T
        curX=np.log(curX.T)
        self.uttStore[uttId]=curX
        self.curUtt.put(uttId)
        self.curSize += 1

    def deleteUtt(self):
        uttId=self.curUtt.get()
        self.uttStore.pop(uttId,None)
        self.curSize -= 1

    def getSP(self,uttId):
        curSP=[]
        while len(curSP) <= 0 :
            curSP=self.uttStore.get(uttId,[])
            if len(curSP) > 0 :
                break
            if self.isFull():
                #print('Cache Full')
                self.deleteUtt()
            self.loadUtt(uttId)
        return curSP
    
    def getFrame(self,uttId,frameIdx):
        curSP=self.getSP(uttId)
        #print(len(curSP))
        curFrame=curSP[frameIdx]
        return curFrame



