import pyworld as pw
from scipy.io import wavfile

class RawWavWithSampleRate:
    def __fixRawWav__(self,rawWav):
        return rawWav.astype(float)
    def __fixSampleRate__(self,sampleRate):
        return float(sampleRate)
    def __fixVar__(self,rawWav,sampleRate):
        rawWav=self.__fixRawWav__(rawWav)
        sampleRate=self.__fixSampleRate__(sampleRate)
        return rawWav,sampleRate

    def __setVar__(self,rawWav,sampleRate):
        self.rawWav=rawWav
        self.sampleRate=sampleRate

    def __init__(self,rawWav,sampleRate):
        rawWav,sampleRate=self.__fixVar__(rawWav,sampleRate)
        self.__setVar__(rawWav,sampleRate)
        
    def getRawWav(self):
        return self.rawWav
    def getSampleRate(self):
        return self.sampleRate

class WavFeature:
    def __init__(self,f0,sp,ap,fs):
        self.feat=dict()
        self.setF0(f0); self.setSP(sp)
        self.setAP(ap); self.setFS(fs)
    def setFeat(self,feat):
        self.feat=feat
    def setF0(self,f0):
        self.feat["F0"]=f0
    def setSP(self,sp):
        self.feat["SP"]=sp
    def setAP(self,ap):
        self.feat["AP"]=ap
    def setFS(self,fs):
        self.feat["FS"]=fs
    def getFeat(self):
        return self.feat
    def getF0(self):
        return self.feat["F0"]
    def getSP(self):
        return self.feat["SP"]
    def getAP(self):
        return self.feat["AP"]
    def getFS(self):
        return self.feat["FS"]
    
class Vocoder:
    def __init__(self):
        self.vocoderType=""
    
    def wavRead(self,wavPath):
        sampleRate, rawWav = wavfile.read(wavPath)
        rawWavWithSampleRate=RawWavWithSampleRate(rawWav,sampleRate)
        return rawWavWithSampleRate
    def extractFeat(self,rawWavWithSampleRate):
        pass
    def wavToFeat(self,wavPath):
        rawWavWithSampleRate=self.wavRead(wavPath)
        wavFeat=self.extractFeat(rawWavWithSampleRate)
        return wavFeat

class WORLD_VOCODER(Vocoder):
    def __init__(self):
        super(WORLD_VOCODER,self).__init__()
    def extractTotalFeat(self,rawWavWithSampleRate):
        rawWav=rawWavWithSampleRate.getRawWav()
        sampleRate=rawWavWithSampleRate.getSampleRate()
        f0, sp, ap = pw.wav2world(rawWav, sampleRate)
        wavFeat=WavFeature(f0,sp,ap,sampleRate)
        return wavFeat
    def extractFeat(self,rawWavWithSampleRate):
        wavFeat=self.extractTotalFeat(rawWavWithSampleRate)
        return wavFeat
        

    
