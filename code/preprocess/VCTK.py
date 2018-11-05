# Input : wav file directory
# Output : feature, spk id dataset
# Read dataSet Directory
# Extract Wav Info List {Wav ID : PATH, SPK_ID, UTT_ID}
# Extract feature (Wav -(Vocoder)-> Feature)
# Save File

# Result : In data/wavFeatDict.pk => { Wav ID : wavFeature() }
from code.include import env, dataSet, neuralNets
from code.include import fileManager as fm
from code.include import logManager as lm
from . import vocoder
import os, sys


class WavMetaInfoExtractor:
    def __init__(self):
        self.wavMetaInfo=dict()
        self.wavMetaInfo["PATH"]=dict()
        self.wavMetaInfo["SPK"]=dict()
        self.wavMetaInfo["UTT"]=dict()
    def __setWavPath__(self,wavId,wavPath):
        self.wavMetaInfo["PATH"][wavId]=wavPath
    def __setWavSpk__(self,wavId,spkId):
        self.wavMetaInfo["SPK"][wavId]=spkId
    def __setWavUtt__(self,wavId,uttId):
        self.wavMetaInfo["UTT"][wavId]=uttId

    def readCorpus(self,corpusDir,wavPathListPath):
        findCommand="find "+corpusDir+" -iname '*.wav' > "+ wavPathListPath
        os.system(findCommand)
        wavPathList=self.__loadWavPathList__(wavPathListPath)
        self.__setWavMeta__(wavPathList)

    def readCorpusfromConfig(self,config):
        corpusDir=config.getCorpusPath()
        wavPathListPath=config.getWorkingFilePath("WAV_PATH_LIST")
        self.readCorpus(corpusDir,wavPathListPath)
    
    def __VCTK_extract_FilePath__(self,line):
        return line[:-1]
    def __VCTK_extract_WavId__(self,wavPath):
        fileName=wavPath.split("/")[-1]
        wavId=fileName.split(".")[0]
        return wavId
    def __VCTK_extract_SpkId__(self,wavId):
        spkId=wavId.split("_")[0]
        return spkId
    def __VCTK_extract_UttId__(self,wavId):
        uttId=wavId.split("_")[0]
        return uttId
    def __loadWavPathList__(self,wavPathListPath):
        wavPathList=[]
        with open(wavPathListPath,'r') as f:
            for line in f:
                wavPath=self.__VCTK_extract_FilePath__(line)
                wavPathList.append(wavPath)
        return wavPathList
    def __setWavMeta__(self,wavPathList):
        for wavPath in wavPathList:
            wavId=self.__VCTK_extract_WavId__(wavPath)
            self.__setWavPath__(wavId,wavPath)

            spkId=self.__VCTK_extract_SpkId__(wavId)
            self.__setWavSpk__(wavId,spkId)

            uttId=self.__VCTK_extract_UttId__(wavId)
            self.__setWavUtt__(wavId,uttId)
    
    def getWavPathDict(self):
        return self.wavMetaInfo["PATH"]

class WavFeatExtractor:
    def __init__(self):
        #self.wavVocoder=vocoder.Vocoder()
        self.wavPath=""
        self.wavPathDict=dict()
    def setVocoder(self,vocoderType):
        if vocoderType=="WORLD":
            self.wavVocoder=vocoder.WORLD_VOCODER()
    def setVocoderfromConfig(self,config):
        vocoderType=config.getVocoderType()
        self.setVocoder(vocoderType)

    def extractFeat(self,wavPath):
        wavFeature=self.wavVocoder.wavToFeat(wavPath)
        return wavFeature

    def extractFeatDict(self,wavPathDict):
        wavFeatDict=dict()
        for wavId,wavPath in wavPathDict.items():
            wavFeature=self.extractFeat(wavPath)
            wavFeatDict[wavId]=wavFeature
        return wavFeatDict
    def saveFeat(self,wavFeatPath,wavFeat):
        fm.saveVariable(wavFeatPath,wavFeat)
    def extractAndSaveFeatDict(self,wavPathDict,config):
        for wavId,wavPath in wavPathDict.items():
            print(wavId)
            wavFeatPath=config.getFeaturePath(wavId)
            wavFeature=self.extractFeat(wavPath)
            self.saveFeat(wavFeatPath,wavFeature)

def main():
    configPath=sys.argv[1]
    config=env.Config(configPath)

    wavMetaInfoExtractor=WavMetaInfoExtractor()
    wavMetaInfoExtractor.readCorpusfromConfig(config)
    wavPathDict=wavMetaInfoExtractor.getWavPathDict()

    wavFeatExtractor=WavFeatExtractor()
    
    wavFeatExtractor.setVocoderfromConfig(config)
    wavFeatExtractor.extractAndSaveFeatDict(wavPathDict,config)

    return 0

if __name__=="__main__":
    main()
