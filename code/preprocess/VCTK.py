# Input : wav file directory
# Output : feature, spk id dataset
# Read dataSet Directory
# Extract Wav Info List {Wav ID : PATH, SPK_ID, UTT_ID}
# Extract feature (Wav -(Vocoder)-> Feature)
# Save File
from code.include import env, dataSet, neuralNets
import vocoder
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
        self.wavMetaInfo["PATH"][wavId]=spkId
    def __setWavUtt__(self,wavId,uttId):
        self.wavMetaInfo["PATH"][wavId]=uttId
    def readWavDirectory(self,originWavDir,wavPathListPath):
        findCommand="find "+originWavDir+" -iname '*.wav' > "+ wavPathListPath
        os.system(findCommand)
        wavPathList=self.__loadWavPathList__(wavPathListPath)
        self.__saveWavMeta__(wavPathList)
    
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
    def __saveWavMeta__(self,wavPathList):
        for wavPath in wavPathList:
            wavId=self.__VCTK_extract_WavId__(wavPath)
            self.__setWavPath__(wavId,wavPath)

            spkId=self.__VCTK_extract_SpkId__(wavId)
            self.__setWavSpk__(wavId,spkId)

            uttId=self.__VCTK_extract_UttId__(wavId)
            self.__setWavUtt__(wavId,uttId)
    
    def getWavPathDict(self):
        return self.wavMetaInfo["PATH"]

def main():
    configPath=sys.argv[1]
    config=env.loadConfig(configPath)
    originDataDir=config["FILE_DIR"]["ORIGIN_DATA_DIR"]
    workingDataDir=config["FILE_DIR"]["WORKING_DATA_DIR"]
    wavPathListFileName=config["FILE_DIR"]["WAV_PATH_LIST"]
    wavPathListPath=workingDataDir+"/"+wavPathListFileName
    wavMetaInfoExtractor=WavMetaInfoExtractor()
    wavMetaInfoExtractor.readWavDirectory(originDataDir,wavPathListPath)
    wavPathDict=wavMetaInfoExtractor.getWavPathDict()


    return 0

if __name__=="__main__":
    main()
