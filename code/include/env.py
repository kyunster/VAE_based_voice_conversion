import json

class Config:
    def __jsonOpen__(self,jsonPath):
        with open(jsonPath, 'r') as f:
            jsonDict = json.load(f)
        return jsonDict

    def __init__(self,cfgPath):
        configDict=self.__jsonOpen__(cfgPath)
        self.config=configDict

    def getConfig(self,groupName,configName):
        return self.config[groupName][configName]

    def getCorpusPath(self):
        corpusPath=self.getConfig("FILE_DIR","CORPUS_DIR")
        return corpusPath
    
    def getVocoderType(self):
        vocoderType=self.getConfig("FEATURE","VOCODER_TYPE")
        return vocoderType

    def getWorkingFilePath(self,fileType):
        workingDataDir=self.config["FILE_DIR"]["WORKING_DATA_DIR"]
        workingFileName=self.config["FILE_NAME"][fileType]
        workingFilePath=workingDataDir+"/"+workingFileName
        return workingFilePath
    
    def getFeaturePath(self,wavId,ext=".pk"):
        wavFeatDir=self.config["FILE_DIR"]["WAV_FEATURE_DIR"]
        wavFeatPath=wavFeatDir+"/"+wavId+ext
        return wavFeatPath
    
    