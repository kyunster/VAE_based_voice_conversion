import pickle as pk
import os

def makeDirectory(dirPath):
    os.system("mkdir -p "+dirPath)

def saveVariable(savePath,saveVariable):
    saveFilebyPickle(savePath,saveVariable)

def loadVariable(loadPath):
    loadVariable=loadFilebyPickle(loadPath)
    return loadVariable

###############################
def saveFilebyPickle(savePath,saveVariable):
    with open(savePath,'wb') as f:
        pk.dump(saveVariable,f)

def loadFilebyPickle(loadPath):
    with open(loadPath,'wb') as f:
        loadVariable=pk.load(f)
    return loadVariable


