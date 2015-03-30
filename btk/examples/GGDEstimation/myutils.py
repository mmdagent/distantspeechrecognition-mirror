import string  
import os
import os.path
import shutil
import gzip
import pickle
import re
from types import FloatType
import getopt, sys

def getOutPrefix( nextFile, IDPos = 3 ):
    baseName =  os.path.basename( nextFile )
    baseNameElem = baseName.split( '.' )
    dirName = os.path.dirname( nextFile )
    outputPrefix = ''
    idx = 0
    for elem in dirName.split( '/' ):
        if idx >= IDPos :
            outputPrefix += elem + '/'
        idx += 1
    outputPrefix += baseNameElem[0]
    return outputPrefix
