#!/bin/python

#===========================================================#
# Contains utility functions for debugging and more.        #
#                                                           #
# Author: Uwe Mayer                                         #
# Date:   2004-10-10                                        #
#===========================================================#

PROTOTYPE_PICKLE_FILE = './prototype.pickle/prototype_M%d_m%d.pickle'
TEMPDIR = './tmp/'
COLOR_COUNT = 0

import sys
import os
import os.path
import pickle
import Gnuplot
import wave
from Numeric import *
from types import *
from struct import unpack

from btk.signalGen import REAL, IMAG, ComplexTypes


def createDirs(*arg):
    """recursively create project directories in <dirlist>"""
    for d in arg:
        if (not os.path.exists(d)):
            try:
                os.makedirs(d)
            except OSError: pass


def padData(data):
    """returns <data> 0-padded to the next 2**X"""
    fillSize = int(log10(len(data))/log10(2) +1)
    d = list(data)
    d.extend(zeros(2**fillSize-len(data)))
    return asarray(d)
 

def log(text):
    """writes text to stderr and flushes channel"""
    sys.stderr.write(text+"\n")
    sys.stderr.flush()


def saveWav(data, filename, channels=1, sampwidth=2, framerate=16000):
    """saves an array of data as a wave file

    Param: data      python list or array with the samples
           filename  output filename; path is created automatically
    """
    if (sampwidth != 2): raise NotImplementedError("sampwidth != 2 not implemented")

    if (not os.path.exists(os.path.dirname(filename))):
        createDirs(os.path.dirname(filename))
    f = wave.open(filename, "w")
    f.setnchannels(channels)
    f.setsampwidth(sampwidth)
    f.setframerate(framerate)
    f.writeframes(asarray(data).astype('s').tostring())
    f.close()


def loadWav(file):
    """return contents of a wav file as array

    Param: file     filename

    The function assumes the data is 16 bit signed (seminar-data).
    """
    w = wave.open(file)
    assert(w.getsampwidth() == 2)
    frames = w.getnframes()
    data = unpack("%dh"%frames, w.readframes(frames))
    w.close()
    return asarray(data)


def color(mod=8):
    """returns cyclic different colors"""
    global COLOR_COUNT
    COLOR_COUNT = (COLOR_COUNT+1)%mod
    return COLOR_COUNT


def createPrototype(M,m):
    """creates a pickled prototype by executing the python script"""
    os.system("python prototypeDesignTest.py %d %d"%(M,m))


def pickleLoadPrototype(M,m, auto=True):
    """returns a pickled version of a prototype

    Prototype is expected to have the filename:
      prototype_M<M>_m<m>.pickle

    Param: M,m   prototype parameters
           auto  wether to automatically create one
                 if none is found with those parameters
                 [default: True]
    """
    filename = PROTOTYPE_PICKLE_FILE %(M,m)
    if (not os.path.exists(filename)):
        createPrototype(M,m)

    return pickle.load(file(filename))
    
    

def gslVectorRead(filename):
    """returns a gsl vector dumped (as column vector) in C

    The vector may be complex or real valued, but the resulting
    python array is always complex.

    Param: filename   the basename of the file relative to \"tmp\"
    """
    try:
        f = open("./tmp/%s.dump"%(filename,));
    except IOError, e:
        print "could not open input file:",filename,e
        sys.exit(1)
        
    result = []
    for l in f.readlines():
        try:
            (re,im) = l.split(" ")
            result.append(complex(float(re), float(im)))
        except ValueError:
            re = l
            result.append(complex(float(re), 0))
    f.close()
    return asarray(result)



def gslVectorPlot(filename, color=1):
    """returns a plotitem of a vector dumped in C

    Param: filename     the basename of the file relative to \"tmp\"
           color        plot color [default: 1]
    """
    vector = gslVectorRead(filename);
    if (vector.typecode() in ComplexTypes):
        re = Gnuplot.Data(REAL(vector), with="lines "+str(color()), title="%s (real)"%(filename,))
        im = Gnuplot.Data(IMAG(vector), with="lines "+str(color()), title="%s (imag)"%(filename,))
        return re,im
    else:
        return Gnuplot.Data(vector, with="lines "+str(color), title=filename)




def gslMatrixRead(filename):
    """returns a gsl matrix dumped in C"""
    try:
        f = open("./tmp/%s.dump"%(filename,))
    except IOError, e:
        print "could not read from file:",filename,e
        sys.exit(1)
        
    format = map(lambda a: int(a), f.readline().split(' ')[1].split('x'))
    matrix = zeros(format, typecode=Complex)
    for row in range(format[0]):        # rows
        for col in range(format[1]):    # columns
            val = f.readline()
            try:
                (re,im) = val.split(" ")
                matrix[row,col] = complex(float(re), float(im))
            except ValueError:
                re = val
                matrix[row,col] = complex(float(re), 0)
    f.close()
    return asarray(matrix)



def dumpMatrix(matrix, f):
    """dumps a matrix to file as rectangular text

    Param: matrix    matrix to be dumped
           f         filename or file object
    """
    try:
        outfile = open(f, "w")
    except IOError, e:
        print "could not write to file:",f,e
        sys.exit(1)

    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            outfile.write("%s "%(matrix[row][col],))
        outfile.write("\n")
    outfile.close()


#-- initialisations ------------------------------------------------------------
createDirs(TEMPDIR)
