import os
import glob
import commands
import string
import time
import errno
import shutil
import random



#==============================================================================
# getWorkingDir
#==============================================================================
def getWorkingDir():
    path = os.getenv('PBS_O_WORKDIR')
    if path==None:
        path = os.getenv('SGE_O_WORKDIR')
    if path==None:
        path = commands.getoutput('pwd')
    return path


#==============================================================================
# getQueueJobId
#==============================================================================
def getQueueJobId():
  qjid = os.getenv('PBS_JOBID')
  if qjid==None:
    qjid = None
  return qjid



#==============================================================================
# getQueueJobName
#==============================================================================
def getQueueJobName():
  qjn = os.getenv('PBS_JOBNAME')
  if qjn==None:
    qjn = None
  return qjn



#==============================================================================
# getHostName
#==============================================================================
def getHostName():
    host     = os.getenv('HOSTNAME')
    hostlist = string.split(host,'.')
    return hostlist[0]



#==============================================================================
# waitFor
#==============================================================================
def waitFor(fn):
    print 'Waiting for %s' %fn
    while True:
        if os.path.exists(fn):
            break
        time.sleep(5)
    print 'Found %s' %fn



#==============================================================================
# touch
#==============================================================================
def touch(filename):
    fp = open(filename, 'w')
    fp.close()



#==============================================================================
# getJobName
#
# (F. Faubel, Feb. 2008)
#==============================================================================
def getJobName():
    jobName = os.getenv('PBS_JOBNAME')
    if jobName==None:
        jobName = os.getenv('JOB_NAME')
    if jobName==None:
        jobName = str(os.getpid())
    return jobName



#==============================================================================
# getJobID
#
# (F. Faubel, Feb. 2008)
#==============================================================================
def getJobID():
    jobID = os.getenv('PBS_JOBID')
    if jobID==None:
        jobID = os.getenv('JOB_ID')
    if jobID==None:
        jobID = str(os.getpid())
    part = string.split(jobID, '.')
    return part[0]



#==============================================================================
# extendByJobName
#
# (F. Faubel, Feb. 2008)
#==============================================================================
def extendByJobName(name):
    jobName = getJobName()
    if len(name)>0:
        fname = '%s.%s' %(name, jobName)
    else:
        fname = '%s' %(jobName)
    return fname



#==============================================================================
# openRW
#
# (F. Faubel, Feb. 2008)
#==============================================================================
def openRW(fileName, copyFrom = ''):

    #-------------------------------
    # try to open file in mode "r+"
    #-------------------------------
    f = None
    cnt = 0
    while (f==None and cnt<3600):
        try:
            f = open(fileName, 'r+')
        except IOError, e:
            #---------------------------
            # create file, if necessary
            #---------------------------
            if e.errno==errno.ENOENT:
                try:
                    if len(copyFrom)>0:
                        shutil.copy(copyFrom, fileName)
                    else:
                        touch(fileName)
                except IOError:
                    pass
            #-----------------------------------------------
            # otherwise: wait some time before trying again
            #-----------------------------------------------
            else:
                time.sleep(1)
            f = None
        cnt += 1

    #---------------------------------------
    # number of retries exceeded: try again
    # to produce an error
    #---------------------------------------
    if cnt>=3600:
        f = open(fileName, 'r+')

    return f





#==============================================================================
# class CriticalSection
#
# (F. Faubel, March 2008)
#==============================================================================
class CriticalSection:

    #==========================================================================
    # constructor
    #==========================================================================
    def __init__(self, name = ''):
        self.path = getWorkingDir()
        print '%s' %(self.path)
        self.name = name
        if not len(self.name)==0:
            self.name = '-' + self.name
        self.fileName = os.path.join(self.path, getJobName() + self.name +'.sem.' + getHostName() + '.' + str(os.getpid()))

    #==========================================================================
    # enter
    #==========================================================================
    def enter(self):
        flist2 = ['...', '...']
        searchStr = os.path.join(self.path, '%s%s.sem.*' %(getJobName(), self.name))
        while len(flist2)>1:
            try:
                os.remove(self.fileName)
            except OSError:
                pass
            flist = glob.glob(searchStr)
            while len(flist)>0:
                time.sleep(random.randint(1,10))
                flist = glob.glob(searchStr)
            touch(self.fileName)
            time.sleep(10)
            flist2 = glob.glob(searchStr)


    #==========================================================================
    # leave
    #==========================================================================
    def leave(self):
        time.sleep(10)
        os.remove(self.fileName)



random.seed(os.getpid())
