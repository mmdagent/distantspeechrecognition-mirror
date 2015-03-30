from fsm import *

fsmInFile = 'test.txt'
fsmOutFile = 'test-out.txt'
wfsa = WFSAcceptorPtr()
wfsa.read(fsmInFile, noSelfLoops = 0)
wfsa.write(fsmOutFile)

fsmInFile = 'test-transducer.txt'
fsmOutFile = 'test-transducer-out.txt'
# wfst = WFSTransducerPtr()
# wfst.read(fsmInFile, noSelfLoops = 0)
# wfst.write(fsmOutFile)
