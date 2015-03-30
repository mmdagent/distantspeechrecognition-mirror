from asr import common
from asr.fsm import *

statelex = LexiconPtr(nm='stateLex')
inlex    = LexiconPtr(nm='inLex',fileName='/home/jmcd/src/asr/test/Lexicon.txt')
outlex   = LexiconPtr(nm='outLex')
tr       = WFSTransducerPtr(statelex,inlex,outlex)

contextLen = 2
eps        = 'eps'
end        = '#'
wb         = 'WB'
tr.buildC(contextLen = contextLen, eps = eps, end = end, wb = wb)
tr.write(fileName='/home/jmcd/src/asr/test/ContextTransducer.txt', useSymbols = 1)
tr.inputLexicon().write('/home/jmcd/src/asr/test/InputLexicon.txt')
tr.outputLexicon().write('/home/jmcd/src/asr/test/OutputLexicon.txt')
tr.stateLexicon().write('/home/jmcd/src/asr/test/StateLexicon.txt')
