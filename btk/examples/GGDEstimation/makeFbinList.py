import sys

M = 512
FbinList = 'fbinL'

if len(sys.argv) > 1:
    M = int(sys.argv[1])

fp = open( FbinList, 'w' )
for i in range(1,M/2+1):
    fp.write( '%d\n' %i )

fp.close()
