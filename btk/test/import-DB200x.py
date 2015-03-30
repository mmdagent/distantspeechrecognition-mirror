
from importdb import convert
from os.path import exists
import os

if not exists("data"):
	os.symlink("../../testdata", "data")

convert("/home/jmcd/data/esst/COMMON/db/DB2005-spk.dat", "DB2005-spk", "KEY")
convert("/home/jmcd/data/esst/COMMON/db/DB2005-utt.dat", "DB2005-utt", "KEY")

convert("/home/jmcd/data/esst/COMMON/db/DB2006-spk.dat", "DB2006-spk", "KEY")
convert("/home/jmcd/data/esst/COMMON/db/DB2006-utt.dat", "DB2006-utt", "KEY")
