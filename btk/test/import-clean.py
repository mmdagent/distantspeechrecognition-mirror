
from importdb import convert
from os.path import exists
import os

if not exists("data"):
	os.symlink("../../testdata", "data")

convert("data/db1/db/clean-spk.dat", "clean-spk")
convert("data/db1/db/clean-utt.dat", "clean-utt", "UTT")
