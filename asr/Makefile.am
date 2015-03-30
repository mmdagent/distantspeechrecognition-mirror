ACLOCAL_AMFLAGS = -I m4
SUBDIRS = \
	dictionary natural gaussian path fsm \
	lattice decoder adapt train sat lib
EXTRA_DIST = m4/swig.m4 swig_rules

pkginclude_HEADERS = config.h
pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = asr.pc

# try to import all modules
PYCHECK="import sys;sys.path.insert(0,\"$(pythondir)\");exec(\"import %s\" % sys.argv[1]);print \"Test passed.\""
installcheck-local:
	@ $(ECHO) $(PYCHECK) | $(PYTHON) - asr.dict
	@ $(ECHO) $(PYCHECK) | $(PYTHON) - asr.natural
	@ $(ECHO) $(PYCHECK) | $(PYTHON) - asr.gaussian
	@ $(ECHO) $(PYCHECK) | $(PYTHON) - asr.path
	@ $(ECHO) $(PYCHECK) | $(PYTHON) - asr.fsm
	@ $(ECHO) $(PYCHECK) | $(PYTHON) - asr.lattice
	@ $(ECHO) $(PYCHECK) | $(PYTHON) - asr.decoder
	@ $(ECHO) $(PYCHECK) | $(PYTHON) - asr.adapt
	@ $(ECHO) $(PYCHECK) | $(PYTHON) - asr.train
	@ $(ECHO) $(PYCHECK) | $(PYTHON) - asr.sat
