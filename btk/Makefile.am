ACLOCAL_AMFLAGS = -I m4
SUBDIRS = \
	include \
	gnulib \
	common \
	utils \
	stream \
	matrix \
	driver \
	feature \
	convolution \
	lms \
	squareRoot \
	portnoff \
	modulated \
	postfilter \
	beamformer \
	sad \
	localization \
	TDEstimator \
	maximumLikelihood \
	objectiveMeasure \
	cancelVP \
	src \
	lib

EXTRA_DIST = m4/swig.m4 swig_rules

pkginclude_HEADERS = btk.h
pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = btk.pc

# try to import all modules
PYCHECK="import sys;sys.path.insert(0,\"$(pythondir)\");exec(\"import %s\" % sys.argv[1]);print \"Test passed.\""
installcheck-local:
	@ $(ECHO) $(PYCHECK) | $(PYTHON) - btk.convolution
	@ $(ECHO) $(PYCHECK) | $(PYTHON) - btk.lms
	@ $(ECHO) $(PYCHECK) | $(PYTHON) - btk.squareRoot
	@ $(ECHO) $(PYCHECK) | $(PYTHON) - btk.portnoff
	@ $(ECHO) $(PYCHECK) | $(PYTHON) - btk.modulated
	@ $(ECHO) $(PYCHECK) | $(PYTHON) - btk.postfilter
	@ $(ECHO) $(PYCHECK) | $(PYTHON) - btk.beamformer
	@ $(ECHO) $(PYCHECK) | $(PYTHON) - btk.localization
	@ $(ECHO) $(PYCHECK) | $(PYTHON) - btk.maximumLikelihood
	@ $(ECHO) $(PYCHECK) | $(PYTHON) - btk.TDEstimator
	@ $(ECHO) $(PYCHECK) | $(PYTHON) - btk.cancelVP
