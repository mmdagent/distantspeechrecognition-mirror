AM_CPPFLAGS = $(PYTHON_CPPFLAGS) $(BTK_CFLAGS) $(OPENCV_CFLAGS)

lib_LTLIBRARIES = libasr_train.la
libasr_train_la_LDFLAGS = $(BTK_LIBS)
libasr_train_la_LIBADD = \
	../dictionary/libasr_dict.la \
	../gaussian/libasr_gaussian.la \
	../path/libasr_path.la \
	../fsm/libasr_fsm.la \
	../lattice/libasr_lattice.la \
	../decoder/libasr_decoder.la \
	../adapt/libasr_adapt.la
libasr_train_la_SOURCES = codebookTrain.cc  distribTrain.cc  estimateAdapt.cc fsa.cc stateClustering.cc

asrtrain_includedir = $(pkgincludedir)/train
asrtrain_include_HEADERS = codebookTrain.h  distribTrain.h  estimateAdapt.h fsa.h stateClustering.h

pkgpython_PYTHON = train.py
pkgpyexec_LTLIBRARIES = _train.la

_train_la_SOURCES = train_wrap.cxx
_train_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_train_la_LIBADD = libasr_train.la
_train_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = train_wrap.cxx train.py
EXTRA_DIST = train.i

SWIG_FILES = train_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = $(AM_CPPFLAGS)

include $(SWIG_RULES)
