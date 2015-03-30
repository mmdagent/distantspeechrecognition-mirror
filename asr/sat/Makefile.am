AM_CPPFLAGS = $(PYTHON_CPPFLAGS) $(BTK_CFLAGS) $(OPENCV_CFLAGS)

lib_LTLIBRARIES = libasr_sat.la
libasr_sat_la_LDFLAGS = $(BTK_LIBS)
libasr_sat_la_LIBADD = \
	../gaussian/libasr_gaussian.la \
	../train/libasr_train.la
libasr_sat_la_SOURCES = cluster.cc  codebookFastSAT.cc  sat.cc

asrsat_includedir = $(pkgincludedir)/sat
asrsat_include_HEADERS = cluster.h  codebookFastSAT.h  sat.h

pkgpython_PYTHON = sat.py
pkgpyexec_LTLIBRARIES = _sat.la

_sat_la_SOURCES = sat_wrap.cxx
_sat_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_sat_la_LIBADD = libasr_sat.la
_sat_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = sat_wrap.cxx sat.py
EXTRA_DIST = sat.i

SWIG_FILES = sat_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = $(AM_CPPFLAGS)

include $(SWIG_RULES)
