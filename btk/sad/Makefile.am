AM_CPPFLAGS = $(CPPFLAGS)
lib_LTLIBRARIES = libbtk_sad.la
libbtk_sad_la_SOURCES = sad.cc ica.cc sadFeature.cc neural_spnsp_det.c
libbtk_sad_la_LIBADD = $(top_srcdir)/stream/libbtk_stream.la $(top_srcdir)/feature/libbtk_feature.la
libbtk_sad_la_LDFLAGS = $(GSL_LIBS)

btk_sad_includedir = $(pkgincludedir)/sad
btk_sad_include_HEADERS = sad.h ica.h sadFeature.h neural_spnsp_incl.h

pkgpython_PYTHON = sad.py
pkgpyexec_LTLIBRARIES = _sad.la

_sad_la_SOURCES = sad_wrap.cxx
_sad_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_sad_la_LIBADD = libbtk_sad.la
_sad_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = sad_wrap.cxx sad.py
EXTRA_DIST = sad.i sad.txt

SWIG_FILES = sad_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include 

include $(SWIG_RULES)
