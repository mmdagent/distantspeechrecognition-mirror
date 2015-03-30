AM_CPPFLAGS = $(CPPFLAGS) 
lib_LTLIBRARIES = libbtk_maximumLikelihood.la
libbtk_maximumLikelihood_la_SOURCES = maximumLikelihood.cc
libbtk_maximumLikelihood_la_LIBADD = $(top_srcdir)/stream/libbtk_stream.la $(top_srcdir)/feature/libbtk_feature.la $(top_srcdir)/modulated/libbtk_modulated.la
libbtk_maximumLikelihood_la_LDFLAGS = $(GSL_LIBS)

btk_maximumLikelihood_includedir = $(pkgincludedir)/maximumLikelihood
btk_maximumLikelihood_include_HEADERS = maximumLikelihood.h

pkgpython_PYTHON = maximumLikelihood.py
pkgpyexec_LTLIBRARIES = _maximumLikelihood.la

_maximumLikelihood_la_SOURCES = maximumLikelihood_wrap.cxx
_maximumLikelihood_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_maximumLikelihood_la_LIBADD = libbtk_maximumLikelihood.la
_maximumLikelihood_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = maximumLikelihood_wrap.cxx maximumLikelihood.py
EXTRA_DIST = maximumLikelihood.i

SWIG_FILES = maximumLikelihood_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include 

include $(SWIG_RULES)
