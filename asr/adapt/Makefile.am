AM_CPPFLAGS = $(BTK_CFLAGS) $(GSL_CFLAGS) $(OPENCV_CFLAGS)

lib_LTLIBRARIES = libasr_adapt.la
libasr_adapt_la_LDFLAGS = $(BTK_LIBS) $(GSL_LIBS) -lbtk_feature 
libasr_adapt_la_LIBADD = \
 ../gaussian/libasr_gaussian.la \
 ../natural/libasr_natural.la
libasr_adapt_la_SOURCES = codebookAdapt.cc transform.cc

asradapt_includedir = $(pkgincludedir)/adapt
asradapt_include_HEADERS = codebookAdapt.h transform.h

pkgpython_PYTHON = adapt.py
pkgpyexec_LTLIBRARIES = _adapt.la

_adapt_la_SOURCES = adapt_wrap.cxx
_adapt_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_adapt_la_LIBADD = libasr_adapt.la
_adapt_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = adapt_wrap.cxx adapt.py
EXTRA_DIST = adapt.i

SWIG_FILES = adapt_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = $(AM_CPPFLAGS)

include $(SWIG_RULES)
