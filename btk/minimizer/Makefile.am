AM_CPPFLAGS = $(SFE_CFLAGS)
lib_LTLIBRARIES = libbtk_minimizer.la
libbtk_minimizer_la_SOURCES = \
	conjugate_pr.c
libbtk_minimizer_la_LDFLAGS = $(GSL_LIBS) $(SFE_LIBS) -lsfe_stream

btk_minimizer_includedir = $(pkgincludedir)/minimizer
btk_minimizer_include_HEADERS = \
	lsv_multimin.h

pkgpython_PYTHON = minimizer.py
pkgpyexec_LTLIBRARIES = _minimizer.la

_minimizer_la_SOURCES = minimizer_wrap.cxx
_minimizer_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_minimizer_la_LIBADD = libbtk_minimizer.la
_minimizer_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = minimizer_wrap.cxx minimizer.py
EXTRA_DIST = minimizer.i

SWIG_FILES = minimizer_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include $(AM_CPPFLAGS)

include $(SWIG_RULES)
