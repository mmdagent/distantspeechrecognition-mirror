AM_CPPFLAGS = $(CPPFLAGS)
lib_LTLIBRARIES = libbtk_cancelVP.la
libbtk_cancelVP_la_SOURCES = \
	cancelVP.cc
libbtk_cancelVP_la_LIBADD  = $(top_srcdir)/stream/libbtk_stream.la
libbtk_cancelVP_la_LDFLAGS = $(GSL_LIBS)

btk_cancelVP_includedir = $(pkgincludedir)/cancelVP
btk_cancelVP_include_HEADERS = \
	cancelVP.h

pkgpython_PYTHON = cancelVP.py
pkgpyexec_LTLIBRARIES = _cancelVP.la

_cancelVP_la_SOURCES = cancelVP_wrap.cxx
_cancelVP_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_cancelVP_la_LIBADD = libbtk_cancelVP.la
_cancelVP_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = cancelVP_wrap.cxx cancelVP.py
EXTRA_DIST = cancelVP.i

SWIG_FILES = cancelVP_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include 

include $(SWIG_RULES)
