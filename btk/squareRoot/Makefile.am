AM_CPPFLAGS = $(CPPFLAGS)

lib_LTLIBRARIES = libbtk_squareroot.la
libbtk_squareroot_la_SOURCES = squareRoot.cc
libbtk_squareroot_la_LDFLAGS = $(GSL_LIBS) 

btksquareroot_includedir = $(pkgincludedir)/squareRoot
btksquareroot_include_HEADERS = squareRoot.h

pkgpython_PYTHON = squareRoot.py kalman.py
pkgpyexec_LTLIBRARIES = _squareRoot.la

_squareRoot_la_SOURCES = squareRoot_wrap.cxx
_squareRoot_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_squareRoot_la_LIBADD = \
$(top_srcdir)/common/libbtk_common.la 
_squareRoot_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = squareRoot_wrap.cxx squareRoot.py
EXTRA_DIST = squareRoot.i

SWIG_FILES = squareRoot_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include 

include $(SWIG_RULES)
