AM_CPPFLAGS = $(CPPFLAGS) $(NUMPY_CPPFLAGS)
AM_CXXFLAGS = $(NUMPY_CPPFLAGS)

lib_LTLIBRARIES = libbtk_dereverberation.la
libbtk_dereverberation_la_SOURCES = \
	dereverberation.cc
libbtk_dereverberation_la_LDFLAGS = $(GSL_LIBS) -lbtk_feature -lbtk_stream

btk_dereverberation_includedir = $(pkgincludedir)/dereverberation
btk_dereverberation_include_HEADERS = \
	dereverberation.h

pkgpython_PYTHON = dereverberation.py
pkgpyexec_LTLIBRARIES = _dereverberation.la

_dereverberation_la_SOURCES = dereverberation_wrap.cxx
_dereverberation_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_dereverberation_la_LIBADD = libbtk_dereverberation.la
_dereverberation_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = dereverberation_wrap.cxx dereverberation.py
EXTRA_DIST = dereverberation.i

SWIG_FILES = dereverberation_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include 

include $(SWIG_RULES)
