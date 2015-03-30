AM_CPPFLAGS = $(CPPFLAGS)
lib_LTLIBRARIES = libbtk_shapeFactorEstimation.la
libbtk_shapeFactorEstimation_la_SOURCES = \
	shapeFactorEstimation.cc
libbtk_shapeFactorEstimation_la_LIBADD  = $(top_srcdir)/stream/libbtk_stream.la
libbtk_shapeFactorEstimation_la_LDFLAGS = $(GSL_LIBS) 

btk_shapeFactorEstimation_includedir = $(pkgincludedir)/shapeFactorEstimation
btk_shapeFactorEstimation_include_HEADERS = \
	shapeFactorEstimation.h

pkgpython_PYTHON = shapeFactorEstimation.py
pkgpyexec_LTLIBRARIES = _shapeFactorEstimation.la

_shapeFactorEstimation_la_SOURCES = shapeFactorEstimation_wrap.cxx
_shapeFactorEstimation_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_shapeFactorEstimation_la_LIBADD = libbtk_shapeFactorEstimation.la
_shapeFactorEstimation_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = shapeFactorEstimation_wrap.cxx shapeFactorEstimation.py
EXTRA_DIST = shapeFactorEstimation.i

SWIG_FILES = shapeFactorEstimation_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include $(AM_CPPFLAGS)

include $(SWIG_RULES)
