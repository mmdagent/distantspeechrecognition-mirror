FEATURE_FLAGS = $(GSL_CFLAGS) $(SNDFILE_CFLAGS) $(SAMPLERATE_CFLAGS) $(JACK_CFLAGS) $(OPENCV_CFLAGS) -I$(top_srcdir)/gnulib
AM_CPPFLAGS = $(CPPFLAGS) $(FEATURE_FLAGS)
lib_LTLIBRARIES = libbtk_TDEstimator.la
libbtk_TDEstimator_la_SOURCES = CCTDE.cc 
libbtk_TDEstimator_la_LIBADD  = $(top_srcdir)/stream/libbtk_stream.la $(top_srcdir)/feature/libbtk_feature.la $(top_srcdir)/modulated/libbtk_modulated.la
libbtk_TDEstimator_la_LDFLAGS = $(GSL_LIBS)

btk_TDEstimator_includedir = $(pkgincludedir)/TDEstimator
btk_TDEstimator_include_HEADERS = CCTDE.h

pkgpython_PYTHON = TDEstimator.py tdoa.py
pkgpyexec_LTLIBRARIES = _TDEstimator.la

_TDEstimator_la_SOURCES = TDEstimator_wrap.cxx
_TDEstimator_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS) 
_TDEstimator_la_LIBADD = libbtk_TDEstimator.la
_TDEstimator_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = TDEstimator_wrap.cxx TDEstimator.py
EXTRA_DIST = TDEstimator.i

SWIG_FILES = TDEstimator_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include $(FEATURE_FLAGS)

include $(SWIG_RULES)
