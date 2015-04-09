FEATURE_FLAGS = $(GSL_CFLAGS) $(SNDFILE_CFLAGS) $(SAMPLERATE_CFLAGS) $(JACK_CFLAGS) $(OPENCV_CFLAGS) -I$(top_srcdir)/gnulib
AM_CPPFLAGS = \
	$(CPPFLAGS) $(FEATURE_FLAGS)
lib_LTLIBRARIES = libbtk_portnoff.la
libbtk_portnoff_la_SOURCES = portnoff.cc
libbtk_portnoff_la_LIBADD  = $(top_srcdir)/stream/libbtk_stream.la $(top_srcdir)/feature/libbtk_feature.la
libbtk_portnoff_la_LDFLAGS = $(GSL_LIBS)

libbtkportnoff_la_includedir = $(pkgincludedir)/portnoff
libbtkportnoff_la_include_HEADERS = portnoff.h

pkgpython_PYTHON = portnoff.py
pkgpyexec_LTLIBRARIES = _portnoff.la

_portnoff_la_SOURCES = portnoff_wrap.cxx
_portnoff_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_portnoff_la_LIBADD = libbtk_portnoff.la
_portnoff_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = portnoff_wrap.cxx portnoff.py
EXTRA_DIST = portnoff.i

SWIG_FILES = portnoff_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include $(FEATURE_FLAGS)

include $(SWIG_RULES)
