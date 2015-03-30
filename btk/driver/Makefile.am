AM_CPPFLAGS = $(JACK_CFLAGS) $(SNDFILE_CFLAGS) -I$(top_srcdir)/gnulib

lib_LTLIBRARIES = libbtk_driver.la
libbtk_driver_la_SOURCES = databuffers.cc  mk4array.cc  mk4cursor.cc  mk4error.cc  mk4lib.cc  mk4_misc.cc  mk4msg.cc
libbtk_driver_la_LIBADD = $(JACK_LIBS) $(SNDFILE_LIBS)

btkdriver_includedir = $(pkgincludedir)/driver
btkdriver_include_HEADERS = mk4lib.h mk4array.h mk4_common.h mk4cursor.h mk4cursor_type.h mk4error.h mk4lib_int.h mk4_misc.h mk4msg.h mk4msg_type.h

pkgpython_PYTHON = driver.py
pkgpyexec_LTLIBRARIES = _driver.la

_driver_la_SOURCES = driver_wrap.cxx
_driver_la_LIBADD = libbtk_driver.la
_driver_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_driver_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS) $(PYTHON_LSPEC)

CLEANFILES = driver_wrap.cxx driver.py

SWIG_FILES = driver_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include $(AM_CPPFLAGS)

include $(SWIG_RULES)
