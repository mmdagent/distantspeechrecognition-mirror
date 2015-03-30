AM_CPPFLAGS = $(BTK_CFLAGS)

lib_LTLIBRARIES = libasr_natural.la
libasr_natural_la_LDFLAGS = $(BTK_LIBS)
libasr_natural_la_SOURCES = natural.cc

asrnatural_includedir = $(pkgincludedir)/natural
asrnatural_include_HEADERS = natural.h

pkgpython_PYTHON = natural.py
pkgpyexec_LTLIBRARIES = _natural.la

_natural_la_SOURCES = natural_wrap.cxx
_natural_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_natural_la_LIBADD = libasr_natural.la
_natural_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = natural_wrap.cxx natural.py
EXTRA_DIST = natural.i

SWIG_FILES = natural_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = $(AM_CPPFLAGS)

include $(SWIG_RULES)
