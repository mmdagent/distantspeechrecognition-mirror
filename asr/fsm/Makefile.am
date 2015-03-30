AM_CPPFLAGS = $(BTK_CFLAGS)

lib_LTLIBRARIES = libasr_fsm.la
libasr_fsm_la_LDFLAGS = $(BTK_LIBS)
libasr_fsm_la_LIBADD = $(top_srcdir)/dictionary/libasr_dict.la
libasr_fsm_la_SOURCES = fsm.cc skhash.cc

asrfsm_includedir = $(pkgincludedir)/fsm
asrfsm_include_HEADERS = fsm.h skhash.h

pkgpython_PYTHON = fsm.py
pkgpyexec_LTLIBRARIES = _fsm.la

_fsm_la_SOURCES = fsm_wrap.cxx
_fsm_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_fsm_la_LIBADD = libasr_fsm.la
_fsm_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = fsm_wrap.cxx fsm.py
EXTRA_DIST = fsm.i

SWIG_FILES = fsm_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = $(AM_CPPFLAGS)

include $(SWIG_RULES)
