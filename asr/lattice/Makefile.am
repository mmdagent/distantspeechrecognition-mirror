AM_CPPFLAGS = $(BTK_CFLAGS)

lib_LTLIBRARIES = libasr_lattice.la
libasr_lattice_la_LDFLAGS = $(BTK_LIBS)
libasr_lattice_la_LIBADD = $(top_srcdir)/gaussian/libasr_gaussian.la ../fsm/libasr_fsm.la
libasr_lattice_la_SOURCES = lattice.cc

asrlattice_includedir = $(pkgincludedir)/lattice
asrlattice_include_HEADERS = lattice.h

pkgpython_PYTHON = lattice.py
pkgpyexec_LTLIBRARIES = _lattice.la

_lattice_la_SOURCES = lattice_wrap.cxx
_lattice_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_lattice_la_LIBADD = libasr_lattice.la
_lattice_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = lattice_wrap.cxx lattice.py
EXTRA_DIST = lattice.i

SWIG_FILES = lattice_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = $(AM_CPPFLAGS)

include $(SWIG_RULES)
