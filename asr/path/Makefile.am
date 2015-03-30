AM_CPPFLAGS = $(BTK_CFLAGS)

lib_LTLIBRARIES = libasr_path.la
libasr_path_la_LDFLAGS = $(BTK_LIBS)
libasr_path_la_LIBADD = $(top_srcdir)/gaussian/libasr_gaussian.la
libasr_path_la_SOURCES = \
	codebookPath.cc  distribPath.cc

asrpath_includedir = $(pkgincludedir)/path
asrpath_include_HEADERS = \
	codebookPath.h  distribPath.h

pkgpython_PYTHON = path.py
pkgpyexec_LTLIBRARIES = _path.la

_path_la_SOURCES = path_wrap.cxx
_path_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_path_la_LIBADD = libasr_path.la
_path_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = path_wrap.cxx path.py
EXTRA_DIST = path.i

SWIG_FILES = path_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = $(AM_CPPFLAGS)

include $(SWIG_RULES)
