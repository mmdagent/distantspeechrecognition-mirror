AM_CPPFLAGS = $(BTK_CFLAGS)

lib_LTLIBRARIES = libasr_dict.la
libasr_dict_la_LDFLAGS = $(BTK_LIBS)
libasr_dict_la_SOURCES = \
	dictionary.cc distribTree.cc \
	phones.cc tags.cc words.cc

asrdictionary_includedir = $(pkgincludedir)/dictionary
asrdictionary_include_HEADERS = \
	dictionary.h distribTree.h \
	phones.h tags.h words.h

pkgpython_PYTHON = dict.py
pkgpyexec_LTLIBRARIES = _dict.la

_dict_la_SOURCES = dict_wrap.cxx
_dict_la_LIBADD = libasr_dict.la
_dict_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_dict_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = dict_wrap.cxx dict.py
EXTRA_DIST = dict.i

SWIG_FILES = dict_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = $(AM_CPPFLAGS)

include $(SWIG_RULES)
