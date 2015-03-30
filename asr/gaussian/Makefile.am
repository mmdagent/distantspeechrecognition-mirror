AM_CPPFLAGS = $(BTK_CFLAGS) $(GSL_CFLAGS) $(OPENCV_CFLAGS)

lib_LTLIBRARIES = libasr_gaussian.la
libasr_gaussian_la_LDFLAGS = $(BTK_LIBS) $(GSL_LIBS) -lbtk_feature
libasr_gaussian_la_SOURCES = \
	codebookBasic.cc distribBasic.cc

asrgaussian_includedir = $(pkgincludedir)/gaussian
asrgaussian_include_HEADERS = \
	codebookBasic.h distribBasic.h

pkgpython_PYTHON = gaussian.py
pkgpyexec_LTLIBRARIES = _gaussian.la

_gaussian_la_SOURCES = gaussian_wrap.cxx
_gaussian_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_gaussian_la_LIBADD = libasr_gaussian.la
_gaussian_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = gaussian_wrap.cxx gaussian.py
EXTRA_DIST = gaussian.i

SWIG_FILES = gaussian_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = $(AM_CPPFLAGS)

include $(SWIG_RULES)
