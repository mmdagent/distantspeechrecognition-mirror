AM_CPPFLAGS = $(CPPFLAGS) 
lib_LTLIBRARIES = libbtk_objectiveMeasure.la
libbtk_objectiveMeasure_la_SOURCES = objectiveMeasure.cc 
libbtk_objectiveMeasure_la_LIBADD  = $(top_srcdir)/stream/libbtk_stream.la $(top_srcdir)/feature/libbtk_feature.la
libbtk_objectiveMeasure_la_LDFLAGS = $(GSL_LIBS) 

btk_objectiveMeasure_includedir = $(pkgincludedir)/objectiveMeasure
btk_objectiveMeasure_include_HEADERS = objectiveMeasure.h

pkgpython_PYTHON = objectiveMeasure.py
pkgpyexec_LTLIBRARIES = _objectiveMeasure.la

_objectiveMeasure_la_SOURCES = objectiveMeasure_wrap.cxx
_objectiveMeasure_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS) 
_objectiveMeasure_la_LIBADD = libbtk_objectiveMeasure.la
_objectiveMeasure_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = objectiveMeasure_wrap.cxx objectiveMeasure.py
EXTRA_DIST = objectiveMeasure.i

SWIG_FILES = objectiveMeasure_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = -I$(top_srcdir)/include $(AM_CPPFLAGS)

include $(SWIG_RULES)
