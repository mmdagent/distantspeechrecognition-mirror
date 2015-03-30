AM_CPPFLAGS = $(PYTHON_CPPFLAGS) $(BTK_CFLAGS)

#noinst_PROGRAMS = test test2 test3 
#test_SOURCES = test.cc
#test_LDADD = libasr_decoder.la

#test2_SOURCES = test2.cc
#test2_LDADD = libasr_decoder.la

#test3_SOURCES = test3.cc
#test3_LDADD = libasr_decoder.la

lib_LTLIBRARIES = libasr_decoder.la
libasr_decoder_la_LDFLAGS = $(BTK_LIBS)
libasr_decoder_la_LIBADD = $(top_srcdir)/lattice/libasr_lattice.la ../path/libasr_path.la
libasr_decoder_la_SOURCES = decoder.cc wfstFlyWeight.cc

asrdecoder_includedir = $(pkgincludedir)/decoder
asrdecoder_include_HEADERS = decoder.h wfstFlyWeight.h

pkgpython_PYTHON = decoder.py
pkgpyexec_LTLIBRARIES = _decoder.la

_decoder_la_SOURCES = decoder_wrap.cxx
_decoder_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(AM_CPPFLAGS)
_decoder_la_LIBADD = libasr_decoder.la
_decoder_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(SWIG_PYTHON_LIBS)

CLEANFILES = decoder_wrap.cxx decoder.py
EXTRA_DIST = decoder.i

SWIG_FILES = decoder_wrap.cxx
SWIG_OPT = $(SWIG_PYTHON_OPT)
SWIG_CPP = $(AM_CPPFLAGS)

include $(SWIG_RULES)
