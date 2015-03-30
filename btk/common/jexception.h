//                              -*- C++ -*-
//
//                           Speech Front End
//                                 (sfe)
//
//  Module:  sfe.common
//  Purpose: Common operations.
//  Author:  Fabian Jakobs
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or (at
// your option) any later version.
// 
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

#ifndef _jexception_h_
#define _jexception_h_

#include <exception>
#include <string>
#include <stdio.h>
#include <stdarg.h>
#include <malloc.h>
#include <limits.h>

using namespace std;

#define BUILDMSG(what_arg) va_list ap; \
  va_start(ap, what_arg); \
  va_start(ap, what_arg); \
  _what = make_msg(what_arg, ap);

typedef enum {
  JERROR,
  JALLOCATION,
  JARITHMETIC,
  JCONSISTENCY,
  JDIMENSION,
  JINDEX,
  JINITIALIZATION,
  JIO,
  JITERATOR,
  JPYTHON,
  JKEY,
  JNUMERIC,
  JPARAMETER,
  JPARSE,
  JTYPE
} error_type;

class j_error : public
 exception {
 public:
  j_error() throw(): _what(""), code(JERROR)  {};
  j_error(const char* what_arg, ...) throw();
  ~j_error() throw();
  virtual const char* what () const throw() { return _what.c_str (); }
  error_type getCode() { return code; }
  void print_trace (void);
 protected:
  string _what;
  error_type code;
  const char* make_msg(const char* what_arg, va_list& ap);
};

class jallocation_error : public j_error {
 public:
  jallocation_error(const char* what_arg, ...) { 
    BUILDMSG(what_arg); 
    code = JALLOCATION;
  }
};

class jarithmetic_error : public j_error {
 public:
  jarithmetic_error(const char* what_arg, ...) { 
    BUILDMSG(what_arg); 
    code = JARITHMETIC;
  }
};

class jconsistency_error : public j_error {
 public:
  jconsistency_error(const char* what_arg, ...) { 
    BUILDMSG(what_arg); 
    code = JCONSISTENCY;
  }
};

class jdimension_error : public j_error {
 public:
  jdimension_error(const char* what_arg, ...) { 
    BUILDMSG(what_arg); 
    code = JDIMENSION;
  }
};

class jindex_error : public j_error {
 public:
  jindex_error(const char* what_arg, ...) {
    BUILDMSG(what_arg);
    code = JINDEX;
  }
};

class jinitialization_error : public j_error {
 public:
  jinitialization_error(const char* what_arg, ...) { 
    BUILDMSG(what_arg); 
    code = JINITIALIZATION;
  }
};

class jio_error : public j_error {
 public:
  jio_error(const char* what_arg, ...) {
    BUILDMSG(what_arg); 
    code = JIO;
  }
};

class jiterator_error : public j_error {
 public:
  jiterator_error(const char* what_arg, ...);
};

class jkey_error : public j_error {
 public:
  jkey_error(const char* what_arg, ...) { 
    BUILDMSG(what_arg); 
    code = JKEY;
  }
};

class jnumeric_error : public j_error {
 public:
  jnumeric_error(const char* what_arg, ...) { 
    BUILDMSG(what_arg); 
    code = JNUMERIC;
  }
};

class jparameter_error : public j_error {
 public:
  jparameter_error(const char* what_arg, ...) { 
    BUILDMSG(what_arg); 
    code = JPARAMETER;
  }
};

class jparse_error : public j_error {
 public:
  jparse_error(const char* what_arg, ...) { 
    BUILDMSG(what_arg); 
    code = JPARSE;
  }
};

class jtype_error : public j_error {
 public:
  jtype_error(const char* what_arg, ...) { 
    BUILDMSG(what_arg); 
    code = JTYPE;
  }
};

#endif
