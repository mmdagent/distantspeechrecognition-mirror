//
//                                Millenium
//                   Automatic Speech Recognition System
//                                  (asr)
//
//  Module:  asr.foo
//  Purpose: Sample module.
//  Author:  John McDonough

%module foo

#ifdef AUTODOC
%section "Foo"
#endif

/* operator overloading */
%rename(__str__) *::puts();

%include jexception.i
%include typedefs.i

%{
#include "foo/foo.h"
%}

// ----- definition for class `Bar' ----- 
// 
%ignore Bar;
class Bar {
 public:
  ~Bar();

  // String puts();
};

class BarPtr {
 public:
  %extend {
    BarPtr(const String name) {
      return new BarPtr(new Bar(name));
    }
  }
  Bar* operator->();
};

