//
//                                Millenium
//                   Automatic Speech Recognition System
//                                  (asr)
//
//  Module:  asr.foo
//  Purpose: Sample module.
//  Author:  John McDonough

#ifndef _foo_h_
#define _foo_h_

#include "common/refcount.h"
#include "common/mlist.h"

class Bar {
  public:
    Bar(const String& name): _name(name) {}
    String puts();
  private:
    String _name;
};
typedef refcount_ptr<Bar> BarPtr;

#endif
