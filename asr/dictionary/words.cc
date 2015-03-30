//
//                               Millenium
//                   Automatic Speech Recognition System
//                                 (asr)
//
//  Module:  asr.dict
//  Purpose: Dictionary module.
//  Author:  Fabian Jakobs.

#include "dictionary/words.h"
#include "dictionary/phones.h"
#include "common/refcount.h"
#include "common/mlist.h"
#include <string>
#include <list>


/* ========================================================================
    Word
   ======================================================================== */

String Word::puts() {
  int i = 0;
  String repr(" {");

#ifdef FixThis
  for (vector<int>::iterator iter = _phnIdx.begin(); iter != _phnIdx.end(); ++iter) {
    if (_tagMap[i]) {
      *repr = *repr + " {";
    }
#ifdef JMcD
    *repr = *repr + " " + *(*iter);
#else
    *repr = *repr + " " + *((*_phones)[*iter]);
#endif
    if (_tagMap[i]) {
      int t = _tagMap[i];
      for (unsigned j = 0; j < 8*sizeof(int); j++) {
        if ( t & (1 << j)) {
          *repr = *repr + " " + (*_tags)[j];
        }
      }
      *repr = *repr + "}";
    }
    i++;
  }
  *repr = *repr +"}";
#endif

  return repr;
}

void Word::check(PhonesPtr PS, TagsPtr TP)
{
  if ( _phones != PS)
    throw j_error("Phones do not match.\n");

  if ( _tags != TP)
    throw j_error("Tags do not match.\n");
}
 
void Word::get(const String& line) {
  typedef list<String>::iterator Iterator;
  list<String> symbols;
  list<String> token;

  int map = 0;
  splitList(line, symbols);
  for (Iterator iter1 = symbols.begin(); iter1 != symbols.end(); ++iter1) {
    token.clear();
    splitList(*iter1, token);
    if (token.size() < 1)
      throw jparse_error("Missing phone name.");

    String s = token.front();
    token.pop_front();

    map = 0;
    if ( !_phones->hasPhone(s) )
      throw jparse_error("Can't find phone '%s'!", s.c_str());

    _phnIdx.push_back(_phones->index(s));

    for (Iterator iter2 = token.begin(); iter2 != token.end(); ++iter2) {
      int t = _tags->index(*iter2);
      if ( t < 0)
        throw jparse_error("Can't find tag '%s'!", iter2->c_str());

      map |= (1 << t);
    }
    _tagMap.push_back(map);
  }
}
