//                              -*- C++ -*-
//
//                               Millenium
//                   Automatic Speech Recognition System
//                                 (asr)
//
//  Module:  asr.dict
//  Purpose: Dictionary module.
//  Author:  Fabian Jakobs.

#ifndef _word_h_
#define _word_h_

#include "common/refcount.h"
#include "dictionary/tags.h"
#include "dictionary/phones.h"
#include <vector>

// Word with tagged phone transcription
class Word {
 public:
  Word() : _phones(new Phones()), _tags(new Tags()) { }

  Word(PhonesPtr p, TagsPtr t) :
    _phones(p), _tags(t) { }

  String puts();
  void get(const String&);

  PhonesPtr phones()  const { return _phones; }
  TagsPtr   tags()    const { return _tags; }
  bool      hasTags() const { return _tagMap.size() != 0; }

  void check(PhonesPtr PS, TagsPtr TP);

  int tag(unsigned i)   const { assert(i < _tagMap.size()); return _tagMap[i]; }
  int phone(unsigned i) const { assert(i < _phnIdx.size()); return _phnIdx[i]; }

  void appendTag(unsigned i) { _tagMap.push_back(i); }
  void appendPhone(unsigned i) { _phnIdx.push_back(i); }

 private:
  PhonesPtr		_phones;
  TagsPtr		_tags;
  vector<int>		_tagMap;
  vector<int>		_phnIdx;
};

typedef refcount_ptr<Word> WordPtr;

#endif
