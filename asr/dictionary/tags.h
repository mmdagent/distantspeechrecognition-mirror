//                              -*- C++ -*-
//
//                               Millenium
//                   Automatic Speech Recognition System
//                                 (asr)
//
//  Module:  asr.dict
//  Purpose: Dictionary module.
//  Author:  Fabian Jakobs.

#ifndef _tags_h_
#define _tags_h_

#include "common/refcount.h"
#include "common/mlist.h"
#include <vector>
#include <stdio.h>

/* A 'Tags' object is an array of strings. */
class Tags {
  typedef vector<String> TagsList;
  typedef TagsList::iterator TagsListIter;
  typedef TagsList::const_iterator TagsListConstIter;

 public:
  Tags():
    _name("(null)"),
    wordBeginTag("WB"),
    wordEndTag("WE"),
    commentChar(';'),
    modMask(0 ^ -1) {};
  Tags(const String& n):
    _name(n),
    wordBeginTag("WB"),
    wordEndTag("WE"),
    commentChar(';'),
    modMask(0 ^ -1) {};

  /* displays the contents of a tags-set */
  String puts();
  
  /* add new tag(s) to a tags-set */
  void add(const String& tag);
  
  /* delete tag(s) from a tags-set */
  void erase(const String& tag);
  
  /* returns the index of tag s */
  signed int index(const String& s);
  
  /* read a tag-set from a file */
  int read(const String& filename);
    
  /* write a set of tags into a open file */
  void writeFile(FILE* fp);

  /* write a tag-set into a file */
  void write(const String& filename);
    
  /* return the name of indexed tag(s) */
  String& name() { return _name; };

  /* index operator to access the tags */
  String& operator[](unsigned index) { return _tags[index]; }

  /* returns number of tags */ 
  int size() { return _tags.size(); }

  bool hasTag(const String& ps) const;

 protected:
  String _name;
  String wordBeginTag;
  String wordEndTag;
  char commentChar;
  int modMask;

  TagsList _tags;
};

typedef refcount_ptr<Tags> TagsPtr;

#endif
