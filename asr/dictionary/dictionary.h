//
//                               Millenium
//                   Automatic Speech Recognition System
//                                 (asr)
//
//  Module:  asr.dict
//  Purpose: Dictionary module.
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


#ifndef _dictionary_h_
#define _dictionary_h_

#include "common/refcount.h"
#include "dictionary/words.h"

// Word with tagged phone transcription
class Dictword;
typedef refcount_ptr<Dictword> DictwordPtr;
class Dictword {
 public:
  Dictword(): _name("(null)"), _word() {};
  Dictword(const String& n): _name(n), _word() {};
  Dictword(const String& n, WordPtr w): _name(n), _word(w) {};

  String	name() const { return _name; }
  String 	puts();

  friend class Dictionary;

 protected:
  String	_name;
  WordPtr	_word;
  DictwordPtr	_nextVariant;
};

/* Set of words */
/* Implementation Note:
 * Dictionary manages a map of words including variants
 * variants have the following repreasentaion:
 *  word(1)
 *  word(2) ...
 * the member nextVariant of Dictword contains the next
 * variant of this word.
*/
class Dictionary {
  typedef List<DictwordPtr>		_DictwordList;
  typedef _DictwordList::Iterator	_Iterator;
  typedef _DictwordList::ConstIterator	_ConstIterator;
 public:
  Dictionary() :
    _dictwords("Dictionary Words"), _phones(), _tags(), _commentChar(';') { }

  Dictionary(PhonesPtr p, TagsPtr t) :
    _dictwords("Dictionary Words"), _phones(p), _tags(t), _commentChar(';') { tagsInit(); }

  Dictionary(const String& n, PhonesPtr p, TagsPtr t) :
    _name(n), _dictwords("Dictionary Words"), _phones(p), _tags(t), _commentChar(';') { tagsInit(); }

  // access members of the dictionary
  DictwordPtr operator[](const String& key);
    
  // displys the first count items of the dictionary
  String puts(int count=20);
    
  // returns true if name is in the dictionary
  bool has_word(const String& name); 

  // add a new word to the set
  void add(const String& n, WordPtr w);
  void add(const String& n, const String& pronunciation);
    
  // remove word from the set
  void erase(const String& n);
    
  // reads a dictionary file
  void read(const String& filename);
    
  // writes a dictionary file
  void write(const String& filename);
    
  // return the spelled word given the index
  String& name() { return _name; };
    
  // freadAdd calls this method
  void __add(const String& s);

 protected:
  String	_name;

  _DictwordList	_dictwords;
  PhonesPtr	_phones;
  TagsPtr	_tags;
  
  char		_commentChar;
 
  int		_wbTags;    // word begin tag
  int		_weTags;    // word end tag
  int		_xwTags;    // other xword tags to test

  void tagsInit();
};

typedef refcount_ptr<Dictionary> DictionaryPtr;

#endif
