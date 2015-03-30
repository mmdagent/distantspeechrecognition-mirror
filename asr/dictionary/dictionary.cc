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


#include "dictionary/dictionary.h"
#include "common/refcount.h"
#include "common/jexception.h"
#include "common/mlist.h"


// ----- methods for class `Dictword' -----
//
String Dictword::puts() {
  String repr("{" + _name + "} "+ *(_word->puts()));
  return repr;
}

// ----- methods for class `Dictionary' -----
//
// access members of the dictionary
DictwordPtr Dictionary::operator[](const String& key) {
  if (has_word(key)) { 
    return _dictwords[key];
  } else {
    throw jkey_error("The word '%s' is not in this dictionary",
                     key.c_str());
  }
}
                        
// displys the first count items of the dictionary
String Dictionary::puts(int count) {
  String repr;
#ifdef FixThis
  int i = count;
  for (_Iterator iter(_dictwords); iter.more(); iter++) {
    *repr = *repr + *(iter->puts()) + "\n";
    if (i-- <= 0) break;
  }
#endif
  return repr;
}
 
// returns true if name is in the dictionary
bool Dictionary::has_word(const String& name) {
  return ( _dictwords.isPresent(name));
}

// add a new word to the set
void Dictionary::add(const String& n, WordPtr w) {

  // cout << "dictionary::add -> " << n << endl;

  if (w->phones() != _phones || w->tags() != _tags) 
    throw j_error("Word is defined over %s/%s and not over %s/%s.\n",
                  w->phones()->name().c_str(),
                  w->tags()->name().c_str(),
                  _phones->name().c_str(),
                  _tags->name().c_str());
  if (has_word(n))
    throw j_error("DictWord '%s' already exists in '%s'.",
                  n.c_str(), _name.c_str());

  DictwordPtr dictword(new Dictword(n, w));

  int cP = n.find('(');
  if (cP != -1 && cP != 0) {
#ifdef JMcD
    String basename = n.substr(0, cP-1);
#else
    static char bname[1000];
    strncpy(bname, n.chars(), cP); bname[cP] = '\0';
    String basename(bname);
#endif
    if (! has_word(basename)) {
      Warning("Variant '%s' occured before base form %s.\n",
	      n.chars(), basename.chars());
    } else {
      dictword = _dictwords[basename]->_nextVariant;
      _dictwords[basename]->_nextVariant = dictword;
    }
  }
  _dictwords.add(n, dictword);
}

void Dictionary::add(const String& name, const String& pronunciation) {
  WordPtr word(new Word(_phones, _tags));
  word->get(pronunciation);
  add(name, word);
}
 
// remove word from the set
void Dictionary::erase(const String& n) {
#ifdef JMcD
  int cP;

  if (! has_word(n))
    throw j_error("Can not delete '%s'. Not in dictionary.", n.c_str());

  if (! (cP = n.find('('))) {
    /* try to delete baseform */
    if ( _dictwords[n]->_nextVariant != NULL ) {
      throw j_error("Can not delete '%s' before all variants are deleted.", n.c_str());
    }
  } else {
    /* find baseform */
    DictwordPtr base = _dictwords[n.substr(cP-1)];
    DictwordPtr variant = _dictwords[n];
    
    /* find predecessor of idx */
    while (base->_nextVariant != variant) {
      if (base->_nextVariant == NULL) {
        throw j_error("Can not delete variant '%s'. Variant list is corrupted.", n.c_str());
      }
      base = base->_nextVariant;
    }

    /* remove variant from the chain */
    base->_nextVariant = variant->_nextVariant;
  }
  _dictwords.erase(n);
#endif
}
 
// reads a dictionary file
void Dictionary::read(const String& filename) {
  freadAdd(filename, _commentChar, this);
}

// writes a dictionary file
void Dictionary::write(const String& filename) {
  FILE* fp = fileOpen(filename, "w");

  if (fp == NULL)
    throw jio_error("Can not open '%s' for writing!", filename.c_str());

  fprintf(fp, "%c -------------------------------------------------------\n",
	  _commentChar);
  fprintf(fp, "%c  Name            : %s\n", _commentChar, _name.c_str());
  fprintf(fp, "%c  Type            : Dictionary\n", _commentChar);
  fprintf(fp, "%c  Number of Items : %d\n", _commentChar, _dictwords.size());
  fprintf(fp, "%c  Date            : %s", _commentChar, dateString());
  fprintf(fp, "%c -------------------------------------------------------\n",
	  _commentChar);

#ifdef FixThis
  for (_ConstIterator iter(_dictwords); iter.more(); iter++) {
    fprintf(fp,"{%s} {%s}\n", (*iter)->name().c_str(), (*iter)->puts()->c_str());
  }
#endif

  fileClose( filename, fp);
}
 
/* freadAdd calls this method */
void Dictionary::__add(const String& s) {
  list<String> line;
  splitList(s, line);
  if (line.size() != 2) {
    throw j_error("line has wrong format");
  }
  add(line.front(), line.back()); 
}

// init start and end tags
void Dictionary::tagsInit() {
  int idx;

  _wbTags = ((idx = _tags->index("WB")) > -1) ? 1<<idx : 0;
  _weTags = ((idx = _tags->index("WE")) > -1) ? 1<<idx :
    (((idx = _tags->index("WB")) > -1) ? 1<<idx : 0);
  _xwTags =  0;

  for (int idx = 0; idx < _tags->size(); idx++) {
    int   tag = 1 << idx;
    if (!(tag & (_wbTags | _weTags))) _xwTags |= tag;
  }
}
