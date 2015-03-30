//                              -*- C++ -*-
//
//                               Millennium
//                   Distant Speech Recognition System
//                                 (dsr)
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


%module dict

#ifdef AUTODOC
%section "Dictionary"
#endif

/* operator overloading */
%rename(__str__) *::puts();
%rename(__getitem__) *::operator[];

%rename(Phones_Iterator) Phones::Iterator;
%rename(DistribTree_Node) DistribTree::Node;
%rename(DistribTree_NodePtr) DistribTree::NodePtr;
%rename(DistribTree_Iterator) DistribTree::Iterator;

%include jexception.i
%include typedefs.i

#%module phones

%{
#include "phones.h"
%}

#ifdef AUTODOC
%subsection "Phones", before 
#endif

%ignore Phones;
class Phones {
 public:
  ~Phones();

  // prints the contents of a set of phone-sets
  String puts();

  // add new phone(s) to a phone-set
  void add(const String phone);

  // returns true if phone is in this class
  bool hasPhone(const String& phone);

  // read a phone-set from a file
  void read(const String filename);

  // write a phone-set from a file
  void write(const String filename);
};

class Phones::Iterator {
 public:
  Phones::Iterator(PhonesPtr& phones);

  void operator++(int);
  bool more() const;
  const String& phone() const;
  const String& next();
};

class PhonesPtr {
 public:
  %extend {
    PhonesPtr() { return new PhonesPtr(new Phones()); }

    // return an iterator
    Phones::Iterator* __iter__() {
      return new Phones::Iterator(*self);
    }
  }

  Phones* operator->();
};


%ignore PhonesSet;
class PhonesSet {
 public:
  ~PhonesSet();

  // displays the contents of a set of phone-sets
  String puts();

  // find a given phone
  PhonesPtr find(const String key) const;

  // add new phone-set to a set of phones-set
  // void add(const String phoneName, const list<String>& phones);
  void add(const String phoneName, PhonesPtr phones);

  // delete phone-set(s) from a set of phone-sets
  void remove(const String phone);

  // read a set of phone-sets from a file
  void read(const String filename);

  // write a set of phone-sets into a file
  void write(const String filename);

  // name of the phone set
  const String name();
};

class PhonesSetPtr {
 public:
  %extend {
    PhonesSetPtr(const String nm = "") {
      return new PhonesSetPtr(new PhonesSet(nm));
    }

    // return a 'Phones' object
    PhonesPtr __getitem__(const String name) {
      return (*self)->find(name);
    }
  }

  PhonesSet* operator->();
};

%{
#include "distribTree.h"
%}


// ----- definition for class `Lexicon' -----
// 
%rename(Lexicon_Iterator)		Lexicon::Iterator;

%ignore Lexicon;
class Lexicon {
 public:
  Lexicon(const String nm, const String fileName = "");
  ~Lexicon();

  void clear();
  String name() const;
  unsigned size() const;

  void read(const String& fileName);
  void write(const String& fileName, bool writeHeader = false /*, WFSAcceptorPtr wfsa = NULL */) const;

  unsigned index(const String& symbol, bool create = false);
  const String symbol(unsigned idx) const;
  bool isPresent(const String& symbol) const;
};

class LexiconPtr {
 public:
  %extend {
    LexiconPtr(const String nm, const String fileName = "") {
      return new LexiconPtr(new Lexicon(nm, fileName));
    }

    // return an iterator
    Lexicon::Iterator* __iter__() {
      return new Lexicon::Iterator(*self);
    }

    // return the index for an entry
    unsigned  __getitem__(const String name) {
      return (*self)->index(name);
    }
  }
  Lexicon* operator->();
};

class Lexicon::Iterator {
 public:
  Lexicon::Iterator(LexiconPtr& lex):
    _lexicon(lex), _iter(_lexicon->_indexMap.begin()) {}

  String name() const;
  bool more();
  String next();

 private:
  LexiconPtr		_lexicon;
  _IndexMapIterator	_iter;
};


// ----- definition for class `DistribTree' -----
// 
%ignore DistribTree;
class DistribTree {
 public:
  DistribTree(const String& nm, PhonesSetPtr& phonesSet, const int contextLength = 1, const String& fileName = "",
	      char comment = ';', const String& wb = "WB", const String& eos = "</s>", bool verbose = false);
  ~DistribTree();

  const String begin(const String context);
  const String middle(const String context);
  const String end(const String context);

  void buildMatrices(LexiconPtr& phonesLexicon);

  void read(const String& treePath, const String& state = "", unsigned offset = 0, bool sphinx = false);
  void write(const String fileName, const String& date = "");

  const String& name();
};

%ignore DistribTree::Node;
class DistribTree::Node {
public:
  Node(PhonesSetPtr& ps, DistribTree* dt,
       const String& nm, const String& ques,
       const String& left, const String& right, const String& unknown, const String& leaf);

  const String& name() const;

  const String& leaf(const String& context) const;
  const String& left()    const;
  const String& right()   const;
  const String& leaf()    const;
  const String& unknown() const;
};

class DistribTree::NodePtr {
 public:

  // this should not be necessary!
  %extend {
    const String& name() const { return (*self)->name(); }

    const String& leaf(const String& context) const { return (*self)->leaf(context); }
    const String& left()    const { return (*self)->left(); }
    const String& right()   const { return (*self)->right(); }
    const String& leaf()    const { return (*self)->leaf(); }
    const String& unknown() const { return (*self)->unknown(); }
    
  }

  DistribTree::Node* operator->();
};

class DistribTree::Iterator {
 public:
  DistribTree::Iterator(DistribTreePtr& tree);

  bool more() const;
  const DistribTree::NodePtr node() const;
  DistribTree::NodePtr next();
};

class DistribTreePtr {
 public:
  %extend {
    DistribTreePtr(const String& nm, PhonesSetPtr& phonesSet, const int contextLength = 1, const String& fileName = "", char comment = ';',
                   const String& wb = "WB", const String& eos = "</s>", bool verbose = false) {
      return new DistribTreePtr(new DistribTree(nm, phonesSet, contextLength, fileName, comment, wb, eos, verbose));
    }

    // return an iterator
    DistribTree::Iterator* __iter__() {
      return new DistribTree::Iterator(*self);
    }

    // return a codebook
    DistribTree::NodePtr __getitem__(const String& name) {
      return (*self)->node(name);
    }
  }
  DistribTree* operator->();
};

%{
#include "tags.h"
%}

#ifdef AUTODOC
%subsection "Tags", before
#endif

%ignore Tags;
class Tags {
  public:
  ~Tags();

  // displays the contents of a tags-set
  StringPtr puts();

  // add new tag(s) to a tags-set
  void add(const String tag);

  // delete tag(s) from a tags-set
  void erase(const String tag);

  // read a tag-set from a file
  void read(const String filename);

  // write a tag-set into a file
  void write(const String filename);

  // return the name of indexed tag(s)
  String name();
};

class TagsPtr {
 public:
  %extend {
    Tags() { return new TagsPtr(new Tags()); }
  }

  Tags* operator->();
};


%{
#include "dictionary.h"
%}

#ifdef AUTODOC
%subsection "Dictionary", before
#endif

/* Word with tagged phone transcription\n */
%ignore Dictword;
class Dictword {
 public:
  ~Dictword();

    /*  */
    String puts();
};

class DictwordPtr {
 public:
  %extend {
    Dictword() { return new DictwordPtr(new Dictword()); }
  }

  Dictword* operator->();
};

%rename(add_sw) add(const String, WordPtr);
%feature("shadow") add(const String, const String) {
def add(*args):
	import types
	if type(args[1]) == types.StringType:
		return apply(dictc.DictionaryPtr_add, args)
	return(dictc.DictionaryPtr_add_sw, args)
}


/* Set of words */
%ignore Dictionary;
class Dictionary {
 public:
  ~Dictionary();

  // displys the first count items of the dictionary
  String puts(int count = 20);

  // returns true if name is in the dictionary
  bool has_word(const String name);

  // add a new word to the set
  void add(const String n, WordPtr w);
  void add(const String n, const String pronunciation);

  // remove word from the set
  void erase(const String n);

  // reads a dictionary file
  void read(const String filename);

  // writes a dictionary file
  void write(const String filename);

  // return the spelled word given the index
  String name();
};

class DictionaryPtr {
 public:
  %extend {
    DictionaryPtr(PhonesPtr p, TagsPtr t) {
      return new DictionaryPtr(new Dictionary(p, t));
    }
  }

  Dictionary* operator->();
};
