//
//                               Millennium
//                    Distant Speech Recognition System
//                                 (dsr)
//
//  Module:  asr.dict
//  Purpose: Dictionary module.
//  Author:  John McDonough
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


#ifndef _distribTree_h_
#define _distribTree_h_

#include "common/common.h"
#include "dictionary/phones.h"

#include <bitset>

// ----- definition for class `Lexicon' -----
//
///
/// 'Lexicon' converts between string symbols and integer indices, and vice versa.
/// Both capabilities are implemented are based on the List<> template type.
///
typedef Countable LexiconCountable;
class Lexicon : public LexiconCountable {
 public:
  Lexicon(const String& nm, const String& fileName = "");
  ~Lexicon() { }

  void clear() { _list.clear(); }
  const String& name() const { return _list.name(); }
  unsigned size() const { return _list.size(); }

  void read(const String& fileName);
  void write(const String& fileName, bool writeHeader = false) const;

  unsigned index(const String& symbol, bool create = false);
  const String& symbol(unsigned idx) const { return _list[idx]; }
  bool isPresent(const String& symbol) const { return _list.isPresent(symbol); }

  class Iterator;  friend class Iterator;

 private:
  typedef List<String>				_List;
  typedef _List::Iterator			_ListIterator;
  typedef _List::ConstIterator			_ListConstIterator;

  char						_commentChar;
  _List						_list;
};

typedef refcountable_ptr<Lexicon> LexiconPtr;


// ----- definition for class `Lexicon::Iterator' -----
//
class Lexicon::Iterator {
 public:
  Iterator(LexiconPtr& lex)
     : _lexicon(lex), _itr(lex->_list) { }

  String operator->()  { return *_itr; }
  String operator*()   { return *_itr; }
  String name() const  { return *_itr; }
  void operator++(int) { _itr++; }
  bool more()          { return ((_lexicon->size() > 0) && (_itr.more())); }
  inline String next();

 private:
  LexiconPtr					_lexicon;
  _ListIterator					_itr;
};

// needed for Python iterator
String Lexicon::Iterator::next() {
  if (!more())
    throw jiterator_error("end of lexicon!");

  String st(name());
  operator++(1);
  return st;
}


// ----- definition for class `DistribTree' -----
// 
class DistribTree;
class Lexicon;
typedef refcountable_ptr<Lexicon> LexiconPtr;
typedef refcountable_ptr<DistribTree> DistribTreePtr;

class DistribTree : public Countable {
 protected:
  typedef enum { No, Yes, Unknown } Answer;
  static bool  Verbose;
  static char* AnswerList[];

  class Question {
  public:
    Question() { }
    Question(int context, PhonesPtr ph, const String& eps = "eps", const String& wb = "WB", const String& eos = "</s>", bool negate = false)
      : _context(context), _phones(ph), _eps(eps), _wb(wb), _eos(eos), _phone(""), _negate(negate) { }
    Question(int context, const String& ph, const String& eps = "eps", const String& wb = "WB", const String& eos = "</s>", bool negate = false)
      : _context(context), _phone(ph), _eps(eps), _wb(wb), _eos(eos), _negate(negate) { }

    operator String() const;

    Answer answer(const String& contextCbk) const;
    const PhonesPtr& getPhones() const { return _phones; }
    int getContext() const { return _context;};
    const String& getPhone() const { return _phone; }
    bool negate() const { return _negate; }
    const String& wb() const { return _wb; }

    const String toString() const {
      static char buffer[50];
      sprintf(buffer, "%d=%s", _context, (_phone == "" ? _phones->name().c_str() : _phone.c_str()));
      return String(buffer);
    }

    Question& operator~();

  private:
    PhonesPtr					_phones;
    String					_phone;
    int						_context;
    String					_eps;
    String					_eos;
    String					_wb;
    bool					_negate;
  };

  class QuestionList {

    typedef list<Question>			_List;
    typedef _List::value_type			_ListType;
    typedef _List::iterator			_ListIterator;
    typedef _List::const_iterator		_ConstListIterator;

  public:
    QuestionList() { }

    class QuestionListList;  friend class QuestionListList;

    class Iterator {
    public:
    Iterator(QuestionList& questionList)
      : _qlist(questionList._qlist), _itr(_qlist.begin()) { }

      bool more() const { return _itr != _qlist.end(); }
      void operator++(int) { if (more()) _itr++; }
      Question& question() const { return *_itr; }
      Question& operator*() const { return *_itr; }

    private:
      _List&					_qlist;
      _ListIterator				_itr;
    };
    friend class Iterator;

    class ConstIterator {
    public:
    ConstIterator(const QuestionList& questionList)
      : _qlist(questionList._qlist), _itr(_qlist.begin()) { }

      bool more() const { return _itr != _qlist.end(); }
      void operator++(int) { if (more()) _itr++; }
      const Question& question()  const { return *_itr; }
      const Question& operator*() const { return *_itr; }

    private:
      const _List&				_qlist;
      _ConstListIterator			_itr;
    };
    friend class ConstIterator;

    void add(const Question& question) { _qlist.push_back(question); }

    const String toString() const;

    Answer answer(const String& context) const;

  private:
    _List					_qlist;
  };

  class QuestionListList {

    typedef list<QuestionList>			_ListList;
    typedef _ListList::value_type		_ListListType;
    typedef _ListList::iterator			_ListListIterator;
    typedef _ListList::const_iterator		_ConstListListIterator;

  public:
    QuestionListList() { }

    class Iterator {
    public:
    Iterator(const QuestionListList& questionList)
      : _qlist(questionList._qlist), _itr(_qlist.begin()) { }

      bool more() const { return _itr != _qlist.end(); }
      void operator++(int) { if (more()) _itr++; }
      const QuestionList& questionList() const { return *_itr; }
      const QuestionList& operator*() const { return *_itr; }

    private:
      const _ListList&				_qlist;
      _ConstListListIterator			_itr;
    };
    friend class Iterator;

    void add(const QuestionList& questionList) { _qlist.push_back(questionList); }

    QuestionListList negate();
    QuestionListList& operator&=(QuestionList questionList);

    const String toString() const;

    bool empty() const { return toString() == "{{}}"; }

    Answer answer(const String& context) const;

  private:
    _ListList					_qlist;
  };

 public:
  class _BitMatrixList;

  class _BitMatrix {

  public:
    _BitMatrix(unsigned rows = 0, unsigned columns = 0);
    _BitMatrix(const _BitMatrix& other);
    _BitMatrix(const Question& question, LexiconPtr& phonesLexicon, unsigned contextLength = 1);
    _BitMatrix(const QuestionList& questionList, LexiconPtr& phonesLexicon, unsigned contextLength = 1);
    ~_BitMatrix();
 
    bool getAt(unsigned row, unsigned column) const;	// returns true if the bit at position (column, row) is set
    void setAt(unsigned row, unsigned column);		// sets the bit at position (column, row) to 1
    void resetAt(unsigned row, unsigned column);	// sets the bit at position (column, row) to 0
    void flipAt(unsigned row, unsigned column);		// flips the bit at position (column, row)
    void set();						// sets all bits in the bit matrx to 1
    void reset();					// sets all bits in the bit matrx to 0

    void setColumn(unsigned column);			// sets all the bits in the column to 1
    void resetColumn(unsigned column);			// sets all the bits in the column to 0

    unsigned nSet(unsigned column);			// returns the number of bits in the column set to 1

    bool isSet(unsigned column) const;			// returns true if all the bits in the column are set to 1
    bool isUnset(unsigned column) const;		// returns true if all the bits in the column are set to 0

    bool isValid() const;				// returns true if there is at least one bit set to 1 in each column
    bool isZero() const;				// returns true if the bitmatrix consits entirely of zeroes

    void dump() const;
    void dump(const LexiconPtr& phoneLex) const;
    void dumpEx() const;

    friend const _BitMatrix operator<<(const _BitMatrix& bm, unsigned places);
    friend const _BitMatrix operator>>(const _BitMatrix& bm, unsigned places);
    friend const _BitMatrix operator& (const _BitMatrix& left, const _BitMatrix& right);
    friend const _BitMatrix operator|(const _BitMatrix& left, const _BitMatrix& right);
    friend bool             operator==(const _BitMatrix& left, const _BitMatrix& right);
    friend bool             operator!=(const _BitMatrix& left, const _BitMatrix& right);

    const _BitMatrix& operator<<=(unsigned places);
    const _BitMatrix& operator>>=(unsigned places);

    const _BitMatrix& operator&=(const _BitMatrix& right);
    const _BitMatrix& operator|=(const _BitMatrix& right);
    const _BitMatrix& operator|=(const _BitMatrixList& right);
    const _BitMatrix& operator=(const _BitMatrix& right);
    const _BitMatrix& operator~();

    unsigned rowsN() const { return _rows; }
    unsigned columnsN() const { return _columns; }

  private:
    unsigned _bitsN(unsigned char byte);

    static unsigned char			_mask[8];

    unsigned					_rows;
    unsigned					_columns;
    size_t					_size;
    unsigned char*				_matrix;
  };

  class _BitMatrixList : public Countable {
    typedef list<_BitMatrix>			_BitMatLst;
    typedef _BitMatLst::iterator		_BitMatLstIterator;
    typedef _BitMatLst::const_iterator		_BitMatLstConstIterator;

  public:
    _BitMatrixList() { }
    _BitMatrixList(unsigned rowsN, unsigned colsN, unsigned silenceX, unsigned padX);
    _BitMatrixList(const QuestionListList& questionList, LexiconPtr& phonesLexicon, unsigned contextLength = 1);
    _BitMatrixList(const _BitMatrix& bm, unsigned count = 1);

    void add(_BitMatrix& bmatrix) { _bitMatLst.push_back(bmatrix); }

    void clear() { _bitMatLst.clear(); }
    unsigned size() const { return _bitMatLst.size(); }

    friend const _BitMatrixList operator<<(const _BitMatrixList& bml, unsigned places);
    friend const _BitMatrixList operator>>(const _BitMatrixList& bml, unsigned places);
    friend const _BitMatrixList operator&(const _BitMatrixList& left, const _BitMatrixList& right);
    friend bool connect(const _BitMatrixList& left, const _BitMatrixList& right);
    friend bool operator==(const _BitMatrixList& left, const _BitMatrixList& right);
    friend bool operator!=(const _BitMatrixList& left, const _BitMatrixList& right);

    const _BitMatrixList& operator<<=(unsigned places);
    const _BitMatrixList& operator>>=(unsigned places);
    const _BitMatrixList& operator&=(const _BitMatrixList& right);
    const _BitMatrixList& operator|=(const _BitMatrixList& right);

    void dump(const LexiconPtr& phoneLex) const;
    void dumpEx() const;
    bool isValid() const;

    class Iterator;       friend class Iterator;
    class ConstIterator;  friend class ConstIterator;

  private:
    _BitMatLst					_bitMatLst;
  };

  typedef refcountable_ptr<_BitMatrixList>      _BitMatrixListPtr;

  class _BitMatrixList::Iterator {
  public:
  Iterator(_BitMatrixList& bmatlist)
    : _bmatlist(bmatlist._bitMatLst), _itr(_bmatlist.begin()) { }
  Iterator(_BitMatrixListPtr& bmatlist)
    : _bmatlist(bmatlist->_bitMatLst), _itr(_bmatlist.begin()) { }

    _BitMatrix& operator*()   { return *_itr; }
    void operator++(int) { _itr++; }
    bool more()          { return (_itr != _bmatlist.end()); }

  private:
    _BitMatLst					_bmatlist;
    _BitMatLstIterator				_itr;
  };

  class _BitMatrixList::ConstIterator {
  public:
  ConstIterator(const _BitMatrixList& bmatlist)
    : _bmatlist(bmatlist._bitMatLst), _itr(_bmatlist.begin()) { }
  ConstIterator(const _BitMatrixListPtr& bmatlist)
    : _bmatlist(bmatlist->_bitMatLst), _itr(_bmatlist.begin()) { }

    const _BitMatrix& operator*()   { return *_itr; }
    void operator++(int) { _itr++; }
    bool more()          { return (_itr != _bmatlist.end()); }

  private:
    const _BitMatLst				_bmatlist;
    _BitMatLstConstIterator			_itr;
  };

  typedef map<String, _BitMatrixList >		_BitmapList;
  typedef _BitmapList::iterator			_BitmapListIterator;
  typedef _BitmapList::const_iterator		_BitmapListConstIterator;

 public:  
  class Node;
  typedef refcountable_ptr<Node>		NodePtr;

 protected:
  typedef map<String, NodePtr>			_Map;
  typedef _Map::value_type			_MapType;
  typedef _Map::iterator			_MapIterator;
  typedef _Map::const_iterator			_MapConstIterator;

 public:
  class Node : public Countable {

    friend class DistribTree;

  public:
    Node(PhonesSetPtr& ps, DistribTree* dt,
	 const String& nm, const String& ques,
	 const String& left, const String& right, const String& unknown = "-", const String& leaf = "-");

    Node(PhonesSetPtr& ps, DistribTree* dt,
	 const String& nm, QuestionListList qlist,
	 const String& left, const String& right, const String& unknown = "-", const String& leaf = "-");

      const String& name() const { return _name; }

      const String& leaf(const String& context) const;
      const String& left()    const { return _left;    }
      const String& right()   const { return _right;   }
      const String& leaf()    const { return _leaf;    }
      const String& unknown() const { return _unknown; }

      void write(FILE* fp);

      LexiconPtr& contexts() { return _contexts; }

    protected:
      DistribTree*				_distribTree;
      const String				_name;
      QuestionListList				_qlist;
      /* const */ String			_left;
      /* const */ String			_right;
      /* const */ String			_unknown;
      /* const */ String			_leaf;
      _BitMatrixList				_bitMatrixList;
      LexiconPtr				_contexts;
  };

  class Iterator {
  public:
    Iterator(DistribTreePtr& tree)
      : _tree(tree), _itr(_tree->_map.begin()) { }

      void operator++(int) { _itr++; }
      bool more() const { return _itr != _tree->_map.end(); }
      const NodePtr node() const { return (*_itr).second; }
      NodePtr next() {
	if (more()) {
	  NodePtr n(node());
	  operator++(1);
	  return n;
	} else {
	  throw jiterator_error("end of distributions!");
	}
      }

  private:
    DistribTreePtr				_tree;
    _MapIterator				_itr;
  };
  friend class Iterator;

  DistribTree(const String& nm, PhonesSetPtr& phonesSet, const int contextLength = 1, const String& fileName = "",
	      char comment = ';', const String& wb = "WB", const String& eos = "</s>", bool verbose = false, bool sphinx = false);

  const String& begin(const String& context);
  const String& middle(const String& context);
  const String& end(const String& context);
  const NodePtr& node(const String& nm) const { return _node(nm); }

  void read(const String& treePath, const String& state = "", unsigned offsetX = 0, bool sphinx = false);

  void write(const String& fileName, const String& date = "");

  unsigned leavesN() const;

  const String& name() const { return _name; };

  const _BitmapList& buildMatrices(LexiconPtr& phonesLexicon);

  const int contextLength() { return _contextLength; };

  static String getDecisionPhone(String decisionPhone);

 protected:

  void _read(const String& fileName);

  Question _parseQuestion(String& context);
  Question _parseQuestionMillennium(String& context);
  Question _parseQuestionSphinx(String& context);
  QuestionList _parseConjunction(String& context);
  QuestionListList _parseCompoundQuestion(const String context);
  void _readSphinx(const String& treePath, const String& state, unsigned offset);
  LexiconPtr _extractPhoneLexicon(LexiconPtr& phoneLex);

  virtual NodePtr _newNode(PhonesSetPtr& ps, DistribTree* dt,
			   const String& nm, const String& ques = "",
			   const String& left = "-", const String& right = "-", const String& unknown = "-", const String& leaf = "-");

  virtual NodePtr _newNode(PhonesSetPtr& ps, DistribTree* dt,
			   const String& nm, QuestionListList& qlist,
			   const String& left, const String& right, const String& unknown = "-", const String& leaf = "-");

  unsigned _maxIndex() const;

  void parseTree(const NodePtr& node, _BitMatrixList& bml, LexiconPtr& inputLexicon);

        NodePtr& _node(const String& nm);
  const NodePtr& _node(const String& nm) const;

  String					_name;
  char						_commentChar;
  PhonesSetPtr					_phonesSet;
  String					_wb;
  String					_eos;
  _Map						_map;
  _BitmapList                   		_leafmatrices;
  const int                     		_contextLength;
  bool						_sphinx;
  char						_open;
  char						_close;
};

#endif
