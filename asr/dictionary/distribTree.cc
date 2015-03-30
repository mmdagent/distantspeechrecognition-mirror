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



#include <stdlib.h>
#include <sstream>
#include "common/mlist.h"
#include "dictionary/distribTree.h"

bool  DistribTree::Verbose      = true;
char* DistribTree::AnswerList[] = { "No", "Yes", "Unknown" };

unsigned char DistribTree::_BitMatrix::_mask[8] = {128U, 64U, 32U, 16U, 8U, 4U, 2U, 1U};

// ----- methods for class `Lexicon' -----
//
Lexicon::Lexicon(const String& nm, const String& fileName)
  : _commentChar(';'), _list(nm)
{
  // cout << "Created " << name() << " lexicon." << endl;

  if (fileName != "") read(fileName);
}

void Lexicon::read(const String& fileName)
{
  if (fileName == "")
    jio_error("File name is null.");

  cout << "Reading lexicon '" << name() << "' from file '" << fileName << "'."
       << endl;

  clear();

  FILE*         fp     = fileOpen(fileName, "r");
  static size_t n      = 1000;
  static char*  buffer = (char*) malloc(1000 * sizeof(char));
  unsigned      nsyms  = 0;

  while(getline(&buffer, &n, fp) > 0) {
    static char* token[2];

    if ( buffer[0] == _commentChar ) continue;

    token[0] = strtok(buffer, " \t\n");
    token[1] = strtok(NULL, " \t\n");

    String symbol(token[0]);
    unsigned index = nsyms;
    if (token[1] != NULL)
      sscanf(token[1], "%u", &index);

    // cout << "Adding " << symbol << " : " << index << " to Lexicon." << endl;

    if (_list.isPresent(symbol)) {
      // throw jkey_error("Symbol %s already exists.", symbol.c_str());
      printf("Symbol %s already exists.\n", symbol.c_str());
      continue;
    }

    _list.add(symbol, symbol);
    nsyms++;
  }

  fileClose(fileName, fp);
}

void Lexicon::write(const String& fileName, bool writeHeader) const
{
  if (fileName == "")
    throw jio_error("File name is null.\n");

  if (_list.size() == 0)
    throw jio_error("Lexicon '%s' has no entries.\n", name().c_str());

  FILE* fp = fileOpen(fileName, "w");

  if (writeHeader) {
    fprintf( fp, "%c -------------------------------------------------------\n",
	     _commentChar);
    fprintf( fp, "%c  Name            : %s\n",		_commentChar,
	     name().c_str());
    fprintf( fp, "%c  Type            : Lexicon\n",	_commentChar);
    fprintf( fp, "%c  Number of Items : %d\n",		_commentChar,
	     size());
    fprintf( fp, "%c  Date            : %s",		_commentChar,
	     dateString());
    fprintf( fp, "%c -------------------------------------------------------\n",
	     _commentChar);
  }

  // write all non-end states
  unsigned cnt = 0;
  for (_ListConstIterator itr(_list); itr.more(); itr++) {
    const String& symbol(*itr);

    fprintf(fp, "%30s %10d\n", symbol.c_str(), cnt++);
  }

  fileClose(fileName, fp);
}

unsigned Lexicon::index(const String& symbol, bool create)
{
  if (create == false)
    return _list.index(symbol);

  if (_list.isPresent(symbol))
    return _list.index(symbol);

  return _list.add(symbol, symbol);
}


// ----- methods for class `DistribTree::Question' -----
//
DistribTree::Question::operator String() const
{
  static char buffer[100];
  sprintf(buffer, "%d", _context);
  String contextString(buffer);

  if (_negate)
    contextString = String("!") + contextString;

  if (_phones.isNull() == false)
    return contextString + "=" + _phones->name();
  else
    return contextString + "=" + _phone;
}

DistribTree::Question& DistribTree::Question::operator~()
{
  if (_negate)
    _negate = false;
  else
    _negate = true;

  return *this;
}

// Sample Context: UW/{JH:WB};{SIL:WB}_{N:WB};{SIL:WB}, corresponds to {SIL:WB}{JH:WB}UW{N:WB}{SIL:WN}
DistribTree::Answer DistribTree::Question::answer(const String& contextCbk) const
{
  String::size_type epos = contextCbk.find_first_of("-");
  String context;
  if (epos == String::npos)
    context = contextCbk;
  else
    context = contextCbk.substr(0, epos);

  String::size_type pos1 = context.find_first_of("/");
  
  String phone(context.substr(0,pos1));
  
  String::size_type pos2 = context.find_first_of("_",pos1+1);
  
  String left(context.substr(pos1+1,pos2-(pos1+1)));
  String right(context.substr(pos2+1));
  
  if (Verbose)
    printf("Left = %s : Phone = %s : Right = %s\n", left.c_str(), phone.c_str(), right.c_str());
  
  String decisionPhone;
  
  if (_context == 0) {
    
    decisionPhone = phone;
    
  } else if (_context < 0) {

    int con = (-_context) - 1;
    for (int i = 0; i < con; i++) {
      String::size_type pos = left.find_first_of(";");
      left.erase(0, pos+1);
    }
    if (left.find_first_of(";") != String::npos) {
      String::size_type pos = left.find_first_of(";");
      left.erase(pos);
    }
    decisionPhone = left;

  } else {
    int con = _context - 1;
    for (int i = 0; i < con; i++) {
      String::size_type pos = right.find_first_of(";");
      right.erase(0, pos+1);
    }
    if (right.find_first_of(";") != String::npos) {
      String::size_type pos = right.find_first_of(";");
      right.erase(pos);
    }
    decisionPhone = right;
  }

  if (decisionPhone == "")
    throw jconsistency_error("Decision phone is empty.");

  if (Verbose)
    printf("Decision Phone Initial = %s\n", decisionPhone.c_str());

  // 'eps' and '</s>' are both treated like the phone 'PAD'
  if (decisionPhone == _eps || decisionPhone == _eos)
    return No;

  // Determine if this is *some* word boundary, begin, end, or single
  bool wb = (decisionPhone.find(_wb) != String::npos), wbSpecific = false;
  if (wb) {
    // Check for specific word boundary condition
    if (_phone.find(_wb) != String::npos)
      wbSpecific = (decisionPhone.find(_phone) != String::npos);
    String::size_type bracketPos    = decisionPhone.find("{");
    decisionPhone.erase(decisionPhone.begin() + bracketPos, decisionPhone.begin() + bracketPos + 1);
    String::size_type underScorePos = decisionPhone.find(":");
    decisionPhone.erase(decisionPhone.begin() + underScorePos, decisionPhone.end());
  }

  if (Verbose)
    printf("Decision Phone Final = %s\n", decisionPhone.c_str());

  DistribTree::Answer returnValue;
  if (_phones.isNull() == false) {
    returnValue = (_phones->hasPhone(decisionPhone)) ? Yes : No;
  } else if (_phone.find(_wb) != String::npos) {
    returnValue = wbSpecific ? Yes : No;
  } else {
    returnValue = (_phone == decisionPhone) ? Yes : No;
  }

  // check for negation
  if (_negate) {
    if (returnValue == Yes)
      returnValue = No;
    else
      returnValue = Yes;
  }

  if (Verbose) {
    if (_phone != "")
      cout << "Phone: " << _phone << endl;
    else
      cout << "Phones: " << _phones->name() << endl;
    cout << "Answer: ";
    if (returnValue == Yes)
      cout << "Yes" << endl;
    else
      cout << "No" << endl;
  }

  return returnValue;
}


// ----- methods for class `DistribTree::QuestionList' -----
//
const String DistribTree::QuestionList::toString() const
{
  String s("{");
  for (_ConstListIterator itr = _qlist.begin(); itr != _qlist.end(); itr++) {
    if (s != "{") s += " ";
    s += String(*itr);
  }
  s += "}";

  return s;
}

DistribTree::Answer DistribTree::QuestionList::answer(const String& context) const
{
  Answer combinedAnswer = Yes;
  for (_ConstListIterator itr = _qlist.begin(); itr != _qlist.end(); itr++) {
    Question question(*itr);
    if (Verbose)
      cout << String(question) << endl;

    Answer ans = question.answer(context);

    if (ans == Unknown)
      throw jconsistency_error("Answer is unknown for context %s.", context.c_str());

    if (ans == No) combinedAnswer = No;
  }

  if (Verbose) {
    cout << "Combined answer: ";
    if (combinedAnswer == Yes)
      cout << "Yes" << endl;
    else
      cout << "No" << endl;
  }
  
  return combinedAnswer;
}


// ----- methods for class `DistribTree::QuestionListList' -----
//
const String DistribTree::QuestionListList::toString() const
{
  if (_qlist.size() == 0)
    return String("{{}}");

  String s("{");
  for (_ConstListListIterator itr = _qlist.begin(); itr != _qlist.end(); itr++) {
    if (s != "{") s += " ";
    s += (*itr).toString();
  }
  s += "}";

  return s;
}

DistribTree::Answer DistribTree::QuestionListList::answer(const String& context) const
{
  if (Verbose)
    cout << "Context: " << context << endl;

  Answer combinedAnswer = No;
  for (_ConstListListIterator itr = _qlist.begin(); itr != _qlist.end(); itr++) {
    const QuestionList& questionList(*itr);

    if (Verbose)
      cout << "Question List: " << questionList.toString() << endl;

    Answer ans = questionList.answer(context);
    if (ans == Yes) {
      combinedAnswer = Yes;

      break;
    }
    if (ans == Unknown)
      throw jconsistency_error("Answer is unknown for context %s.", context.c_str());
  }

  if (Verbose) {
    cout << "Final answer: ";
    if (combinedAnswer == Yes)
      cout << "Yes" << endl;
    else
      cout << "No" << endl;
  }

  return combinedAnswer;
}

DistribTree::QuestionListList DistribTree::QuestionListList::negate()
{
  QuestionList     qlist;
  QuestionListList qqlist;
  qqlist.add(qlist);

  for (_ListListIterator itr = _qlist.begin(); itr != _qlist.end(); itr++) {
    QuestionList& right(*itr);  qqlist &= right;
  }

  return qqlist;
}

DistribTree::QuestionListList& DistribTree::QuestionListList::operator&=(QuestionList questionList)
{
  // negate questions to be added
  for (QuestionList::Iterator itr(questionList); itr.more(); itr++) {
    Question& question(*itr);  ~question;
  }
    
  _ListList nlist;
  for (_ListListIterator itr = _qlist.begin(); itr != _qlist.end(); itr++) {
    for (QuestionList::Iterator qitr(questionList); qitr.more(); qitr++) {
      QuestionList qqlist(*itr);
      Question question(*qitr);  
      qqlist.add(question);
      nlist.push_back(qqlist);
    }
  }

  _qlist = nlist;
  return *this;
}


// ----- methods for class `DistribTree::Node' -----
//
DistribTree::Node::Node(PhonesSetPtr& ps, DistribTree* dt,
			const String& nm, const String& ques,
			const String& left, const String& right, const String& unknown,
			const String& leaf)
  : _distribTree(dt), _name(nm), _left(left), _right(right), _unknown(unknown), _leaf(leaf)
{
  if (ques == "") return;

  if (Verbose)
    printf("Name = %s : Question = %s : Context = %s    %s    %s\n",
	   nm.c_str(), ques.c_str(), _left.c_str(), _right.c_str(), _unknown.c_str());

  list<String> entries;
  splitList(ques, entries);

  QuestionList questionList;
  for (list<String>::iterator itr = entries.begin(); itr != entries.end(); itr++) {
    const String& qu(*itr);
    String::size_type pos = qu.find_first_of("=");
    String contextString  = qu.substr(0,pos);
    int    context        = atoi(contextString.c_str());
    String phoneClass     = qu.substr(pos+1);

    if (Verbose)
      printf("%s : %d : %s\n", qu.c_str(), context, phoneClass.c_str());

    if (ps->hasPhones(phoneClass))
      questionList.add(Question(context, ps->find(phoneClass)));
    else
      questionList.add(Question(context, phoneClass));
  }
  _qlist.add(questionList);
}

DistribTree::Node::Node(PhonesSetPtr& ps, DistribTree* dt,
			const String& nm, QuestionListList qlist,
			const String& left, const String& right, const String& unknown,
			const String& leaf)
  : _distribTree(dt), _name(nm), _qlist(qlist), _left(left), _right(right),
    _unknown(unknown), _leaf(leaf), _contexts(NULL) { }

// for the moment, assume there are only AND questions
// the assignment is:
//   1. No      --> _left
//   2. Yes     --> _right
//   3. Unknown --> _unknown
//
const String& DistribTree::Node::leaf(const String& context) const
{
  if (Verbose)
    printf("Name = %s : Context = %s    %s    %s\n", _name.c_str(), _left.c_str(), _right.c_str(), _unknown.c_str());

  if (_left == "-" && _right == "-" && _unknown == "-") {
    if (_leaf == "-")
      throw jconsistency_error("Leaf is empty.");
    return _leaf;
  }

  if (Verbose)
    cout << _qlist.toString() << endl;

  Answer combinedAnswer = _qlist.answer(context);

  if (combinedAnswer == No) {

    if (Verbose)
      printf("Searching for node %s\n", _left.c_str());

    const NodePtr& nnode(_distribTree->node(_left));
    assert(nnode->name() == _left);
    return nnode->leaf(context);	// i.e., No
  }

  if (Verbose)
    printf("Searching for node %s\n", _right.c_str());

  const NodePtr& ynode(_distribTree->node(_right));
  assert(ynode->name() == _right);  
  return ynode->leaf(context);		// i.e., Yes
}

void DistribTree::Node::write(FILE* fp)
{
  fprintf(fp, "%s %s %s %s %s %s\n",
	  _name.c_str(), _qlist.toString().c_str(), _left.c_str(), _right.c_str(), _unknown.c_str(), _leaf.c_str());

  if (_left != "-")
    _distribTree->node(_left)->write(fp);

  if (_right != "-")
    _distribTree->node(_right)->write(fp);
}

DistribTree::NodePtr& DistribTree::_node(const String& nm)
{
  _MapIterator itr = _map.find(nm);

  if (itr == _map.end())
    throw jkey_error("Could not find node '%s'.", nm.c_str());

  return (*itr).second;
}

const DistribTree::NodePtr& DistribTree::_node(const String& nm) const
{
  _MapConstIterator itr = _map.find(nm);

  if (itr == _map.end())
    throw jkey_error("Could not find node '%s'.", nm.c_str());

  return (*itr).second;
}


// ----- methods for class 'DistribTree::_BitMatrix' ------
//
DistribTree::_BitMatrix::_BitMatrix(const _BitMatrix& other)
  : _rows(other._rows), _columns(other._columns), _size(other._size),
    _matrix(new unsigned char[_size])
{
  memcpy(_matrix, other._matrix, _size);
}

DistribTree::_BitMatrix::_BitMatrix(unsigned rows, unsigned columns)
  : _rows(rows), _columns(columns), _size(0), _matrix(NULL)
{
  if (_rows == 0 && _columns == 0) return;

  unsigned rest = rows * columns & 7;
  if (rest != 0)
    _size = (_rows * _columns >> 3) + 1;
  else
    _size = (_rows * _columns >> 3);
  _matrix = new unsigned char[_size];

  memset(_matrix, 255, _size);
}

DistribTree::_BitMatrix::_BitMatrix(const Question& question, LexiconPtr& phonesLexicon, unsigned contextLength)
  : _rows(0), _columns(0), _size(0), _matrix(NULL)
{
  unsigned context = contextLength + question.getContext();
  _BitMatrix bm(phonesLexicon->size(), 2 * contextLength + 1);  bm.reset();
  const String& phoneQ(question.getPhone());
  if (phoneQ == "") {					// handle class of phones case 
    const PhonesPtr& phones = question.getPhones();
    unsigned idx = 0;
    for (Lexicon::Iterator itr(phonesLexicon); itr.more(); itr++) {
      const String& phone(getDecisionPhone(*itr));
      if (phones->hasPhone(phone)) bm.setAt(idx, context);
      idx++;
    }
  } else {
    if (phoneQ.find(question.wb()) == String::npos) {	// handle single phone case
      unsigned idx = 0;
      for (Lexicon::Iterator itr(phonesLexicon); itr.more(); itr++) {
	const String& phone(getDecisionPhone(*itr));
	if (phone == phoneQ)
	  bm.setAt(idx, context);
	idx++;
      }
    } else {						// handle case for word boundary conditions
      unsigned idx = 0;
      for (Lexicon::Iterator itr(phonesLexicon); itr.more(); itr++) {
	const String& phone(*itr);
	if (phone.find(phoneQ) != String::npos) bm.setAt(idx, context);
	idx++;
      }
    }
  }

  if (question.negate()) ~bm;
  for (unsigned colX = 0; colX < 2 * contextLength + 1; colX++) {
    if (colX == context) continue;
    bm.setColumn(colX);
  }
  *this = bm;
}

DistribTree::_BitMatrix::_BitMatrix(const QuestionList& questionList, LexiconPtr& phonesLexicon, unsigned contextLength)
  : _rows(0), _columns(0), _size(0), _matrix(NULL)
{
  _BitMatrix bm(phonesLexicon->size(), 2 * contextLength + 1);
  for (QuestionList::ConstIterator itr(questionList); itr.more(); itr++) {
    const Question& question(*itr);
    bm &= _BitMatrix(question, phonesLexicon, contextLength);
  }
  *this = bm;
}

DistribTree::_BitMatrix::~_BitMatrix()
{
  delete[] _matrix;
}
 
bool DistribTree::_BitMatrix::getAt(unsigned row, unsigned column) const
{
  unsigned basis = column*_rows + row;
  return ((_matrix[basis >> 3] & _mask[basis & 7]) == 0) ? false : true;
}
 
void DistribTree::_BitMatrix::setAt(unsigned row, unsigned column)
{
  unsigned basis = column*_rows + row;
  _matrix[basis >> 3] |= _mask[basis & 7];
}
 
void DistribTree::_BitMatrix::resetAt(unsigned row, unsigned column)
{
  unsigned basis = column*_rows + row;
  _matrix[basis >> 3] &= ~_mask[basis & 7];
}

void DistribTree::_BitMatrix::flipAt(unsigned row, unsigned column)
{
  unsigned basis = column*_rows + row;
  _matrix[basis >> 3] ^= _mask[basis & 7];
}

void DistribTree::_BitMatrix::set()
{
  memset(_matrix, 255, _size);
}

void DistribTree::_BitMatrix::reset()
{
  memset(_matrix, 0, _size);
}

void DistribTree::_BitMatrix::setColumn(unsigned column)
{
  unsigned startByte = (column*_rows) >> 3;  // equivalent to / 8
  unsigned char startBits = (column*_rows) & 7 ; // equivalent to % 8
  unsigned endByte   = ((column+1)*_rows) >> 3;
  unsigned char endBits = ((column+1)*_rows) & 7;

  // the whole column is contained in one byte
  if (startByte == endByte) {
    _matrix[startByte] |= ~((255 << (8-startBits)) | (255 >> endBits));
    return;
  }
  
  // set the start fragment
  _matrix[startByte] |= (255 >> startBits);
  
  // set the end fragment
  if (endBits != 0)
    _matrix[endByte] |= (255 << (8-endBits));
  
  // set the inbetween fragments
  for (unsigned i = startByte+1; i < endByte; i++)
    _matrix[i] |= 255;

}

void DistribTree::_BitMatrix::resetColumn(unsigned column)
{
  unsigned startByte = (column*_rows) >> 3;  // equivalent to / 8
  unsigned char startBits = (column*_rows) & 7 ; // equivalent to % 8
  unsigned endByte   = ((column+1)*_rows) >> 3;
  unsigned char endBits = ((column+1)*_rows) & 7;

  // the whole column is contained in one byte
  if (startByte == endByte) {
    _matrix[startByte] &= ((255 << (8-startBits)) | (255 >> endBits));
    return;
  }

  // unset the start fragment
  _matrix[startByte] &= ~(255 >> startBits);
  
  // unset the end fragment
  if (endBits != 0)
    _matrix[endByte] &= (255 >> endBits);
  
  // unset the inbetween fragments
  for (unsigned i = startByte+1; i < endByte; i++)
    _matrix[i] &= 0;
}

unsigned DistribTree::_BitMatrix::nSet(unsigned column)
{
  unsigned count = 0;
  
  unsigned startByte = (column*_rows) >> 3;  // equivalent to / 8
  unsigned char startBits = (column*_rows) & 7 ; // equivalent to % 8
  unsigned endByte   = ((column+1)*_rows) >> 3;
  unsigned char endBits = ((column+1)*_rows) & 7;

  // the whole column is contained in one byte
  if (startByte == endByte) {
    return _bitsN(_matrix[startByte] & ~((255 << (8-startBits)) | (255 >> endBits)));
  }

  // count the bits in the start fragment
  count = _bitsN(_matrix[startByte] << startBits);

  // count the bits in the end fragment
  if (endBits != 0)
    count += _bitsN(_matrix[endByte] >> (8-endBits));
  
  // count the bits in the inbetween fragments
  for (unsigned i = startByte+1; i < endByte; i++)
    count += _bitsN(_matrix[i]);
  
  return count; 
}

bool DistribTree::_BitMatrix::isSet(unsigned column) const
{
  unsigned startByte = (column*_rows) >> 3;  // equivalent to / 8
  unsigned char startBits = (column*_rows) & 7 ; // equivalent to % 8
  unsigned endByte   = ((column+1)*_rows) >> 3;
  unsigned char endBits = ((column+1)*_rows) & 7;
  
  unsigned char mask;

  // the whole column is contained in one byte
  if (startByte == endByte) {
    mask = ~((255 << (8-startBits)) | (255 >> endBits));
    return ((_matrix[startByte] & mask) == mask);
  }

  // check the start fragment
  mask = 255 >> startBits;
  if ((unsigned char)(_matrix[startByte] & mask) != mask)
    return false;

  // check the end fragment
  if (endBits != 0){
    mask = ~(255 >> endBits);
    if ((unsigned char)(_matrix[endByte] & mask) != mask)
      return false;
  }

  // check the inbetween fragments; all have to be 1s
  for (unsigned i = startByte+1; i < endByte; i++)
    if (_matrix[i] != 255)
      return false;
  
  return true;
}

bool DistribTree::_BitMatrix::isUnset(unsigned column) const
{
  unsigned startByte = (column*_rows) >> 3;  // equivalent to / 8
  unsigned char startBits = (column*_rows) & 7 ; // equivalent to % 8
  unsigned endByte   = ((column+1)*_rows) >> 3;
  unsigned char endBits = ((column+1)*_rows) & 7;
  
  unsigned char mask;

  // the whole column is contained in one byte
  if (startByte == endByte) {
    mask = ~((255 << (8-startBits)) | (255 >> endBits));
    return ((_matrix[startByte] & mask) == 0);
  }

  // check the start fragment
  if ((unsigned char)(_matrix[startByte] << startBits) != 0)
    return false;

  // check the end fragment
  if (endBits != 0)
    if ((unsigned char)(_matrix[endByte] >> (8-endBits)) != 0U)
      return false;

  // check the inbetween fragments; all have to be 1s
  for (unsigned i = startByte+1; i < endByte; i++)
    if (_matrix[i] != 0)
      return false;
  
  return true;
}

bool DistribTree::_BitMatrix::isValid() const
{
  bool res = true;
  for (int i = 0; i < _columns; i++)
    res = res && !isUnset(i);
  return res;
}

bool DistribTree::_BitMatrix::isZero() const
{
  for (int i = 0; i < _size-1; i++)
    if (_matrix[i] != 0)
      return false;
  
  int spareBits = _size*8 - _columns*_rows;
  if ((unsigned char)(_matrix[_size-1] << spareBits) != 0)
    return false;
  
  return true;
}

void DistribTree::_BitMatrix::dump() const
{
  for (int i = 0;  i < _rows; i++) {
    cout << endl;
    for (int j = 0; j < _columns; j++)
      cout << (getAt(i,j) ? 1 : 0 );
  }
  cout << endl;
}

void DistribTree::_BitMatrix::dump(const LexiconPtr& phoneLex) const
{
  for (unsigned i = 0; i < _rows; i++) {
    cout.width(15);
    cout << phoneLex->symbol(i) << " : ";
    for (unsigned j = 0; j < _columns; j++)
      cout << (getAt(i,j) ? 1 : 0);
    cout << endl;
  }
  cout << endl; 
}

void DistribTree::_BitMatrix::dumpEx() const
{
  for (size_t i = 0;  i < _size; i++) {
    printf((_matrix[i] & 1) != 0 ? "1" : "0");
    printf((_matrix[i] & 2) != 0 ? "1" : "0");
    printf((_matrix[i] & 4) != 0 ? "1" : "0");
    printf((_matrix[i] & 8) != 0 ? "1" : "0");
    printf((_matrix[i] & 16) != 0 ? "1" : "0");
    printf((_matrix[i] & 32) != 0 ? "1" : "0");
    printf((_matrix[i] & 64) != 0 ? "1" : "0");
    printf((_matrix[i] & 128) != 0 ? "1" : "0");
    printf(" ");
  }
  printf("\n");
}

const DistribTree::_BitMatrix operator<<(const DistribTree::_BitMatrix& bm, unsigned places)
{
  return DistribTree::_BitMatrix(bm) <<= places;
}

const DistribTree::_BitMatrix& DistribTree::_BitMatrix::operator<<=(unsigned places)
{
  int nBits = places*_rows;
  unsigned nBytes = nBits >> 3;
  
  if (nBytes > 0) {
    memmove(_matrix+nBytes, _matrix, _size-nBytes);
    memset(_matrix, 255, nBytes);
  }
  
  int restBits = nBits & 7;
  
  if (restBits > 0) {
    int bits8 = 8-restBits;
    for (int i = _size-1; i>(signed int)(nBytes-1); i--)
      _matrix[i] = (_matrix[i] >> restBits) | (_matrix[i-1] << bits8);
  }
  return *this;
}

const DistribTree::_BitMatrix operator>>(const DistribTree::_BitMatrix& bm, unsigned places)
{
  return DistribTree::_BitMatrix(bm) >>= places;
}
 
const DistribTree::_BitMatrix& DistribTree::_BitMatrix::operator>>=(unsigned places)
{
  int nBits = places*_rows;
  int nBytes = nBits >> 3;
  
  int spareBits = _rows*_columns & 7;
  if (spareBits != 0)
    _matrix[_size-1] |= 255 >> spareBits;
  
  if (nBytes > 0) {
    memmove(_matrix, _matrix+nBytes, _size-nBytes);
    memset(_matrix+_size-nBytes, 255, nBytes);
  }
  
  int restBits = nBits & 7;
  
  if (restBits > 0) {
    int bits8 = 8-restBits;
    for (unsigned i = 0; i<_size-nBytes-1; i++)
      _matrix[i] = (_matrix[i] << restBits) | (_matrix[i+1] >> bits8);
    _matrix[_size-nBytes-1] <<= restBits;
    _matrix[_size-nBytes-1] |= 255 >> bits8;
  }
  
  return *this;
}
  
const DistribTree::_BitMatrix operator&(const DistribTree::_BitMatrix& left, const DistribTree::_BitMatrix& right)
{
  typedef DistribTree::_BitMatrix     _BitMatrix;

  _BitMatrix bitMatrix(left);
  bitMatrix &= right;
  return bitMatrix;
}

const DistribTree::_BitMatrix& DistribTree::_BitMatrix::operator&=(const _BitMatrix& right)
{
  for (unsigned i = 0; i < _size; i++)
    _matrix[i] &= right._matrix[i];
  return *this;
}

const DistribTree::_BitMatrix operator|(const DistribTree::_BitMatrix& left, const DistribTree::_BitMatrix& right)
{
  typedef DistribTree::_BitMatrix     _BitMatrix;

  _BitMatrix bitMatrix(left);
  bitMatrix |= right;
  return bitMatrix;
} 
 
const DistribTree::_BitMatrix& DistribTree::_BitMatrix::operator|=(const DistribTree::_BitMatrix& right)
{
  for (unsigned i = 0; i < _size; i++)
    _matrix[i] |= right._matrix[i];
  return *this;
}

const DistribTree::_BitMatrix& DistribTree::_BitMatrix::operator|=(const DistribTree::_BitMatrixList& right)
{
  for (_BitMatrixList::ConstIterator itr(right); itr.more(); itr++) {
    const _BitMatrix& bmatrix(*itr);  *this |= bmatrix;
  }
  return *this;
}

const DistribTree::_BitMatrix& DistribTree::_BitMatrix::operator=(const DistribTree::_BitMatrix& right) 
{
  if (this == &right)
    return *this;

  delete[] _matrix;

  _size = right._size;
  _rows = right._rows;
  _columns = right._columns;

  _matrix = new unsigned char[_size];
  memcpy(_matrix, right._matrix, _size);

  return *this;
}

const DistribTree::_BitMatrix& DistribTree::_BitMatrix::operator~()
{
  for (unsigned rowX = 0; rowX < _rows; rowX++)
    for (unsigned colX = 0; colX < _columns; colX++)
      flipAt(rowX, colX);

  return *this;
}

bool operator==(const DistribTree::_BitMatrix& left, const DistribTree::_BitMatrix& right)
{
  for (unsigned i = 0; i<left._size-1; i++)
    if (left._matrix[i] != right._matrix[i]) return false;
  
  int spareBits = left._size*8 - left._columns*left._rows;
  
  if ((unsigned char)(left._matrix[left._size-1] >> spareBits) !=  (unsigned char)(right._matrix[left._size-1] >> spareBits))
    return false;
  
  return true;
} 

bool operator!=(const DistribTree::_BitMatrix& left, const DistribTree::_BitMatrix& right)
{
  return !(left == right);
} 

unsigned DistribTree::_BitMatrix::_bitsN(unsigned char byte)
{
  int count = 0;
  ((byte & _mask[0]) != 0) ? count++ : count;
  ((byte & _mask[1]) != 0) ? count++ : count;
  ((byte & _mask[2]) != 0) ? count++ : count;
  ((byte & _mask[3]) != 0) ? count++ : count;
  ((byte & _mask[4]) != 0) ? count++ : count;
  ((byte & _mask[5]) != 0) ? count++ : count;
  ((byte & _mask[6]) != 0) ? count++ : count;
  ((byte & _mask[7]) != 0) ? count++ : count;
  return count;
}


// ----- methods for class 'DistribTree::_BitMatrixList' ------
//
DistribTree::_BitMatrixList::_BitMatrixList(unsigned rowsN, unsigned colsN, unsigned silenceX, unsigned padX)
{
  _BitMatrix mat(rowsN, colsN);
  mat.resetAt(silenceX, (colsN-1)/2);
  mat.resetAt(padX, (colsN-1)/2);
  _bitMatLst.push_back(mat);
}

DistribTree::_BitMatrixList::_BitMatrixList(const _BitMatrix& bm, unsigned count)
{
  for (unsigned matX = 0; matX < count; matX++)
    _bitMatLst.push_back(bm);
}

DistribTree::_BitMatrixList::_BitMatrixList(const QuestionListList& questionList, LexiconPtr& phonesLexicon, unsigned contextLength)
{
  for (QuestionListList::Iterator itr(questionList); itr.more(); itr++) {
    _BitMatrix bmatrix(*itr, phonesLexicon, contextLength);  _bitMatLst.push_back(bmatrix);
  }
}

bool operator==(const DistribTree::_BitMatrixList& left, const DistribTree::_BitMatrixList& right){
  if (left.size() != right.size()) return false;
  for (DistribTree::_BitMatrixList::ConstIterator litr(left); litr.more(); litr++) {
    const DistribTree::_BitMatrix& lmatrix(*litr);
    bool matches = false;
    for (DistribTree::_BitMatrixList::ConstIterator ritr(right); ritr.more(); ritr++) {
      const DistribTree::_BitMatrix& rmatrix(*ritr);
      if (lmatrix == rmatrix) { matches = true; break; }
    }
    if (matches == false) return false;
  }  

  return true;
}

bool operator!=(const DistribTree::_BitMatrixList& left, const DistribTree::_BitMatrixList& right)
{
  return !(left == right);
}

const DistribTree::_BitMatrixList operator&(const DistribTree::_BitMatrixList& left, const DistribTree::_BitMatrixList& right)
{
  typedef DistribTree::_BitMatrix     _BitMatrix;
  typedef DistribTree::_BitMatrixList _BitMatrixList;

  _BitMatrixList bitMatrixList;
  for (_BitMatrixList::ConstIterator litr(left); litr.more(); litr++) {
    const _BitMatrix& lmatrix(*litr);
    for (_BitMatrixList::ConstIterator ritr(right); ritr.more(); ritr++) {
      const _BitMatrix& rmatrix(*ritr);
      _BitMatrix bmatrix(lmatrix & rmatrix);

      if (bmatrix.isValid() == false) continue;

      bool found = false;
      for (_BitMatrixList::Iterator bmitr(bitMatrixList); bmitr.more(); bmitr++)
	if (*bmitr == bmatrix) { found = true; break; }
      if (found == false) bitMatrixList.add(bmatrix);
    }
  }

  return bitMatrixList;
}

bool connect(const DistribTree::_BitMatrixList& left, const DistribTree::_BitMatrixList& right)
{
  typedef DistribTree::_BitMatrix     _BitMatrix;
  typedef DistribTree::_BitMatrixList _BitMatrixList;

  _BitMatrixList bitMatrixList;
  for (_BitMatrixList::ConstIterator litr(left); litr.more(); litr++) {
    const _BitMatrix& lmatrix(*litr);
    for (_BitMatrixList::ConstIterator ritr(right); ritr.more(); ritr++) {
      const _BitMatrix& rmatrix(*ritr);
      _BitMatrix matrix(lmatrix & rmatrix);
      if (matrix.isValid()) return true;
    }
  }

  return false;
}

const DistribTree::_BitMatrixList operator>>(const DistribTree::_BitMatrixList& bml, unsigned places)
{
  DistribTree::_BitMatrixList newlist(bml);  newlist >>= places;

  return newlist;
}

const DistribTree::_BitMatrixList& DistribTree::_BitMatrixList::operator>>=(unsigned places)
{
  for (_BitMatLstIterator itr = _bitMatLst.begin(); itr != _bitMatLst.end(); itr++) {
    _BitMatrix& bmatrix(*itr);  bmatrix >>= places;  bmatrix.setColumn(bmatrix.columnsN() - 1);
  }
  
  return *this;
}

const DistribTree::_BitMatrixList& DistribTree::_BitMatrixList::operator&=(const _BitMatrixList& right)
{
  _BitMatLst bitMatLst;
  for (_BitMatLstConstIterator itr = _bitMatLst.begin(); itr != _bitMatLst.end(); itr++) {
    const _BitMatrix tmatrix(*itr);
    for (_BitMatLstConstIterator ritr = right._bitMatLst.begin(); ritr != right._bitMatLst.end(); ritr++) {
      const _BitMatrix rmatrix(*ritr);
      _BitMatrix bmatrix(tmatrix & rmatrix);
      if (bmatrix.isValid() == false) continue;
      bool found = false;
      for (_BitMatLstConstIterator bmitr = bitMatLst.begin(); bmitr != bitMatLst.end(); bmitr++)
	if (*bmitr == bmatrix) { found = true; break; }
      if (found == false) bitMatLst.push_back(bmatrix);
    }
  }
  _bitMatLst = bitMatLst;

  return *this;
}

const DistribTree::_BitMatrixList& DistribTree::_BitMatrixList::operator|=(const _BitMatrixList& right)
{
  for (_BitMatLstConstIterator itr = right._bitMatLst.begin(); itr != right._bitMatLst.end(); itr++) {
    const _BitMatrix& bmatrix(*itr);  _bitMatLst.push_back(bmatrix);
  }
}

void DistribTree::_BitMatrixList::dump(const LexiconPtr& phoneLex) const
{
  for (_BitMatLstConstIterator itr = _bitMatLst.begin(); itr != _bitMatLst.end(); itr++) {
    const _BitMatrix& bmatrix(*itr);  bmatrix.dump(phoneLex);
  }
}

void DistribTree::_BitMatrixList::dumpEx() const
{
  for (_BitMatLstConstIterator itr = _bitMatLst.begin(); itr != _bitMatLst.end(); itr++) {
    const _BitMatrix& bmatrix(*itr);  bmatrix.dumpEx();
  }
}

bool DistribTree::_BitMatrixList::isValid() const
{
  if (_bitMatLst.size() == 0) return false;

  for (_BitMatLstConstIterator itr = _bitMatLst.begin(); itr != _bitMatLst.end(); itr++) {
    const _BitMatrix& bmatrix(*itr);
    if (bmatrix.isValid()) return true;
  }

  return false;
}


// ----- methods for class `DistribTree' -----
//
DistribTree::DistribTree(const String& nm, PhonesSetPtr& phonesSet, const int contextLength, const String& fileName,
			 char comment, const String& wb, const String& eos, bool verbose, bool sphinx)
  : _name(nm), _commentChar(comment), _phonesSet(phonesSet), _wb(wb), _eos(eos), _contextLength(contextLength), _sphinx(sphinx),
    _open(sphinx ? '(' : '{'), _close(sphinx ? ')' : '}')
{
  if (fileName != "") read(fileName);

  // Verbose = verbose;
  Verbose = false;
}

DistribTree::NodePtr DistribTree::
_newNode(PhonesSetPtr& ps, DistribTree* dt,
	 const String& nm, const String& ques,
	 const String& left, const String& right, const String& unknown, const String& leaf)
{
  return NodePtr(new Node(ps, dt, nm, ques, left, right, unknown, leaf));
}

DistribTree::NodePtr DistribTree::
_newNode(PhonesSetPtr& ps, DistribTree* dt,
	 const String& nm, QuestionListList& qlist,
	 const String& left, const String& right, const String& unknown, const String& leaf)
{
  return NodePtr(new Node(ps, dt, nm, qlist, left, right, unknown, leaf));
}

unsigned DistribTree::_maxIndex() const
{
  unsigned index = 0;
  for (_MapConstIterator itr = _map.begin(); itr != _map.end(); itr++) {
    const String& name((*itr).first);

    String::size_type pos1 = name.find_first_of("(");

    if (pos1 == String::npos) continue;

    String::size_type pos2 = name.find_first_of(")");

    pos1++;
    String indexString(name.substr(pos1, pos2 - pos1));
    unsigned idx = atoi(indexString);
    if (idx > index) index = idx;
  }

  return index;
}

const String& DistribTree::begin(const String& context)
{
  // cout << "Context: " << context << endl;
  // return node("ROOT-b")->leaf(context);
  return node("ROOT-0")->leaf(context);
}

const String& DistribTree::middle(const String& context)
{
  // return node("ROOT-m")->leaf(context);
  return node("ROOT-1")->leaf(context);
}

const String& DistribTree::end(const String& context)
{
  // return node("ROOT-e")->leaf(context);
  return node("ROOT-2")->leaf(context);
}

void DistribTree::read(const String& treePath, const String& state, unsigned offsetX, bool sphinx)
{
  _sphinx = sphinx;
  if (sphinx) {
    _open = '(';  _close = ')';
    _readSphinx(treePath, state, offsetX);
  } else {
    _open = '{';  _close = '}';
    _read(treePath);
  }
}

// read a distribution tree from a file
void DistribTree::_read(const String& fileName)
{
  if (fileName == "")
    jio_error("File name is null.");

  FILE* fp = fileOpen(fileName, "r");
  char  line[1000];
  if (fp == NULL)
    throw jio_error("Cannot open decision tree file '%s' for reading.\n", fileName.c_str());
 
  while (true) {
    char* p = line;
    list<String> entries;
    entries.clear();

    if (fscanf(fp,"%[^\n]", line) != 1) break;
    else fscanf(fp,"%*c");

    if ( line[0] == _commentChar) continue;

    for (p = line; *p != '\0'; p++)
      if (*p>' ') break; if (*p=='\0') continue;

    String lineString(line);
    size_t first = lineString.find_first_of(' ');
    String name(lineString.substr(0, first));

    size_t last = lineString.find_last_of(_close);
    String ques(lineString.substr(first + 1, last - first + 1));

    QuestionListList qlist(_parseCompoundQuestion(ques));

    splitList(lineString.substr(last + 2), entries);
    String left(entries.front());	entries.pop_front();
    String right(entries.front());	entries.pop_front();
    String unknown(entries.front());	entries.pop_front();
    String leaf(entries.front());	entries.pop_front();

    _map.insert(_MapType(name, _newNode(_phonesSet, this, name, qlist, left, right, unknown, leaf)));
  }

  fileClose(fileName, fp);
}

DistribTree::Question DistribTree::_parseQuestion(String& context)
{
  if (_sphinx)
    return _parseQuestionSphinx(context);
  else
    return _parseQuestionMillennium(context);
}

DistribTree::Question DistribTree::_parseQuestionMillennium(String& context)
{
  String::size_type first = 0;
  bool negate = false;
  if (context[0] == '!') {
    first = 1;  negate = true;
  }
  String::size_type pos	= context.find_first_of("=");
  String contextString  = context.substr(first, pos - first);
  int    position       = atoi(contextString.c_str());

  // String::size_type last= context.find(' ');
  // if (last == String::npos)
  //   last = context.find(_close);
  // if (last == String::npos)
  //   throw j_error("Context %s is ill formed", context.c_str());
  
  unsigned last = 0;
  while (context[last] != ' ' && context[last] != _close) last++;

  String phoneClass(context.substr(pos+1, last - pos - 1));

  if (Verbose)
    printf("%s : %d : %s\n", context.c_str(), position, phoneClass.c_str());

  context = context.substr(last);

  if (_phonesSet->hasPhones(phoneClass)) {
    // Question question(ctxt, phones, String("eps"), String("WB"), String("</s>"), negate);

    Question question(position, _phonesSet->find(phoneClass), String("eps"), String("WB"), String("</s>"), negate);
    cout << "Question :" << String(question) << endl;
    return question;
  } else {
    Question question(position, phoneClass, String("eps"), String("WB"), String("</s>"), negate);
    cout << "Question :" << String(question) << endl;
    return question;
  }
}

DistribTree::Question DistribTree::_parseQuestionSphinx(String& context)
{
  char* s = Cast<char*>(context.c_str());
    
  // skip leading whitespace
  for (; *s != '\0' && isspace((int)*s); s++);

  if (*s == '\0') {	// Nothing to parse
    context = String("");
    return Question();
  }

  bool negate = false;
  if (*s == '!') {
    negate = true;
    ++s;
    if (*s == '\0')
      throw j_error("question syntax error");
  }
    
  char* sp = strchr(s, ' ');
  if (sp == NULL)
    throw j_error("Expected space after question name in context %s", s);

  *sp = '\0';

  PhonesPtr phones(_phonesSet->find(s));
  
  s = sp+1;

  *sp = ' ';	// undo set to null

  // skip whitespace
  for (; *s != '\0' && isspace((int)*s); s++);

  int ctxt;
  if (s[0] == '-') {
    if (s[1] == '1') {
      ctxt = -1;
    }
    s += 2;
  } else if (s[0] == '0') {
    ctxt = 0;
    s++;
  }
  else if (s[0] == '1') {
    ctxt = 1;
    s++;
  }

  // skip trailing whitespace, if any
  for (; *s != '\0' && isspace((int)*s); s++);

  Question question(ctxt, phones, String("eps"), String("WB"), String("</s>"), negate);

  cout << "Question :" << String(question) << endl;

  context = String(s);

  return question;
}

DistribTree::QuestionList DistribTree::_parseConjunction(String& context)
{
  char* s = Cast<char*>(context.c_str());

  if (*s == '\0') {
    context = String("");
    return QuestionList();
  }

  // skip leading whitespace
  for (; *s != '\0' && isspace((int)*s); s++);
    
  if (*s == '\0') {
    context = String("");
    return QuestionList();
  }

  if (*s != _open)
    throw j_error("Expected %c before conjunction: '%s'", _open, s);

  ++s;	// skip left parenthesis

  for (; *s != '\0' && isspace((int)*s); s++);

  if (*s == '\0')
    throw j_error("No terms and close paren in conjunction %s\n", context.c_str());

  context = String(s);
  QuestionList questionList;
  while (context[0] != _close && context != "")  {
    Question question(_parseQuestion(context));
    questionList.add(question);
    s = Cast<char*>(context.c_str());
    for (; *s != '\0' && isspace((int)*s); s++);
    context = String(s);
  }
  s = Cast<char*>(context.c_str());

  if (*s != _close)
    throw j_error("Expected %s after conjunction", _close);

  s++;
  context = String(s);

  cout << "Question List :" << questionList.toString() << endl;

  return questionList;
}

DistribTree::QuestionListList DistribTree::_parseCompoundQuestion(String context)
{
  if (context[0] == _open && context[1] == _close)
    return QuestionListList();

  String initialContext(context);

  // remove leading white space
  while (context[0] == ' ')
    context.erase(0, 1);

  if (context == "")
    throw j_error("Empty string seen for composite question");
  if (context[0] != _open)
    throw j_error("Composite question does not begin with %c : '%s'\n", _open, context.c_str());
	
  // skip the left parentheses
  context.erase(0, 1);

  QuestionListList questionListList;
  do {
    QuestionList questionList(_parseConjunction(context));
    if (context == "")
      throw j_error("Error while parsing %s\n", initialContext.c_str());

    questionListList.add(questionList);
    while (context[0] == ' ')
      context.erase(0, 1);

    cout << "Current context " << context << endl;

  } while (context != "" && context[0] == _open);

  cout << "Question List List:" << questionListList.toString() << endl;

  return questionListList;
}

#include <string.h>

char *
read_line(char *buf,
	  size_t max_len,
	  unsigned *n_read,
	  FILE *fp)
{
    char *out;
    char *start;
    char *end;
    int read = 0;
    
    if (n_read != NULL)
	read = *n_read;

    do {
	out = fgets(buf, max_len, fp);
	read++;
    } while ((out != NULL) && (out[0] == '#'));

    if (strlen(buf) == (max_len-1)) {
	throw j_error("line %d may be truncated because it's longer than max_len %d",
	       read, max_len);
    }
    
    if (n_read != NULL)
	*n_read = read;
    
    if (out == NULL)
	return out;
    
    start = out;
    end = out + strlen(out) - 1;

    while (*start == ' ' || 
	   *start == '\t') {
	start++;
    }

    while ((end >= start) &&
	   (*end == ' ' || 
           *end == '\t' ||
	   *end == '\r' ||
	   *end == '\n'))
	end--;
    *(++end) = 0;
    
    memmove(out, start, end - start + 1);
    
    return out;
}

typedef unsigned uint32;
typedef double float64;
typedef float float32;

static const int NO_ID = -1;

void DistribTree::_readSphinx(const String& treePath, const String& state, unsigned offsetX)
{  
  String fileName(treePath + String("/") + state + String(".dtree"));

  char ln[4096], *s, str[512];
  FILE* fp = fileOpen(fileName, "r");
  char  line[1000];

  uint32 n_read, n_scan, n_node;
  uint32 i, node_id, node_id_y, node_id_n;
  float64 ent;
  float32 occ;
  unsigned mapX = _map.size();

  if (fp == NULL)
    throw jio_error("Cannot open decision tree file '%s' for reading.\n", fileName.c_str());

  read_line(ln, 4096, &n_read, fp);

  s = ln;
  sscanf(s, "%s%n", str, &n_scan);
  if (strcmp(str, "n_node") == 0) {
    s += n_scan;
    sscanf(s, "%u", &n_node);
  } else {
    throw j_error("Format error; expecting n_node");
  }

  while (read_line(ln, 4096, &n_read, fp)) {
    s = ln;

    sscanf(s, "%u%n", &node_id, &n_scan);
    s += n_scan;
    sscanf(s, "%s%n", str, &n_scan);
    s += n_scan;
    if (strcmp(str, "-") == 0) {
      node_id_y = NO_ID;
    } else {
      node_id_y = atoi(str);
    }
    sscanf(s, "%s%n", str, &n_scan);
    s += n_scan;
    if (strcmp(str, "-") == 0) {
      node_id_n = NO_ID;
    } else {
      node_id_n = atoi(str);
    }
    sscanf(s, "%le%n", &ent, &n_scan);
    s += n_scan;
    sscanf(s, "%e%n", &occ, &n_scan);
    s += n_scan;

    std::stringstream nameStream;
    if (node_id == 0)
      nameStream << state;
    else
      nameStream << state << String("(") << (mapX + node_id) << String(")");
    String name(nameStream.str());

    // Sphinx allows for compound questions and assigns
    //   1. Yes     --> _left
    //   2. No      --> _right
    if ((node_id_y != NO_ID) && (node_id_n != NO_ID)) {
      // this is an interior node
      QuestionListList qlist = _parseCompoundQuestion(String(s));

      std::stringstream leftStream;
      leftStream << state << String("(") << (mapX + node_id_n) << String(")");
      String left(leftStream.str());

      std::stringstream rightStream;
      rightStream << state << String("(") << (mapX + node_id_y) << String(")");
      String right(rightStream.str());

      _map.insert(_MapType(name, _newNode(_phonesSet, this, name, qlist, left, right, /*unknown=*/ "-", /*leaf=*/ "-")));
    } else {
      // this is a leaf node
      std::stringstream leafStream;
      leafStream << state << String("(") << offsetX << String(")");
      String leaf(leafStream.str());
      _map.insert(_MapType(name, _newNode(_phonesSet, this, name, /*ques=*/ "", /*left=*/ "-", /*right=*/ "-", /*unknown=*/ "-", leaf)));
      offsetX++;
    }
  }

  fileClose(fileName, fp);
}

// write a distribution tree to a file
void DistribTree::write(const String& fileName, const String& date)
{
  if (fileName == "")
    jio_error("File name is null.");

  FILE* fp = fileOpen(fileName, "w");

  fprintf(fp, "; -------------------------------------------------------\n");
  fprintf(fp, ";  Name            : %s\n", name().c_str());
  fprintf(fp, ";  Type            : Distribution Tree\n");
  fprintf(fp, ";  Number of Items : %d\n", _map.size());
  fprintf(fp, ";  Date            : %s\n", date.c_str());
  fprintf(fp, "; -------------------------------------------------------\n");

#if 1
  node("ROOT-b")->write(fp);
  node("ROOT-m")->write(fp);
  node("ROOT-e")->write(fp);
#else
  node("ROOT-0")->write(fp);
  node("ROOT-1")->write(fp);
  node("ROOT-2")->write(fp);
#endif

  fileClose(fileName, fp);
}

unsigned DistribTree::leavesN() const
{
  unsigned leafN = 0;
  for (_MapConstIterator itr = _map.begin(); itr != _map.end(); itr++) {
    const NodePtr& node((*itr).second);
    if (node->_leaf != "-")
      leafN++;
  }

  return leafN;
}

// construct a new phone lexicon without sos, eos, end, and backoff symbols
// for indexing into the bit matrices
LexiconPtr DistribTree::_extractPhoneLexicon(LexiconPtr& phoneLex)
{
  LexiconPtr newLex(new Lexicon("New Phone Lexicon"));
  for (Lexicon::Iterator pitr(phoneLex); pitr.more(); pitr++) {
    const String& phone(*pitr);
    if (phone.find("#") == String::npos &&
	phone.find("<s>") == String::npos &&
	phone.find("</s>") == String::npos &&
	phone.find("%") == String::npos &&
	phone.find("eps") == String::npos /* &&
					     phone.find("PAD") == String::npos */ ) {
      newLex->index(phone, true);
    }
  }
  return newLex;
}

/* Constructs the set of bitmaps associated with the tree. Each bitmatrix assigned to a leaf tells us which
phone is allowed at what position. Example:

     -101
  AH  001
  AX  110
      ...
  ZH  000

means that phone AH is allowed at the first position and phone AX is the centerphone for that leaf
allowed at position -1. ZH is not allowed anywhere.*/
const DistribTree::_BitmapList& DistribTree::buildMatrices(LexiconPtr& phonesLexicon)
{
  LexiconPtr extractedLexicon(_extractPhoneLexicon(phonesLexicon));
  unsigned silenceX = extractedLexicon->index("SIL");
  unsigned padX     = extractedLexicon->index("PAD");

  cout << "Distribution tree context length: " << _contextLength << endl;
  _BitMatrixList bml(extractedLexicon->size(), 2*_contextLength+1, silenceX, padX);

  cout << "Parsing the ROOT-b subtree ..." << endl;
#if 1
  const NodePtr& bnode(node("ROOT-b"));
#else
  const NodePtr& bnode(node("ROOT-0"));
#endif
  parseTree(bnode, bml, extractedLexicon);

  cout << "Parsing the ROOT-m subtree ..." << endl;
#if 1
  const NodePtr& mnode(node("ROOT-m"));
#else
  const NodePtr& mnode(node("ROOT-1"));
#endif
  parseTree(mnode, bml, extractedLexicon);

  cout << "Parsing the ROOT-e subtree ..." << endl;
#if 1
  const NodePtr& enode(node("ROOT-e"));
#else
  const NodePtr& enode(node("ROOT-2"));
#endif
  parseTree(enode, bml, extractedLexicon);

  cout << "Found " << _leafmatrices.size() << " leaves on the tree." << endl;
  return _leafmatrices;
}

String DistribTree::getDecisionPhone(String decisionPhone)
{
  String::size_type bracketPos    = decisionPhone.find("{");
  if (bracketPos != String::npos)
    decisionPhone.erase(decisionPhone.begin() + bracketPos, decisionPhone.begin() + bracketPos + 1);
  String::size_type underScorePos = decisionPhone.find(":");
  if (underScorePos != String::npos)
    decisionPhone.erase(decisionPhone.begin() + underScorePos, decisionPhone.end());
  String::size_type endBracketPos = decisionPhone.find("}");
  if (endBracketPos != String::npos)
    decisionPhone.erase(decisionPhone.begin() + endBracketPos, decisionPhone.end());

  return decisionPhone;
}

void DistribTree::parseTree(const NodePtr& node, _BitMatrixList& bml, LexiconPtr& phonesLexicon)
{
  node->_bitMatrixList = bml;
  QuestionListList qlistTilde(node->_qlist.negate());
  if (Verbose) {
    cout << "Entering parseTree for node " << node->name() << endl;
    cout << "Question:  " << node->_qlist.toString() << endl;
    cout << "~Question: " << qlistTilde.toString() << endl;
  }

  if (node->left() == "-" && node->right() == "-" && node->unknown() == "-") {
    if (node->leaf() == "-")
      throw jconsistency_error("Leaf is empty.");

    // reached a leaf; add bit matrix list to leaf map
    if (bml.isValid() == false) {
      // throw j_error("Invalid bitmap list for leaf %s", node->leaf().c_str());
      cout << "Invalid bitmap list for leaf " << node->name() << endl;
      bml.dump(phonesLexicon);
      return;
    }

    if (Verbose) {
      cout << "Leaf: " << node->leaf() << " Bit Matrix List:" << endl; 
      bml.dump(phonesLexicon);
    }

    // add valid bit matrix list to leaf node
    _leafmatrices[node->leaf()] = bml;
    return;
  }

  // convert question list to bit matrix list
  _BitMatrixList quesBitMatrixList(node->_qlist, phonesLexicon);
  _BitMatrixList quesTildeBitMatrixList(qlistTilde, phonesLexicon);
  if (Verbose) {
    cout << "Question bit matrix list:" << endl;
    quesBitMatrixList.dump(phonesLexicon);

    cout << "~Question bit matrix list:" << endl;
    quesTildeBitMatrixList.dump(phonesLexicon);
  }

  // create 'no' bit matrix list and propagate down tree
  _BitMatrixList nlist(quesTildeBitMatrixList & bml);
  //  if (nlist.isValid()) {
    if (Verbose) {
      cout << "Going left at node " << node->name() << " to node " << node->left() << endl;
      cout << "'No' Bit Matrix List: " << endl;
      nlist.dump(phonesLexicon);
    }
    if (node->left() != "-") {
      const NodePtr& nnode(DistribTree::node(node->left()));
      assert(nnode->name() == node->left());
      parseTree(nnode, nlist, phonesLexicon);
    }
    //  }

  // create 'yes' bit matrix list and propagate down tree
  _BitMatrixList ylist(quesBitMatrixList & bml);
  // if (ylist.isValid() == false) {
  //   cout << "'Yes' Bit Matrix List: " << endl;    
  //   ylist.dump(phonesLexicon);
  //   throw j_error("ylist must be valid");
  // }
  if (Verbose) {
    cout << "Going right at node " << node->name() << " to node " << node->right() << endl;
    cout << "'Yes' Bit Matrix List: " << endl;
    ylist.dump(phonesLexicon);
  }
  if (node->right() != "-") {
    const NodePtr& ynode(DistribTree::node(node->right()));
    assert(ynode->name() == node->right());
    parseTree(ynode, ylist, phonesLexicon);
  }
}
