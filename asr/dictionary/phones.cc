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


#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <list>
#include "dictionary/phones.h"
#include "common/mlist.h"


// ----- methods for class `Phones' -----
//
// returns the contents of a set of phone-sets
String Phones::puts() {
  String repr("");

  for (_ConstIterator iter(_phones); iter.more(); iter++)
    repr = repr + " " + *iter;

  return repr;
};

// add new phone(s) to a phone-set
void Phones::add(const String& phones) {
  _phones.add(phones, phones);
};

// returns true is phone is in this class
bool Phones::hasPhone(const String& phone) {
  return _phones.isPresent(phone);
}

// read a phone-set from a file
int Phones::read(const String& fileName)
{
  FILE* fp = fileOpen(fileName, "r");
  char  line[1000];

  if (fp == NULL)
    throw jio_error("Can't open phones file '%s' for reading.\n", fileName.c_str());
 
  while (true) {
    char *p = line;
    list<String> phones;
    phones.clear();

    if ( fscanf(fp,"%[^\n]",line)!=1) break;
    else fscanf(fp,"%*c");

    if ( line[0] == _commentChar) continue;

    for (p=line; *p!='\0'; p++)
      if (*p>' ') break; if (*p=='\0') continue;

    splitList(line, phones);
  
    for (list<String>::iterator iter=phones.begin(); iter != phones.end(); ++iter)
      add(*iter);
  }
  fileClose(fileName, fp);  

  return _phones.size();
}

int phonesCharFunc(PhonesPtr superset, PhonesPtr subset, char* charFunc)
{
  int ill=0;

  memset( charFunc, '-', superset->phonesN());
  charFunc[superset->phonesN()]='\0';

  for (unsigned x = 0; x < subset->phonesN(); x++) {
    if ( superset->hasPhone((*subset)[x]) ) {
      unsigned i = superset->index((*subset)[x]);
      charFunc[i]='+';
    } else ill++;
  }

  return ill;
}

// write a set of phone-sets into a open file
void Phones::writeFile(FILE* fp)
{
  for (_ConstIterator iter(_phones); iter.more(); iter++)
    fprintf(fp, "%s ", (*iter).c_str());
}

/* write a phone-set from a file */
// FIXME in contrast to the old implementation this function 
// give the Phones in alphabetical order 
void Phones::write(const String& filename) {
  FILE *fp = fopen(filename.c_str(),"w");

  if (fp == NULL)
    throw jio_error("Can't open phones file '%s' for writing.\n",
                    filename.c_str());

  fprintf( fp, "%c -------------------------------------------------------\n",
	   _commentChar);
  // FIXME should we give every object a unique name? 
  // I think it's a relict from TCL
  fprintf( fp, "%c  Name            : %s\n",  _commentChar,
	   _name.c_str());
  fprintf( fp, "%c  Type            : Phones\n", _commentChar);
  fprintf( fp, "%c  Number of Items : %d\n",  _commentChar,
	   _phones.size());
  fprintf( fp, "%c  Date            : %s", _commentChar,
	   dateString());
  fprintf( fp, "%c -------------------------------------------------------\n",
	   _commentChar);

  writeFile(fp);
  fprintf(fp,"\n");
  fclose(fp);
}
 

// ----- methods for class `PhonesSet' -----
//
// displays the contents of a set of phone-sets
String PhonesSet::puts() {
  String repr("");

#ifdef JMcD
  for (_ConstIterator iter(_phonesSet); iter.more(); iter++)
    repr = repr + (*iter) + " ";
#endif

  return repr;
}

// add new phone-set to a set of phones-set
void PhonesSet::add(const String& phoneName, const PhonesPtr& phones) {
  _phonesSet.add(phoneName, phones);
}

void PhonesSet::add(const String& phoneName, const list<String>& phones) {
  Phones* p = new Phones(phoneName);
  for (list<String>::const_iterator iter=phones.begin(); iter!=phones.end(); ++iter) {
    p->add(*iter);
  }
  _phonesSet.add(phoneName, p);
}

// delete phone-set(s) from a set of phone-sets
void PhonesSet::remove(const String& phoneName) {
#ifdef JMcD
  _phonesSet.erase(phoneName);
#endif
}


// read a set of phone-sets from a file
int PhonesSet::read(const String& filename)
{
  FILE* fp = fileOpen(filename,"r");
  char  line[4000];
  int   n  = 0;

  if (fp == NULL)
    throw jio_error("Can't open phone set file '%s' for reading.\n",
		    filename.c_str());

  while (true) {
    char* p = NULL;
    list<String> items;
    items.clear();

    if ( fscanf(fp,"%[^\n]",&(line[0]))!=1) break;
    else fscanf(fp,"%*c");

    if ( line[0] == _commentChar) continue;

    for (p = line; *p != '\0'; p++)
      if (*p > ' ') break;

    if (*p=='\0') continue;

    splitList(line, items);

    String name = items.front();
    items.pop_front();
    add(name, items);

    n++;
  }
  fileClose(filename, fp);

  return n;
}

// write a set of phone-sets into a file
void PhonesSet::write(const String& filename)
{
  FILE* fp = fileOpen(filename, "w");

  if (fp == NULL)
    throw jio_error("Can't open phone set file 'i%s' for writing.\n",
		    filename.c_str());

  fprintf( fp, "%c -------------------------------------------------------\n",
	   _commentChar);
  fprintf( fp, "%c  Name            : %s\n",		_commentChar,
	   _name.c_str());
  fprintf( fp, "%c  Type            : PhonesSet\n",	_commentChar);
  fprintf( fp, "%c  Number of Items : %d\n",		_commentChar,
	   _phonesSet.size());
  fprintf( fp, "%c  Date            : %s",		_commentChar,
	   dateString());
  fprintf( fp, "%c -------------------------------------------------------\n",
	   _commentChar);

  for (_ConstIterator iter(_phonesSet); iter.more(); iter++) {
    fprintf(fp,"%-20s ", (*iter)->name().c_str());
    (*iter)->writeFile(fp);
    fprintf(fp, "\n");
  }
  fileClose(filename, fp);
}

// returns true is phone is in this class
bool PhonesSet::hasPhones(const String& phones) {
  return _phonesSet.isPresent(phones);
}

