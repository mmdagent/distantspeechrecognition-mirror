//
//                               Millenium
//                   Automatic Speech Recognition System
//                                 (asr)
//
//  Module:  asr.dict
//  Purpose: Dictionary module.
//  Author:  Fabian Jakobs.

#include <iostream>
#include <algorithm>
#include <list>
#include <stdio.h>
#include "common/jexception.h"
#include "dictionary/tags.h"
#include "common/mlist.h"


// ----- methods for class `Tags' -----
//
// displays the contents of a tags set
String Tags::puts() {
  String repr("");
  for (TagsListConstIter iter=_tags.begin(); iter!=_tags.end(); iter++) {
    repr = repr + " " + *iter;
  }
  return repr;
}
 
// add new tag(s) to a tags-set
void Tags::add(const String& tag) {
  _tags.push_back(tag);
}
 
// delete tag(s) from a tags-set
void Tags::erase(const String& tag) {
  _tags.erase(find(_tags.begin(), _tags.end(), tag));
}

// returns the index of tags
signed int Tags::index(const String& s) {
  int size = _tags.size();
  for (int i = 0; i < size; i++) {
    if (_tags[i] == s) return i;
  }
  return -1;
}

bool Tags::hasTag(const String& ps) const {
  int size = _tags.size();
  for (int i = 0; i < size; i++) {
    if (_tags[i] == ps) return true;
  }
  return false;
}
 
// read a tag-set from a file
int Tags::read(const String& filename) {
  FILE *fp;
  char  line[1000];

  if ((fp=fopen(filename.c_str(),"r"))==NULL) {
    throw jio_error("Can't open tags file '%s' for reading.\n",
                    filename.c_str());
  }

  while (1) {
    char *p = line;
    list<String> tags;
    tags.clear();

    if ( fscanf(fp,"%[^\n]",line)!=1) break;
    else fscanf(fp,"%*c");

    if ( line[0] == commentChar) continue;

    for (p=line; *p!='\0'; p++)
    if (*p>' ') break; if (*p=='\0') continue;

        splitList(line, tags);
        for (list<String>::iterator iter=tags.begin(); iter != tags.end(); ++iter) {
            add(*iter);
        }
  }
  fclose(fp);
  return _tags.size();

}

// write a set of tags into a open file
void Tags::writeFile(FILE* fp) {
  for (TagsListConstIter iter=_tags.begin(); iter != _tags.end(); ++iter) {
    fprintf(fp,"%s ",iter->c_str());
  }
}
  
// write a tag-set into a file
void Tags::write(const String& filename) {
  FILE *fp;

  if ((fp=fopen(filename.c_str(),"w"))==NULL) {
    throw jio_error("Can't open tags file '%s' for writing.\n",
                    filename.c_str());
  }

  fprintf( fp, "%c -------------------------------------------------------\n",
             commentChar);
  fprintf( fp, "%c  Name            : %s\n",  commentChar,
             _name.c_str());
  fprintf( fp, "%c  Type            : Tags\n", commentChar);
  fprintf( fp, "%c  Number of Items : %d\n",  commentChar,
             _tags.size());
  fprintf( fp, "%c  Date            : %s", commentChar,
             dateString());
  fprintf( fp, "%c -------------------------------------------------------\n",
             commentChar);
  writeFile(fp);
  fclose(fp);
}
 
