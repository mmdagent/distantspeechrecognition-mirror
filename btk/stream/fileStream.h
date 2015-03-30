//
//                                Millenium
//                   Automatic Speech Recognition System
//                                  (asr)
//
//  Module:  asr.common
//  Purpose: Common operations for files.
//  Author:  Kenichi Kumatani
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

#ifndef _fileStream_h_
#define _fileStream_h_

#include <stdio.h>
#include "common/mach_ind_io.h"
#include "common/mlist.h"
#include "common/refcount.h"

class FileHandler {
 public:
  FileHandler( const String &filename, const String &mode );
  ~FileHandler();
  int readInt();
  String readString( );
  void writeInt( int val );
  void writeString( String val );

 private:
  FILE* _fp;
  String _filename;
  //String _readline;
  char _buf[FILENAME_MAX];  
};

typedef refcount_ptr<FileHandler>  FileHandlerPtr;

#endif /* #ifndef _fileStream_h_ */

