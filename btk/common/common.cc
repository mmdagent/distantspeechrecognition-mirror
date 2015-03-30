//
//                           Speech Front End for Beamforming
//
//  Module:  btk.common
//  Purpose: Common operations.
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


#include "common/mlist.h"
#include <ctype.h>
#include <list>
#include "common/jexception.h"

using namespace std;

#ifdef WINDOWS
  #include <fcntl.h>

  /* set file open (fopen) mode to binary and not to text */
  int _fmode = _O_BINARY;

#endif

void hello() {
  printf( "\n");
  printf( "                              Speech Front End                      \n");
  printf( "\n");
}

void splitList(const String& line, std::list<String>& out)
{
  String tmp = "";
  bool inWord = false;
  bool writeWord = false;
  int  parendepth = 0;

  String::const_iterator iter = line.begin();
  while (iter != line.end()) {

    if (!inWord) {
      if (!isspace(*iter)) inWord=true;
    }
    if (inWord) {
      if (parendepth == 0) {
	if (isspace(*iter)) {
	  inWord = false;
	  writeWord = true;
	} else if (*iter == '{') {
	  parendepth++;
	  iter++;
	  continue;
	} else { // iter not space and not '{'
	  tmp += *iter;
	}
      } else if (parendepth > 0) {
	if (*iter == '{') parendepth++;
	if (*iter == '}') parendepth--;
	if (parendepth == 0) {
	  inWord = false;
	  writeWord = true;
	} else {
	  tmp += *iter;
	}
      }
      if (parendepth < 0)
	throw jparse_error("'}' expected in line %s!", line.c_str());
    }
    if (writeWord) {
      out.push_back(tmp);
      tmp = "";
      writeWord = false;
    }
    iter++;
  }
  if (inWord) {
    out.push_back(tmp);
  }
}

char* dateString() { time_t t=time(0); return (ctime(&t)); }


// fileOpen: open a file for reading/writing
//
FILE* fileOpen(const char* fileName, const char* mode)
{
  int   retry = 20;
  int   count = 0;
  int   l     = strlen(fileName);
  int   pipe  = 0;
  FILE* fp    = NULL;
  char itfBuffer[500];

  // if (strchr(mode,'w')) itfMakePath(fileName,0755);

  if        (! strcmp( fileName + l - 2, ".Z")) {
    if      (! strcmp( mode, "r")) sprintf(itfBuffer,"zcat %s",       fileName);
    else if (! strcmp( mode, "w")) sprintf(itfBuffer,"compress > %s", fileName);
    else
      throw jio_error("Can't popen using mode.\n");

    pipe = 1;

  } else if (! strcmp( fileName + l - 3, ".gz")) {
    if      (! strcmp( mode, "r")) sprintf(itfBuffer,"gzip -d -c  '%s'", fileName);
    else if (! strcmp( mode, "w")) sprintf(itfBuffer,"gzip -c >  '%s'",  fileName);
    else if (! strcmp( mode, "a")) sprintf(itfBuffer,"gzip -c >> '%s'",  fileName);
    else
      throw jio_error("Can't popen using mode.\n");

    pipe = 1;

  } else if (! strcmp( fileName + l - 4, ".bz2")) {
    if      (! strcmp( mode, "r")) sprintf(itfBuffer,"bzip2 -cd    '%s'", fileName);
    else if (! strcmp( mode, "w")) sprintf(itfBuffer,"bzip2 -cz >  '%s'", fileName);
    else if (! strcmp( mode, "a")) sprintf(itfBuffer,"bzip2 -cz >> '%s'", fileName);
    else
      throw jio_error("Can't popen using mode.\n");

    pipe = 1;
  }

  while (count <= retry) {
    if (! (fp = ( pipe) ? popen( itfBuffer, mode) :
                          fopen( fileName,  mode))) {
      sleep(5); count++;
    }
    else break;
  }
  if ( count > retry)
    throw jio_error("'fileOpen' failed for %s.\n", fileName);

  return fp;
}

// fileClose:  close previously openend file
//
void fileClose(const char* fileName, FILE* fp)
{
  int l = strlen(fileName);

  fflush( fp);

  if      (! strcmp( fileName + l - 2, ".Z"))   pclose( fp);
  else if (! strcmp( fileName + l - 3, ".gz"))  pclose( fp);
  else if (! strcmp( fileName + l - 4, ".bz2")) pclose( fp);
  else                                          fclose( fp);
}

static int        _line;
static const char* _file;

int _setErrLine(int line, const char* file)
{
  _line = line; _file = file;
  
  return 1;
}

void _warnMsg(const char* message, ...)
{
   va_list ap;    /* pointer to unnamed args */
   FILE* f = stdout;

   fflush(f);    /* flush pending output */
   va_start(ap, message);
   fprintf(f, " >>> Warning: ");
   vfprintf(f, message, ap);
   va_end(ap);
   fprintf(f,"\n");
   fprintf(f," >>>          at line %d of \'%s\'\n", _line, _file);
   fprintf(f," >>> Continuing ... \n\n");
}


