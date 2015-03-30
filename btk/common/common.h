//                              -*- C++ -*-
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


#ifdef __cplusplus
extern "C" {
#endif

#ifndef _common_h_
#define _common_h_

#include <btk.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* This is needed for glibc 2.2.2 */
#ifdef LINUX
#include <float.h>
#endif

/* (fuegen) assuming that all platforms except windows have unistd.h
 * I have removed all HAVE_UNISTD_H defines in other files */
#ifndef WINDOWS
	#include <unistd.h>
#else
        #include "rand48/rand48.h"
	#include <float.h>
	#undef  ERROR
	#include <windows.h>
	#undef  ERROR
	#define ERROR msgHandlerPtr(__FILE__,__LINE__,-3,0)
	#undef  MEM_FREE
#endif

extern FILE *STDERR;
extern FILE *STDOUT;
extern FILE *STDIN;

#include "error.h"

#define MAX_NAME 256

/* ------------------------------------------------------------------------
    Common macros:
   ------------------------------------------------------------------------ */
#define streq(s1,s2)    (strcmp (s1,s2) == 0)     /* string equality        */
#define streqc(s1,s2)   (strcasecmp (s1,s2) == 0) /* not case sensitive     */
#define strneq(s1,s2)   (strcmp (s1,s2) != 0)     /* string inequality      */
#define ABS(x)          ((x)>=0?(x):-(x))         /* absolute value         */
#define SGN(s1)		(s1==0 ? 0 : (s1>0 ? 1:(-1) )) /* signum function   */
#define MIN(a,b)        (((a)<(b))?(a):(b))            /* minimum           */
#define MAX(a,b)        (((a)>(b))?(a):(b))            /* maximum           */

#ifndef TRUE
#define TRUE  1
#endif
#ifndef FALSE
#define FALSE 0
#endif

#ifdef WINDOWS
#define popen(x,y)         _popen (x, y)
#define pclose(x)          _pclose (x)
#define sleep(x)            Sleep (1000*x)
#define usleep(x)           Sleep (x/1000)
#define gethostname(x,y)   -1
#define getgid()            0
#define getuid()            0
#define strncasecmp(x,y,z) _stricmp (x,y)
#define strcasecmp(x,y)     strcmp (x,y)
#define isnan(x)           _isnan (x)
#define finite(x)          _finite (x)

#define M_PI 3.14159265358979323846

/* pragma directives */
#ifndef _DEBUG
#define DISABLE_WARNINGS
#pragma warning( disable : 4305 4244 4101 )
#else
#define DISABLE_WARNINGS
#pragma warning( disable : 4305 4244 )
#endif /* _DEBUG */
#endif /* WINDOWS */

#

#ifdef HAVE_MALLOC_H
#include <malloc.h>
#else
#if __APPLE__
#define memalign(x, y) malloc(y)
#endif /* __APPLE__ */
#ifdef WINDOWS
#define memalign(x,y) malloc (y)
#endif /* WINDOWS */
#endif

#endif /* _common */

void hello();
FILE* fileOpen(const char* fileName, const char* mode);
void fileClose(const char* fileName, FILE* fp);

#ifdef __cplusplus
}
#endif
