
#ifndef _fopen_h_
#define _fopen_h_

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

FILE* fileOpen(const char* fileName, const char* mode);
void fileClose(const char* fileName, FILE* fp);

#endif
