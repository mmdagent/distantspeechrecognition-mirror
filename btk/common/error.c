
#include "common.h"
#include <stdarg.h>
#include <string.h>

char errorRCSVersion[]= "@(#)1$Id: error.c 1075 2005-03-02 15:25:50Z fabian $";

/* ------------------------------------------------------------------------
    Local Variables
   ------------------------------------------------------------------------ */

#define MAXERRACCU 10000
char errorMsg[MAXERRACCU];
static char *errorFileName;
static int   errorLine;
static int   errorType;
static int   errorMode;

/* ------------------------------------------------------------------------
    messageHandlerPtr
   ------------------------------------------------------------------------ */

MsgHandler* msgHandlerPtr(char* file, int line, int type, int mode)
{
  errorFileName = file;
  errorLine     = line;
  errorType     = type;
  errorMode     = mode;
  return &msgHandler;
}

/* ------------------------------------------------------------------------
    msgHandler
   ------------------------------------------------------------------------ */

int msgHandler( char *format, ... )
{

  va_list  ap; 
  char     buf[MAXERRACCU] = "", *format2 = format;

  if ( format) { 
    va_start(ap,format);
    vsnprintf(buf, MAXERRACCU, format2, ap); 
    va_end(ap);
  }
  
  snprintf(errorMsg, MAXERRACCU, "(%s,%d): %s", errorFileName, errorLine, buf);
  return 0;
}

char* getErrMsg(void) {
  return errorMsg;
}

/* ------------------------------------------------------------------------
    Error_Init
   ------------------------------------------------------------------------ */

int Error_Init(void)
{
  static int errorInitialized = 0;

  if (! errorInitialized) {
    errorInitialized = 1;
    sprintf(errorMsg, "kein Fehler");
  }
  return 0;
}
