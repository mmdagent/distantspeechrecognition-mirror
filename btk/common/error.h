#ifdef __cplusplus
extern "C" {
#endif

#ifndef _ERROR
#define _ERROR

#define INFO     msgHandlerPtr(__FILE__,__LINE__,-1,0)
#define WARN     msgHandlerPtr(__FILE__,__LINE__,-2,0)
#define SWARN    msgHandlerPtr(__FILE__,__LINE__,-2,1)
#define ERROR    msgHandlerPtr(__FILE__,__LINE__,-3,0)
#define SERROR   msgHandlerPtr(__FILE__,__LINE__,-3,1)
#define FATAL    msgHandlerPtr(__FILE__,__LINE__,-4,0)

#define MSGCLEAR msgHandlerPtr(__FILE__,__LINE__,0,2)
#define MSGPRINT msgHandlerPtr(__FILE__,__LINE__,0,3)

typedef int MsgHandler(   char*, ... );

extern MsgHandler*   msgHandlerPtr(char* file, int line, int type, int mode);
extern int           msgHandler(   char*, ... );
extern int           Error_Init(void);
extern char*         getErrMsg(void);

#endif



#ifdef __cplusplus
}
#endif
