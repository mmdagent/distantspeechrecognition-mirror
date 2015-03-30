//========================= Official Notice ===============================
//
// "This software was developed at the National Institute of Standards
// and Technology by employees of the Federal Government in the course of
// their official duties. Pursuant to Title 17 Section 105 of the United
// States Code this software is not subject to copyright protection and
// is in the public domain.
//
// The NIST Data Flow System (NDFS) is an experimental system and is
// offered AS IS. NIST assumes no responsibility whatsoever for its use
// by other parties, and makes no guarantees and NO WARRANTIES, EXPRESS
// OR IMPLIED, about its quality, reliability, fitness for any purpose,
// or any other characteristic.
//
// We would appreciate acknowledgement if the software is used.
//
// This software can be redistributed and/or modified freely provided
// that any derivative works bear some notice that they are derived from
// it, and any modified versions bear some notice that they have been
// modified from the original."
//
//=========================================================================
#ifndef _MK3LIB_H_
#define _MK3LIB_H_

#ifdef HAVE_LIBMARKIV

#include "driver/mk4_common.h"
#include "driver/mk4lib_int.h"
#include <stdio.h>

/****************************************/
// 'mk4array' type

/**********/
// Some functions authorized values 

enum { // Authorized 'speed' values
  mk4array_speed_22K = 22050,
  mk4array_speed_44K = 44100,
};

enum { // mk4 'status' ordered values
  mk4array_status_wait_comminit,   // Need Communication Initialization
  mk4array_status_wait_initparams, // Need to check/set speed,slave, ...
  mk4array_status_not_capturing,   // Ready to capture
  mk4array_status_capturing,       // Capturing
};

enum { // True / False
  mk4_true   = 99,
  mk4_false  = 101,
};
/* Some functions will use and/or return an 'int' that is to be compared
   to these value to check the yes/no value/status */

/*
  ******************** WARNING ********************

   It is very important to understand the difference between 
   a "data frame" and a "network frame".

   Per the Mk3 design:

   1x "network frame" = 5 x "data frame"

   therfore only 1 timestamp can be given per 5 data frames.
   Since the creation function allows the user to specifiy
   a "requested data frames per buffer" value, if this value is
   under 5 (authorized value), some timestamps will be identical,
   and the library will _not_ consider this a problem.

*/

/********************/
// Public functions (by 'status' order)

/**********/ // 0) before anything

/*
 'mk4array_create'
   arg 1 : int        the request amount of memory to use for the internal
                      storage of the collected data (0 will make the program
		      use its default value)
   arg 2 : int        when receiving data, this value allow the user to
                      request a certains amount of data frames to be delivered
		      to the 'get' functions                      
   arg 3 : mk4error*  in order to set the error state of the function (if any)
   return: mk4array*  newly created 'mk4array' (to be deleted using
                      'mk4array_delete')
 The functions create and returns an opaque handler for a 'mk4array'
 but does not initialize the connection
*/
mk4array* mk4array_create(int megabytes, int requested_data_frames_per_buffer, mk4error *err);
 
/**********/ // 1) 'mk4array_status_wait_comminit'

/*
 'mk4array_comminit'
   arg 1 : mk4array*   'mk4array' handle
   arg 2 : const char* the IP address of the Mk3 to connect to
   arg 3 : mk4error*   error state of function (if any)
   return: void
 Create the communication socket between the local system and the Mk3
 (and to 'check' that the connection is working order the Mk3 to stop capture)
*/
void mk4array_comminit(mk4array *it, const char *dip, mk4error *err);

/**********/ // 2) 'mk4array_status_wait_initparams'

/*
 'mk4array_ask_speed'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : mk4error*  error state of function (if any)
   return: int        one of 'mk4array_speed_22K' or 'mk4array_speed_44K'
 Query the Mk3 (if no value was set by the user) to check the current value 
 of the sampling rate (== capture 'speed')
*/
int mk4array_ask_speed(mk4array *it, mk4error *err);

/*
 'mk4array_set_speed'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : int        one of 'mk4array_speed_22K' or 'mk4array_speed_44K'
   arg 3 : mk4error*  error state of function (if any)
   return: void
 Set the Mk3 sampling rate
*/
void mk4array_set_speed(mk4array *it, int speed, mk4error *err);

/*
 'mk4array_ask_slave'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : mk4error*  error state of function (if any)
   return: int        one of 'mk4_true' or 'mk4_false'
 Query the Mk3 (if no value was set by the user) to check the current value 
 of the "slave" mode ("true", the Mk3 clock is slave to an external clock, ...)
*/
int mk4array_ask_slave(mk4array *it, mk4error *err);

/*
 'mk4array_set_slave'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : int        one of 'mk4_true' or 'mk4_false'
   arg 3 : mk4error*  error state of function (if any)
   return: void
 Set the "slave" mode (warning: can not check that there is an external clock)
*/
void mk4array_set_slave(mk4array *it, int slave, mk4error *err);

/*
 'mk4array_ask_id'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : mk4error*  error state of function (if any)
   return: int
   Return the value of the 'id' of the Mk3 (requested during comminit)
*/
int mk4array_ask_id(mk4array *it, mk4error *err);

/*
 'mk4array_ask_promnb'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : char*      address of a char[9] (at least: 8 for info + '\0')
   arg 3 : mk4error*  error state of function (if any)
   return: void
 Set 'arg 2' with the value of the 'prom_nb' (requested during comminit)
*/
void mk4array_ask_promnb(mk4array *it, char *promnb, mk4error *err);

/*
 'mk4array_initparams'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : mk4error*  error state of function (if any)
   return: void
 Check that 'speed' and 'slave' are set then advance automaton
*/
void mk4array_initparams(mk4array *it, mk4error *err);

/*
 'mk4array_initparams_wp' ("wp": with paramaters)
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : int        one of 'mk4array_speed_22K' or 'mk4array_speed_44K'
   arg 3 : int        one of 'mk4_true' or 'mk4_false'
   arg 4 : mk4error*  error state of function (if any)
   return: void
 Set 'speed' and 'slave', then advance automaton (useful as it allows to 
 initialize all the required parameters in one function. Mostly used when
 one knows the settings under which the capture will be performed)
*/
void mk4array_initparams_wp(mk4array *it, int speed, int slave, mk4error *err);

/**********/ // 3) 'mk4array_status_not_capturing'

/*
  'mk4array_wait_nfnbr'
    arg 1 : mk4array* 'mk4array' handle
    arg 2 : int       network data frame number to wait for
    arg 3 : mk4error* error state of function (if any)
  Wait until 'nfnbr' (network data frame number) is seen before allowing for
  next step of capture to start. Useful when trying to synchronize master
  and slave arrays; insure the capture on all start at the same nfnbr.
  Note: next step (drop X first frames) will start at 'nfnbr + 1'
*/
void mk4array_wait_nfnbr(mk4array *it, int nfnbr, mk4error *err);

/*
 'mk4array_fix_drop_X_first_frames'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : int        X value (in number of samples)
   arg 3 : mk4error*  error state of function (if any)
   return: void
 'fix': Set the internal value for dropping a certain number of frames
 Note: a problem has been noted that might require this when starting capture
 Note: the 'X' value used should be a multiple of 5 (since we collect network
 frames, not data frames from the hardware), or the closest value ad
*/
void mk4array_fix_drop_X_first_frames(mk4array *it, int X, mk4error *err);

/*
 'mk4array_fix_one_sample_delay'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : int        one of 'mk4_true' or 'mk4_false'
   arg 3 : mk4error*  error state of function (if any)
   return: void
 'fix': Set the internal value for correcting a one sample delay between 
 odd channels and even channels
*/
void mk4array_fix_one_sample_delay(mk4array *it, int truefalse, mk4error *err);

/*
 'mk4array_display_warnings'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : int        one of 'mk4_true' or 'mk4_false'
   arg 3 : mk4error*  error state of function (if any)
   return: void
 'warn': Set the internal value for printing to 'stderr' when an specific
 situation is encountered (buffer overflow, lost packets)
 odd channels and even channels
*/
void mk4array_display_warnings(mk4array *it, int truefalse, mk4error *err);

/*
 'mk4array_normalize'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : FILE*      file handle for the 'mean' information
   arg 3 : FILE*      file handle for the 'gain' information
   arg 4 : mk4error*  error state of function (if any)
   return: void
 If used, this function will force a normalization procedure to be applied on 
 the captured data using data created for this particular microphone array
 by the mean/gain file creator.
*/
void mk4array_normalize(mk4array *it, FILE* mean_fd, FILE* gain_fd, mk4error *err);

/*****/

/*
 'mk4array_capture_on'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : mk4error*  error state of function (if any)
   return: void
 Starts data capture on the Mk3
*/
void mk4array_capture_on(mk4array *it, mk4error *err);

/**********/ // 4) 'mk4array_status_capturing'

/*
 'mk4array_wait_capture_started'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : mk4error*  'mk4error' handle
   return: void
 Blocking opeation: Wait for capture 'status' to be set before returning
*/
void mk4array_wait_capture_started(mk4array *it, mk4error *err);

/**********/

/* Two possiblities are available to users when accessing the data itself */

/*****/
/* 1: Using "data buffers" (see functions definition a little further down).
      Asks the library to copy some data from its internal data types to 
      a user provided "data buffer" */

/*
  'mk4array_get_databuffer'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : char*      pointer to the "data buffer" location in which 
                      to copy the data
   arg 3 : timespec*  pointer to a localy created 'timespec' that will
                      contain the timespec detailling the capture time
		      of the requested buffer
   arg 4 : mk4error*  error state of function (if any)
   return: void
 Get the next available databuffer (blocking)
*/
void mk4array_get_databuffer(mk4array *it, char *databuffer, struct timespec *timestamp, mk4error *err);

/*
  'mk4array_get_databuffer_with_nfnbr'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : char*      pointer to the "data buffer" location in which 
                      to copy the data
   arg 3 : timespec*  pointer to a localy created 'timespec' that will
                      contain the timespec detailling the capture time
		      of the requested buffer
   arg 4 : timespec*  pointer to a localy created 'timespec' that will
                      contain the timespec detailling the capture time
		      of the requested buffer (a 'NULL' value means do
		      not return the corresponding timestamp)
   arg 5 : mk4error*  error state of function (if any)
   return: void
 Get the next available databuffer (blocking), and include the 
 'network data frame number' among the returned information (usualy only
 needed when trying to synchronize multiple microphone arrays together)
*/
void mk4array_get_databuffer_nfnbr(mk4array *it, char *databuffer, struct timespec *timestamp, int *nfnbr, mk4error *err);

/*
  'mk4array_get_databuffer_nonblocking'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : char*      pointer to the "data buffer" location in which 
                      to copy the data
   arg 3 : timespec*  pointer to a localy created 'timespec' that will
                      contain the timespec detailling the capture time
		      of the requested buffer (a 'NULL' value means do
		      not return the corresponding timestamp)
   arg 4 : mk4error*  error state of function (if any)
   return: int        'mk4_false' if no data was read / 'mk4_true' otherwise
 Get the next available databuffer (non-blocking), and move to cursor to next.
*/
int mk4array_get_databuffer_nonblocking(mk4array *it, char *db, struct timespec *ts, mk4error *err);

/*
  'mk4array_get_databuffer_nonblocking_with_nfnbr'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : char*      pointer to the "data buffer" location in which 
                      to copy the data
   arg 3 : timespec*  pointer to a localy created 'timespec' that will
                      contain the timespec detailling the capture time
		      of the requested buffer (a 'NULL' value means do
		      not return the corresponding timestamp)
   arg 4 : timespec*  pointer to a localy created 'timespec' that will
                      contain the timespec detailling the capture time
		      of the requested buffer (a 'NULL' value means do
		      not return the corresponding timestamp)
   arg 5 : mk4error*  error state of function (if any)
   return: int        'mk4_false' if no data was read / 'mk4_true' otherwise
 Get the next available databuffer (non-blocking), and move to cursor to next.
 Include the 'network data frame number' among the returned information
 (usualy only needed when trying to synchronize multiple microphone arrays 
 together)
*/
int mk4array_get_databuffer_nonblocking_with_nfnbr(mk4array *it, char *db, struct timespec *ts, int *nfnbr, mk4error *err);

/*
 'mk4array_get_current_databuffer_timestamp'
  arg 1 : mk4array*  'mk4array' handle
  arg 2 : timespec*  pointer to a localy created 'timespec' that will
                      contain the timespec detailling the capture time
		      of the requested buffer
  arg 3 : mk4error*  error state of function (if any)
  return: void
 Get the next available timestamp (blocking), but do not consider the
 databuffer as read (one can either 'skip' or 'get' it there after)
*/
void mk4array_get_current_databuffer_timestamp(mk4array *it, struct timespec *ts, mk4error *err);

/*
 'mk4array_get_current_databuffer_nfnbr'
  arg 1 : mk4array*  'mk4array' handle
  arg 2 : mk4error*  error state of function (if any)
  return: int        databuffer's 'network data frame number'
 Get the next available 'network data frame number' (blocking), but do not
 consider the databuffer as read (one can either 'skip' or 'get' it there
 after)
*/
int mk4array_get_current_databuffer_nfnbr(mk4array *it, mk4error *err);

/*
 'mk4array_skip_current_databuffer'
  arg 1 : mk4array*  'mk4array' handle
  arg 2 : mk4error*  error state of function (if any)
  return: void
 Consider the next available databuffer read and move cursor to next.
*/
void mk4array_skip_current_databuffer(mk4array *it, mk4error *err);

/*
 'mk4array_check_databuffer_overflow'
  arg 1 : mk4array*  'mk4array' handle
  arg 2 : mk4error*  error state of function (if any)
  return: int        'mk4_false' or 'mk4_true' dependent on overflow of 
                     internal structure
 Check if an ring buffer overflow occured for 'databuffer' access
*/
int mk4array_check_databuffer_overflow(mk4array* it, mk4error *err);

/*****/
/* 2: Using an access cursor (see "mk4cursor" functions below)
      Gives direct access to the data from the library.
      Perfect method if one does not want to do have to worry about creating
      memory buffers in which to store the data (see 'databuffers' otehrwise)
      Also, perfect method for capture clients, since the user does not have
      to worry about maintaining its own ring buffer, the library's internal
      is used.
      ***WARNING***: This is a "direct access" to the location of the data
      in memory and therefore no modification to this data should be made
      without copying it into a temporary buffer first and working from
      this buffer ('databuffers' might be more userful then, see method 1)
*/

/*
  'mk4array_get_cursorptr'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : mk4cursor* 'mk4cursor' handle
   arg 3 : char**     pointer to a 'char*' that will contain the address
                      of the data (ie, for 'char *ptr', use '&ptr')
   arg 4 : timespec*  pointer to a localy created 'timespec' that will
                      contain the timespec detailling the capture time
		      of the requested buffer (a 'NULL' value means do
		      not return the corresponding timestamp)
   arg 5 : mk4error*  error state of function (if any)
   return: void
 Get the next available data pointer from the Mk3's cursor (blocking)
*/
void mk4array_get_cursorptr(mk4array *it, mk4cursor *c, char** ptr, struct timespec *ts, mk4error *err);

/*
  'mk4array_get_cursorptr_with_nfnbr'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : mk4cursor* 'mk4cursor' handle
   arg 3 : char**     pointer to a 'char*' that will contain the address
                      of the data (ie, for 'char *ptr', use '&ptr')
   arg 4 : timespec*  pointer to a localy created 'timespec' that will
                      contain the timespec detailling the capture time
		      of the requested buffer (a 'NULL' value means do
		      not return the corresponding timestamp)
   arg 5 : int *      pointer to a localy created 'int' that will contain the
                      'network data frame number' of the first element
		      in the returned pointer (a 'NULL' value means do not
		      return the corresponding 'nfnbr')
   arg 6 : mk4error*  error state of function (if any)
   return: void
 Get the next available data pointer from the Mk3's cursor (blocking), and
 include the 'network data frame number' among the returned information
 (usualy only needed when trying to synchronize multiple microphone arrays
 together)
*/
void mk4array_get_cursorptr_with_nfnbr(mk4array *it, mk4cursor *c, char** ptr, struct timespec *ts, int *nfnbr, mk4error *err);

/*
  'mk4array_get_cursorptr_nonblocking'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : mk4cursor* 'mk4cursor' handle
   arg 3 : char**     pointer to a 'char*' that will contain the address
                      of the data (ie, for 'char *ptr', use '&ptr')
   arg 4 : timespec*  pointer to a localy created 'timespec' that will
                      contain the timespec detailling the capture time
		      of the requested buffer (a 'NULL' value means do
		      not return the corresponding timestamp)
   arg 5 : mk4error*  error state of function (if any)
   return: int        'mk4_false' if no data was read / 'mk4_true' otherwise
 Get the next available data pointer from the Mk3's cursor (non-blocking)
*/
int mk4array_get_cursorptr_nonblocking(mk4array *it, mk4cursor *c, char** ptr, struct timespec *ts, mk4error *err);

/*
  'mk4array_get_cursorptr_nonblocking_with_nfnbr'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : mk4cursor* 'mk4cursor' handle
   arg 3 : char**     pointer to a 'char*' that will contain the address
                      of the data (ie, for 'char *ptr', use '&ptr')
   arg 4 : timespec*  pointer to a localy created 'timespec' that will
                      contain the timespec detailling the capture time
		      of the requested buffer (a 'NULL' value means do
		      not return the corresponding timestamp)
   arg 5 : int *      pointer to a localy created 'int' that will contain the
                      'network data frame number' of the first element
		      in the returned pointer (a 'NULL' value means do not
		      return the corresponding 'nfnbr')
   arg 6 : mk4error*  error state of function (if any)
   return: int        'mk4_false' if no data was read / 'mk4_true' otherwise
 Get the next available data pointer from the Mk3's cursor (non-blocking), and
 include the 'network data frame number' among the returned information
 (usualy only needed when trying to synchronize multiple microphone arrays
 together)
*/
int mk4array_get_cursorptr_nonblocking_with_nfnbr(mk4array *it, mk4cursor *c, char** ptr, struct timespec *ts, int *nfnbr, mk4error *err);

/* NOTE: No function to read just the timestamp or skip a buffer is given, 
   since in this mode no data is memcpy-ed, it is simple to just skip it */

/*
 'mk4array_check_cursor_overflow'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : mk4cursor* 'mk4cursor' handle
   arg 3 : mk4error*  error state of function (if any)
   return: int        'mk4_false' or 'mk4_true' dependent on overflow status
 Check if an ring buffer overflow occured for specified cursor
*/
int mk4array_check_cursor_overflow(mk4array *it, mk4cursor *c, mk4error *err);

/**********/

/*
 'mk4array_check_lostdataframes'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : mk4error*  error state of function (if any)
   return: int        total number of data frames lost so far
 Return an 'int' containing the total number of data frames lost since the
 begining of the capture (reminder: 1 network frame = 5 data frames)
*/
int mk4array_check_lostdataframes(mk4array *it, mk4error *err);

/*
 ' mk4array_check_capture_ok'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : mk4error*  error state of function (if any)
   return: int        'mk4_true' if capture ok, 'mk4_false' otherwise
  Check that no error are occuring in the capture process
*/
int mk4array_check_capture_ok(mk4array *it, mk4error *err);

/**********/

/*
 'mk4array_capture_off'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : mk4error*  error state of function (if any)
   return: void
 Stops data capture on the Mk3
*/
void mk4array_capture_off(mk4array *it, mk4error *err);

/**********/ // X) after everything

/* 
 'mk4array_delete'
   arg 1 : mk4array*   'mk4array' handle
   arg 2 : mk4error*   error state of function (if any)
   return: void
 The function frees memory used by the 'mk4array'
*/
void mk4array_delete(mk4array* it, mk4error *err);




/************************************************************/
/****************************************/
/*
  Once a 'mk4array' handler has been created (with the requested
  number of data per buffer), it is possible to have the library
  automaticaly malloc/free "data buffer(s)", where 1 "data buffer"
  contains "requested number of data per buffer" elements of "1 dataframe"

   ie:
   1 "data buffer" = 'requested_data_frames_per_buffer' "data frames"
*/

/*
  'mk4array_create_databuffers'
   arg 1 : mk4array*   'mk4array' handle
   arg 2 : int         'quantity' of "data buffers" to create
   arg 3 : int*        pointer to store the total size of the data buffer
                       created ('NULL' not to use)
   arg 4 : mk4error*   error state of function (if any)
   return: char*       pointer to the data buffer
 Create 'quantity' "data buffers" 
*/
char* mk4array_create_databuffers(mk4array *it, int quantity, int *size, mk4error *err);

/*
  'mk4array_create_databuffer'
   arg 1 : mk4array*   'mk4array' handle
   arg 2 : int*        pointer to store the total size of the data buffer 
                       created ('NULL' not to use)
   arg 3 : mk4error*   error state of function (if any)
   return: char*       pointer to the data buffer
 Create 1 "data buffer"
*/
char* mk4array_create_databuffer(mk4array *it, int *size, mk4error *err);

/*
  'mk4array_delete_databuffer'
   arg 1 : mk4array*   'mk4array' handle
   arg 2 : char*       pointer of the "data buffer"(s)
   arg 3 : mk4error*   error state of function (if any)
   return: void
 Free the memory previously allocated for a "data buffer" pointer (1 or more)
*/
void mk4array_delete_databuffer(mk4array *it, char *db, mk4error *err);

/*
 'mk4array_get_one_databuffersize'
  arg 1 : mk4array*  'mk4array' handle
  arg 2 : mk4error*   error state of function (if any)
  return: int         size of 1 "data buffer"
 Returns the actual bytes size of 1x "data buffer"
*/
int mk4array_get_databuffersize(mk4array *it, mk4error *err);




/************************************************************/
/****************************************/
/*
  'mk4cursor' are access handlers for direct access to the internal storage
  methods, and are very useful if you intend to access the data location 
  directly, and/or need to use more than one access cursor.
  WARNING: in this case, you still need to make sure that the cursors 
           'requested data frame per buffer' ('rdfpb') are multiples of
	   one another, and use the highest value in the 'mk4array_create'
	   function.
  WARNING: since this method gives direct access to the internal storage area
           _extreme_ caution must be taken _not_ to modify the content.
*/  

/*
 'mk4cursor_create'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : int        'rdfpb'
   arg 3 : mk4error   error state of function (if any)
   return: mk4cursor*
 Creates a 'mk4cursor' to directly access the internal data
*/
mk4cursor *mk4cursor_create(mk4array* it, int requested_data_frames_per_buffer, mk4error* err);

/*
 'mk4cursor_delete'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : mk4cursor* 'mk4cursor' handle
   arg 3 : mk4error   error state of function (if any)
   return: void
 Delete a 'mk4cursor
*/
void mk4cursor_delete(mk4array* it, mk4cursor *c, mk4error *err);

/*
 'mk4cursor_get_datasize'
   arg 1 : mk4array*  'mk4array' handle
   arg 2 : mk4cursor* 'mk4cursor' handle
   arg 3 : mk4error   error state of function (if any)
   return: int
 Returns the size of one element pointed by when obtained from a cursor
*/
int mk4cursor_get_datasize(mk4array* it, mk4cursor *c, mk4error *err);





/************************************************************/
/****************************************/

/*
 'mkarray_init_mk4error'
  arg 1 : mk4error*   error state to initialize
  return: void
 Initialize a newly created 'mk4error'
*/
void mk4array_init_mk4error(mk4error *err);

/*
 'mk4array_perror'
  arg 1 : mk4error*   error state to print
  arg 2 : char*       Comment to accompany error message
  return: void
 Prints a error message corresponding to the error object given.
*/
void mk4array_perror(const mk4error *err, const char *comment);

/*
 'mk4array_check_mk4error'
  arg 1 : mk4error*   error state to check
  return: int         one 'mk4_false' if no error, or 'mk4_true' if error
 A function to allow check of error return value without having to compare
 it to 'MK3_OK'
*/
int mk4array_check_mk4error(mk4error *err);

#endif /* #ifdef HAVE_LIBMARKIV */

#endif


