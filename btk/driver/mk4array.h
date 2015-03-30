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

#ifndef __MK4ARRAY_H__
#define __MK4ARRAY_H__

#include "driver/mk4cursor_type.h"
#include "driver/mk4msg_type.h"
#include "driver/mk4error.h"
#include "driver/mk4_misc.h"
#include "driver/mk4_common.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <pthread.h>

/********************/
// MK3 array handler
typedef struct {

  /********************/
  // "public" side of the mk4
  // (values 'read' using user avilable functions)

  char *dip;            // The destination IP
  int id;               // The mk4 id
  char *prom_nb;        // The mk4 Prom Number
  int speed;            // The capture speed
  int slave;            // The slave mode

  int status;           // The mk4 capture status

  /********************/
  // "private" side of the mk4

  // Normalization
  int    do_norm;
  double *gain;
  int    *mean;

  // socket related
  int fd;
  struct sockaddr_in adr;

  // Capture error information
  mk4error   cerr;

  // Capture thread
  pthread_t       tid;
  pthread_attr_t  attr;
  pthread_mutex_t mutex;
  volatile int   running;

  // Capture Ring buffer
  int nring;
  int nring_mb;
  unsigned char *RING_buffer;
  struct timespec *RING_bufferts;
  int *RING_buffer_nfnbr;
  int wpos;
  int rdfpb;

  // Default access cursor is the '0'th one (the one used when using 
  mk4cursor *cursors[_mk4cursor_maxnbr];
  int cursors_count;

  // Capture buffers (for re-ordering of data)
  _mk4msg* msg_flip;
  _mk4msg* msg_flop;
  int flXp; // Selector
  // Fix the Mk3 sample delay 
  int fixdelay;

  // Wait until specficic netword data frame number
  int wait_nfnbr;

  // Drop the X first samples
  int todrop;
  int samples;

  // Print warnings
  int warn;

  // Total number of packets lost
  int total_nbp_lost;

  // Note that ring buffer overflows are related to cursors only, not global
} mk4array;

/********************/
// Functions

// capture
int _mk4array_query_capture(mk4array *it, mk4error *err);
void _mk4array_enforce_capture(mk4array *it, int truefalse, mk4error *err);

// The capture thread itself
void *_mk4array_capture( void *it);

// ID
void _mk4array_query_ids(mk4array *it, mk4error *err);

// DIP
void _mk4array_set_dip(mk4array *it, const char *dip, mk4error *err);

// speed
void _mk4array_query_speed(mk4array *it, mk4error *err);
void _mk4array_enforce_speed(mk4array *it, int speed, mk4error *err);

// slave
void _mk4array_query_slave(mk4array *it, mk4error *err);
void _mk4array_enforce_slave(mk4array *it, int truefalse, mk4error *err);

// Wait on status
void _mk4array_wait_status(mk4array *it, int status, mk4error *err);

// databuffer 
int _mk4array_get_databuffer(mk4array *it, char *db, struct timespec *ts, int *nfnbr, mk4error *err, int blocking);
void _mk4array_get_current_databuffer_timestamp(mk4array* it, struct timespec *ts, mk4error *err);
int _mk4array_get_current_databuffer_nfnbr(mk4array* it, mk4error *err);
void _mk4array_skip_databuffer(mk4array *it, mk4error *err);
int _mk4array_check_databuffer_overflow(mk4array* it, mk4error* err);

/********************/
// Ring buffer methods

void _mk4array_rb_add_w(mk4array* it);
void _mk4array_rb_add(mk4array* it, mk4cursor *c);
int _mk4array_rb_diff(mk4array *it, mk4cursor *c);
int _mk4array_rb_canread(mk4array * it, mk4cursor *c);

#endif
