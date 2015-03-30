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

#ifdef HAVE_LIBMARKIV

#include "mk4msg.h"
#include "mk4_common.h"

#define _PRINTF_DEBUG
#ifdef _PRINTF_DEBUG
#include <stdio.h>
#endif 

#include <string.h>
#include <string.h>
#include <stdlib.h>
#include <sys/poll.h>

_mk4msg* _mk4msg_create(mk4error *err)
{
  _mk4msg* it = NULL;

  it = (_mk4msg *) calloc(1, sizeof(*it));
  if (it == NULL) {
    err->error_code = MK4_MALLOC;
    return NULL;
  }

  // Most of the messages need to be initialized to '0' : 'calloc'
  it->msg = (unsigned char *) calloc(1, _mk4msg_maxsize);
  if (it->msg == NULL)
    err->error_code = MK4_MALLOC;
  it->len = 0;
    
  err->error_code = MK4_OK;
  return it;
}

/*****/

void _mk4msg_delete(_mk4msg *msg, mk4error *err)
{
  if (msg->msg != NULL) free(msg->msg);
  free(msg);

  err->error_code = MK4_OK;
}

/*****/

void _mk4msg_send(mk4array *it, _mk4msg *msg, mk4error *err)
{
  err->error_code = MK4_OK;

  if (sendto(it->fd, msg->msg, msg->len, 0, 
	     (const struct sockaddr *) &(it->adr), sizeof(it->adr)) <0)
    err->error_code = MK4_COMM_SEND;
}

/*****/

void _mk4msg_recv(mk4array *it, _mk4msg *msg, mk4error *err)
{
  struct pollfd pfd;
  int counter, res;

  pfd.fd = it->fd;
  pfd.events = POLLIN;
  res = poll(&pfd, 1, 100);

  /*if (! res) {*/
  if ( res <= 0 ) {
#ifdef _PRINTF_DEBUG
    fprintf(stderr,"poll() returned %d\n", res);
#endif
    err->error_code = MK4_COMM_RECV;
    return;
  }

  msg->len = recv(it->fd, msg->msg, _mk4msg_maxsize, 0);

  err->error_code = MK4_OK;
}

/*****/

void _mk4msg_recv_dataframe(mk4array *it, _mk4msg *msg, mk4error *err)
{
  int done = 0;

  while (done == 0) {
    _mk4msg_recv(it, msg, err);
    if (MK4QEC(err)) return;

    done = (msg->msg[0] == 0x86);
  }
}

/*****/

void _mk4msg_queryloop(mk4array *it, _mk4msg *omsg, _mk4msg *imsg, int waiton, mk4error *err)
{
  int done = 0;

  while (! done) {
    _mk4msg_send(it, omsg, err);
    if (MK4QEC(err)) return;
  
    _mk4msg_recv(it, imsg, err);
    if (MK4QEC(err)) return;

    if (! imsg->len)
      continue;

    done = (imsg->msg[0] == waiton);
  }

  err->error_code = MK4_OK;
}

/*****/

int _mk4msg_extract_framenumber(_mk4msg *msg)
{
  return ((msg->msg[1] << 8) | msg->msg[2]);
}

/*****/

char * _mk4msg_extract_dataptr(_mk4msg *msg)
{
  return (char *) (msg->msg + 4);
}

#endif /* #ifdef HAVE_LIBMARKIV */
