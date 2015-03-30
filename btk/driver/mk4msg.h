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

#ifndef __MK4MSG_H__
#define __MK4MSG_H__

#include "driver/mk4array.h"
#include "driver/mk4error.h"

/********************/
// Messages handler for communication with the mk4 (not user availble)
#include "driver/mk4msg_type.h"

/********************/
// Functions
_mk4msg* _mk4msg_create(mk4error *err);
void _mk4msg_delete(_mk4msg *msg, mk4error *err);

// Send message to mk4
void _mk4msg_send(mk4array *it, _mk4msg *msg, mk4error *err);

// Recv message from mk4
void _mk4msg_recv(mk4array *it, _mk4msg *msg, mk4error *err);

// Recv data frame (_discard_ all other data | used while capturing)
void _mk4msg_recv_dataframe(mk4array *it, _mk4msg *msg, mk4error *err);

// Extract functions
int _mk4msg_extract_framenumber(_mk4msg *msg);
char * _mk4msg_extract_dataptr(_mk4msg *msg);

// Query Loop: 1) send omsg | 2) recv imsg | 3) wait on 'imsg[2] == waiton'
void _mk4msg_queryloop(mk4array* it, _mk4msg *omsg, _mk4msg *imsg, int waiton, mk4error *err);

#endif
