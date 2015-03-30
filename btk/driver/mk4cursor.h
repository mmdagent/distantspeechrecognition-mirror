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

#ifndef __MK4CURSOR_H__
#define __MK4CURSOR_H__

#ifdef HAVE_LIBMARKIV

#include "driver/mk4array.h"

#include <pthread.h>


// Type
#include "mk4cursor_type.h"

/********************/
// Internal functions

// Check that the 'mk4cursor' is really part of the 'mk4array'
int mk4cursor_isof(mk4array *it, mk4cursor *c);

// Re-initialize the values of the 'mk4cursor' as per its creation
void mk4cursor_reinit(mk4array* it, mk4cursor *c, mk4error *err);

// Get the cursor data
int mk4cursor_get_ptr(mk4array *it, mk4cursor *c, char** ptr, struct timespec *ts, int *nfnbr, mk4error *err, int blocking, int incrpos);

// Check the overflow status
int mk4cursor_check_overflow(mk4array *it, mk4cursor *c, mk4error *err);

#endif /* #ifdef HAVE_LIBMARKIV */

#endif
