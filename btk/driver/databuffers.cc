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

#include "driver/mk4lib.h"

#include "string.h"
#include "stdlib.h"

char* mk4array_create_databuffers(mk4array *it, int number, int *size, mk4error *err)
{
  char *tmp;
  int tsize;
  mk4cursor *c = it->cursors[0];

  tmp = (char *) calloc(number, c->rdfpb_size);
  if (tmp == NULL) {
    err->error_code = MK4_MALLOC;
    return NULL;
  }

  tsize = number * c->rdfpb_size;

  if (size != NULL)
    *size = tsize;

  err->error_code = MK4_OK;
  return tmp;
}

/*****/

char* mk4array_create_databuffer(mk4array *it, int *size, mk4error *err)
{
  return mk4array_create_databuffers(it, 1, size, err);
}


/*****/

void mk4array_delete_databuffer(mk4array *it, char *db, mk4error *err)
{
  if (it == NULL) {
    err->error_code = MK4_NULL;
    return;
  }

  free(db);

  err->error_code = MK4_OK;
}


/*****/

int mk4array_get_one_databuffersize(mk4array *it, mk4error *err)
{
  err->error_code = MK4_OK;
  return _mk4array_dfsize;
}

#endif /* #ifdef HAVE_LIBMARKIV */
