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

#include "mk4lib.h"

#include <string.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

void _mk4array_query_ids(mk4array *it, mk4error *err)
{
  // Communication with the Mk3
  _mk4msg *omsg, *imsg;

  omsg = _mk4msg_create(err);
  if (MK4QEC(err)) return;
  imsg = _mk4msg_create(err);
  if (MK4QEC(err)) return;

  // "Create" the output message
  omsg->msg[1] = 3;
  omsg->len = 4;

  // Query the Mk3
  _mk4msg_queryloop(it, omsg, imsg, 3, err);
  if (MK4QEC(err)) return;

  it->id = imsg->msg[4] | (imsg->msg[5] << 8);
  memcpy(it->prom_nb, imsg->msg + 11, 8);
  it->prom_nb[8] = '\0';

  // Clean up
  _mk4msg_delete(imsg, err);
  if (MK4QEC(err)) return;
  _mk4msg_delete(omsg, err);
  if (MK4QEC(err)) return;

  err->error_code = MK4_OK;
}

/*****/

int _mk4array_query_capture(mk4array *it, mk4error *err)
{
  // Communication with the Mk3
  _mk4msg *omsg, *imsg;
  int res;

  omsg = _mk4msg_create(err);
  if (MK4QEC(err)) return mk4_limbo;
  imsg = _mk4msg_create(err);
  if (MK4QEC(err)) return mk4_limbo;

  // "Create" the output message
  omsg->msg[1] = 5;
  omsg->len = 4;

  // Query the Mk3
  _mk4msg_queryloop(it, omsg, imsg, 5, err);
  if (MK4QEC(err)) return mk4_limbo;
  
  res = (int) imsg->msg[2];

  // Clean up
  _mk4msg_delete(imsg, err);
  if (MK4QEC(err)) return mk4_limbo;
  _mk4msg_delete(omsg, err);
  if (MK4QEC(err)) return mk4_limbo;

  // Interpret result
  err->error_code = MK4_OK;
  if (res == 1)
    return mk4_true;
  else if (res == 0)
    return mk4_false;
  else {
    err->error_code = MK4_GIBBERISH;
    return mk4_limbo;
  }
}

/*****/

void _mk4array_enforce_capture(mk4array *it, int truefalse, mk4error *err)
{
    // Communication with the Mk3
  _mk4msg *omsg;
  int done;

  omsg = _mk4msg_create(err);
  if (MK4QEC(err)) return;
  
  // "Create" the output message
  omsg->msg[1] = 4;
  if (truefalse == mk4_true) {
    omsg->msg[2] = 0;
    omsg->msg[3] = 1;
    omsg->msg[4] = 0xff;
    omsg->msg[5] = 0xff;
    omsg->msg[6] = 0xff;
    omsg->msg[7] = 0xff;
    omsg->msg[8] = 0xff;
    omsg->msg[9] = 0xff;
    omsg->msg[10] = 0xff;
    omsg->msg[11] = 0xff;
  }  
  omsg->len = 12;

  // Set the Mk3
  done = mk4_limbo;
  while (done != truefalse) {
    _mk4msg_send(it, omsg, err);
    if (MK4QEC(err)) return;

    done = _mk4array_query_capture(it, err);
    if (MK4QEC(err)) return;
  }

  _mk4msg_delete(omsg, err);

  err->error_code = MK4_OK;
}

/*****/

void _mk4array_set_dip(mk4array *it, const char *dip, mk4error *err)
{
  int len;
  
  // Too long IP ? ('254.254.254.254' max)
  len = strlen(dip);
  if (len > _mk4array_dip_maxsize) {
    err->error_code = MK4_DIP_ERROR;
    return;
  } 
  
  strncpy(it->dip, dip, len);
  
  err->error_code = MK4_OK;
}

/*****/

void _mk4array_query_speed(mk4array *it, mk4error *err)
{
  // Communication with the Mk3
  _mk4msg *omsg, *imsg;
  int res;

  omsg = _mk4msg_create(err);
  if (MK4QEC(err)) return;
  imsg = _mk4msg_create(err);
  if (MK4QEC(err)) return;

  // "Create" the output message
  omsg->msg[1] = 8;
  omsg->len = 4;

  // Query the Mk3
  _mk4msg_queryloop(it, omsg, imsg, 7, err);
  if (MK4QEC(err)) return;

  res = (int) imsg->msg[2];

  // Clean up
  _mk4msg_delete(imsg, err);
  if (MK4QEC(err)) return;
  _mk4msg_delete(omsg, err);
  if (MK4QEC(err)) return;

  // Interpret result
  if (res == 4)
    it->speed = mk4array_speed_44K;
  else if (res == 2)
    it->speed = mk4array_speed_22K;
  else {
    err->error_code = MK4_GIBBERISH;
    return;
  }

  err->error_code = MK4_OK;
}

/*****/

void _mk4array_enforce_speed(mk4array *it, int speed, mk4error *err)
{
  // Communication with the Mk3
  _mk4msg *omsg;
  
  omsg = _mk4msg_create(err);
  if (MK4QEC(err)) return;
  
  // "Create" the output message
  omsg->msg[1] = 7;
  if (speed == mk4array_speed_44K) {
    omsg->msg[2] = 0;
    omsg->msg[3] = 4;
  } else if (speed == mk4array_speed_22K) {
    omsg->msg[2] = 0;
    omsg->msg[3] = 2;
  }
  omsg->len = 4;

  // Set the Mk3
  while(it->speed  != speed) {
    _mk4msg_send(it, omsg, err);
    if (MK4QEC(err)) return;
    
    _mk4array_query_speed(it, err);
    if (MK4QEC(err)) return;
  }

  _mk4msg_delete(omsg, err);
  if (MK4QEC(err)) return;

  err->error_code = MK4_OK;
}

/*****/

void _mk4array_query_slave(mk4array *it, mk4error *err)
{
  // Communication with the Mk3
  _mk4msg *omsg, *imsg;
  int res;

  omsg = _mk4msg_create(err);
  if (MK4QEC(err)) return;
  imsg = _mk4msg_create(err);
  if (MK4QEC(err)) return;

  // "Create" the output message
  omsg->msg[1] = 2;
  omsg->len = 4;

  // Query the Mk3
  _mk4msg_queryloop(it, omsg, imsg, 2, err);
  if (MK4QEC(err)) return;

  res = (int) imsg->msg[2];

  // Clean up
  _mk4msg_delete(imsg, err);
  if (MK4QEC(err)) return;
  _mk4msg_delete(omsg, err);
  if (MK4QEC(err)) return;

  // Interpret result
  if (res == 1)
    it->slave = mk4_true;
  else if (res == 0)
    it->slave = mk4_false;
  else {
    err->error_code = MK4_GIBBERISH;
    return;
  }

  err->error_code = MK4_OK;
}

/*****/

void _mk4array_enforce_slave(mk4array *it, int slave, mk4error *err)
{
  // Communication with the Mk3
  _mk4msg *omsg;
  
  omsg = _mk4msg_create(err);
  if (MK4QEC(err)) return;
  
  // "Create" the output message
  omsg->msg[1] = 1;
  if (slave == mk4_true) {
    omsg->msg[2] = 0xff;
    omsg->msg[3] = 0xff;
  }  
  omsg->len = 4;

  // Set the Mk3
  while (it->slave  != slave) {
    _mk4msg_send(it, omsg, err);
    if (MK4QEC(err)) return;

    _mk4array_query_slave(it, err);
    if (MK4QEC(err)) return;
  }

  _mk4msg_delete(omsg, err);
  if (MK4QEC(err)) return;

  err->error_code = MK4_OK;
}

/*****/

void _mk4array_wait_status(mk4array *it, int status, mk4error *err)
{
  struct timespec req;

  // Check that the requested status exist ?
  if ((it->status < mk4array_status_wait_initparams) 
    && (status > mk4array_status_capturing)) {
    err->error_code = MK4_AUTOMATON_ORDER;
    return;
  }

  req.tv_sec  = 0;
  req.tv_nsec = 50000000;

  while (it->status != status) {
    // We wait for 50ms (1s is too long for this kind of data bandwidth)
    nanosleep(&req, NULL);
  }
  
  err->error_code = MK4_OK;
}

/************************************************************/
// 'databuffer' functions
// Note: ALL 'databuffer' functions always refer to 'cursors[0]'

int _mk4array_get_databuffer(mk4array *it, char *db, struct timespec *ts, int *nfnbr, mk4error *err, int blocking)
{
  mk4cursor *c = it->cursors[0];
  char *ptr = NULL;
  int tell;
  
  // Get the pointer info
  tell = mk4cursor_get_ptr(it, c, &ptr, ts, nfnbr, err, blocking, mk4_true);
  
  // We have no data to read yet (if non blocking), or 'running' off, or error
  if (tell != mk4_true)
    return tell;
  
  // Copy the data itself
  memcpy(db, ptr, c->rdfpb_size);
  
  err->error_code = MK4_OK;
  return mk4_true;
}

/*****/

void _mk4array_get_current_databuffer_timestamp(mk4array* it, struct timespec *ts, mk4error *err)
{
  mk4cursor *c = it->cursors[0];
  int tell;

  tell = mk4cursor_get_ptr(it, c, NULL, ts, NULL, err, mk4_true, mk4_false);
  
  // Problem 
  if (tell != mk4_true)
    return;
  
  err->error_code = MK4_OK;
}

/*****/

int _mk4array_get_current_databuffer_nfnbr(mk4array* it, mk4error *err)
{
  mk4cursor *c = it->cursors[0];
  int tmp;
  int tell;

  tell = mk4cursor_get_ptr(it, c, NULL, NULL, &tmp, err, mk4_true, mk4_false);
  
  // Problem 
  if (tell != mk4_true)
    return 0;
  
  err->error_code = MK4_OK;
  return tmp;
}

/*****/

void _mk4array_skip_databuffer(mk4array *it, mk4error *err)
{
  mk4cursor *c = it->cursors[0];
  int tell;

  tell = mk4cursor_get_ptr(it, c, NULL, NULL, NULL, err, mk4_true, mk4_true);

  // Problem
  if (tell != mk4_true)
    return;

  err->error_code = MK4_OK;
} 

/*****/

int _mk4array_check_databuffer_overflow(mk4array* it, mk4error *err)
{
  mk4cursor *c = it->cursors[0];

  return mk4cursor_check_overflow(it, c, err);
}

/************************************************************/
// Ring Buffer methods

void _mk4array_rb_add_w(mk4array* it)
{
  it->wpos += _mk4array_dfpnf;

  if (it->wpos >= it->nring)
    it->wpos -= it->nring;
}

/*****/

void _mk4array_rb_add(mk4array* it, mk4cursor *c)
{
  c->rpos += c->rdfpb;

  if (c->rpos >= it->nring)
    c->rpos -= it->nring;
}

/*****/

int _mk4array_rb_diff(mk4array *it, mk4cursor *c)
{
  int w = it->wpos;

  if (w < c->rpos) w += it->nring;

  return (w - c->rpos);
}

/*****/

int _mk4array_rb_canread(mk4array * it, mk4cursor *c)
{
  if (_mk4array_rb_diff(it, c) < c->rdfpb)
    return 0;
  else
    return 1;
}

/************************************************************/
// The capture thread functions


void _mk4array_capture_wait_nfnbr(mk4array *it)
{
  if (it->wait_nfnbr == _mk4array_nfnbr_default)
    return;

  while (it->running) {
    _mk4msg_recv_dataframe(it, it->msg_flip, &(it->cerr));
    if (_mk4msg_extract_framenumber(it->msg_flip) == it->wait_nfnbr)
      return;
  }
}

/*****/

void _mk4array_capture_dropinit(mk4array *it)
{
  while ((it->running) && (it->samples < it->todrop)) {
    _mk4msg_recv_dataframe(it, it->msg_flip, &(it->cerr));
    if (MK4CQEC(it->cerr))
      return;
    it->samples += _mk4array_dfpnf;
  }

  it->cerr.error_code = MK4_OK;
}

/*****/

void _mk4array_normalize(mk4array *it, char *where, int size)
{
  // Note: This function is 'intel byte order'ed
  int i;
  int dsf = 0; // done so far

  unsigned char *ptr = (unsigned char *) where;

  while (dsf < size) {
    for (i = 0; i < _mk4array_nc; i++) {
      int buffer = 0;
      unsigned char *bptr = (unsigned char *) &buffer; // buffer pointer

      // cma 24 to intel 32
      buffer = ptr[2] | (ptr[1] << 8)  | ((signed char) ptr[0] << 16);

      // Normalization
      buffer = (int) (((double) (buffer - it->mean[i])) / it->gain[i]);

      // Write the data back (intel 32 -> cma 24)
      ptr[2] = bptr[0];
      ptr[1] = bptr[1];
      ptr[0] = bptr[2];

      // Increase 'ptr'
      ptr += _mk4array_nbpc;
    }

    // Move to the next data frame
    dsf += _mk4array_dfsize;
  }
}

/*****/

void _mk4array_write_fixed(char *where, char* cur, char *prev, int size)
{
  int dsf = 0; // done so far

  char *inprev = prev + (_mk4array_dfpnf - 1) * _mk4array_dfsize;
  char *incur  = cur;

  while (dsf < size) {
    // Microphone array mikes are numbered from 0 to 63
    char *dst  = where + dsf;
    char *even = incur;  // 0 2 4 ... 62
    char *odd  = inprev; // 1 3 5 ... 63
    int i;

    odd += 3; // 'odd' needs to start on channel 1 (not 0)
    for (i = 0; i < _mk4array_nc; i += 2) {
      // Even
      dst[0] = even[0];
      dst[1] = even[1];
      dst[2] = even[2];

      even += 6;
      dst  += 3;

      // Odd
      dst[0] = odd[0];
      dst[1] = odd[1];
      dst[2] = odd[2];

      odd += 6;
      dst += 3;
    }
    
    inprev = cur + dsf;
    dsf += _mk4array_dfsize;
    incur  = cur + dsf;
  }
}

/*****/

void _mk4array_incwpos(mk4array *it)
{
  int i, j;
  
  // Mutex
  // we increase the value to the writting pointer only once data has
  // been added to the buffer, indicating that it is safe to read it now
  pthread_mutex_lock(&(it->mutex));
  _mk4array_rb_add_w(it);
  pthread_mutex_unlock(&(it->mutex));
  
  // If the user can get some data, "signal" the proper cursor
  j = 0;
  for (i = 0; ((j < it->cursors_count) && (i < _mk4cursor_maxnbr)); i++) {
    mk4cursor *c = it->cursors[i];
    if (c != NULL) {
      if (_mk4array_rb_canread(it, c))
	pthread_cond_signal(&(c->cond));
      j++;
    }
  }
}

/*****/

int _mk4array_check_nblost(mk4array *it, int nb, int expected_nb)
{
  int nb_lost;

  if (expected_nb == -1)
    return 0;

  nb_lost = (nb < expected_nb) ? (65535 - expected_nb) + nb : nb - expected_nb;

  if (nb_lost > 0) {
    // Print a message ?
    if (it->warn == mk4_true) {
      fprintf(stderr, "[MK4LIB] Lost %d-%d=>%d at %s"
	      , expected_nb, (nb-1) & 0xffff, nb_lost, _mk4array_showtime());
      fflush(stderr);
    }

    // Set the value
    it->total_nbp_lost += nb_lost;
  }

  return nb_lost;
}

/*****/

void _mk4array_capture_fixdelay_true(mk4array *it)
{
  int expected_nb, cur_nb, nb_lost;
  int rl = 0; // Recover from lost state
  int inc_wpos; // Increment 'wpos' ?
  int wait_onprev; // Wait on previous message (ie wait for 2 messages)

  expected_nb = cur_nb = -1;
  nb_lost = 0;
  rl = 0;
  wait_onprev = 1;
  while (it->running) {
    struct timeval last_tv, tv;
    struct timespec *access_ts;
    int *access_nfnbr;

    /* We capture one network data at a time (but we need 2 In for 1 Out) */
    inc_wpos = 1;

    // Collect the time info
    gettimeofday(&tv, 0);
    access_ts = it->RING_bufferts;
    access_ts += it->wpos / _mk4array_dfpnf;
    
    // Get the 'network frame number' position
    access_nfnbr = it->RING_buffer_nfnbr;
    access_nfnbr += it->wpos / _mk4array_dfpnf;

    if (nb_lost > 0) {
      /*****/ // We have lost some buffers, put '0's instead
      memset(it->RING_buffer + it->wpos * _mk4array_dfsize, 0, _mk4array_nfsize);
      cur_nb = (cur_nb + 1) & 0xffff;
      nb_lost--;
    } else if (rl != 0) {
      /*****/ // We have finished filling the lost buffers with zeros,
      // restart capture (using the "normal run" continuation loop),
      // yet we have to remember that we have now 1 network frame in the 
      // 'prev' buffer and it is going to be discarded too, we therefore
      // need to compensate for it by using '0's too
      memset(it->RING_buffer + it->wpos * _mk4array_dfsize, 0, _mk4array_nfsize);
      cur_nb = (cur_nb + 1) & 0xffff;
      rl = 0;
    } else if (wait_onprev == 1) {
      /*****/ // We have not collected two buffers yet -> need the first
      // (somehow this should only be called the very first time)
      _mk4msg_recv_dataframe(it, it->msg_flip, &(it->cerr));
      if (MK4CQEC(it->cerr)) return;

      // Get the network frame number, and set the expected one
      expected_nb = (_mk4msg_extract_framenumber(it->msg_flip) + 1) & 0xffff;

      it->flXp = mk4_false;
      wait_onprev = 0;
      inc_wpos = 0;
    } else {
      /*****/ // Normal run
      int nb;
      _mk4msg *cur, *prev;

      // select the current and previous buffer
      if (it->flXp == mk4_false) {
	cur  = it->msg_flop;
	prev = it->msg_flip;
	it->flXp = mk4_true;
      } else {
	cur  = it->msg_flip;
	prev = it->msg_flop;
	it->flXp = mk4_false;
      }

      // Receive the network data
      _mk4msg_recv_dataframe(it, cur, &(it->cerr));
      if (MK4CQEC(it->cerr)) return;

      // Get the network frame number
      nb = _mk4msg_extract_framenumber(cur);

      // Have we lost packets ?
      nb_lost = _mk4array_check_nblost(it, nb, expected_nb);
      if (nb_lost > 0) { 
	// we will have to recover later (after we wrote all the blanks)
	rl = 1;
	
	// Btw, we have not added any data this time
	inc_wpos = 0;
      } else {
	//Copy the "fixed" message data to the ring buffer
	_mk4array_write_fixed((char *)it->RING_buffer + it->wpos * _mk4array_dfsize, 
			      _mk4msg_extract_dataptr(cur),
			      _mk4msg_extract_dataptr(prev),
			      _mk4array_nfsize);
	cur_nb = nb;
      }

      // We can then increase 'expected_nb'
      expected_nb = (nb + 1) & 0xffff;
    }

    // Increase 'wpos' (if required) & set the timestamp and the nf_nbr
    if ((it->running) && inc_wpos) {
      access_ts->tv_sec  = last_tv.tv_sec;
      access_ts->tv_nsec = 1000 * last_tv.tv_usec;

      *access_nfnbr = cur_nb;

      // Before increasing 'wpos' we also do the normalization (if any)
      if (it->do_norm == mk4_true)
	_mk4array_normalize(it, (char *)it->RING_buffer + it->wpos * _mk4array_dfsize, 
			    _mk4array_nfsize);
      
      _mk4array_incwpos(it);
    }
    
    // Set 'last_tv'
    last_tv.tv_sec  = tv.tv_sec;
    last_tv.tv_usec = tv.tv_usec;
  }
  
  it->cerr.error_code = MK4_OK;
}

/*****/

void _mk4array_capture_fixdelay_false(mk4array *it)
{
  int expected_nb, cur_nb, nb_lost;
  int rl = 0; // Recover from lost state
  int inc_wpos; // Increment 'wpos' ?

  expected_nb = cur_nb = -1;
  nb_lost = 0;
  rl = 0;
  while (it->running) {
    struct timeval tv;
    struct timespec *access_ts;
    int *access_nfnbr;

    /* We capture one network data at a time */
    inc_wpos = 1;

    // Collect the time info
    gettimeofday(&tv, 0);
    access_ts = it->RING_bufferts;
    access_ts += it->wpos / _mk4array_dfpnf;
    access_ts->tv_sec  = tv.tv_sec;
    access_ts->tv_nsec = 1000 * tv.tv_usec;

    // Get the 'network frame number' position
    access_nfnbr = it->RING_buffer_nfnbr;
    access_nfnbr += it->wpos / _mk4array_dfpnf;

    if (nb_lost > 0) {
      /*****/ // We have lost some buffers, put '0's instead
      memset(it->RING_buffer + it->wpos * _mk4array_dfsize, 0, _mk4array_nfsize);
      cur_nb = (cur_nb + 1) & 0xffff;
      nb_lost--;
    } else if (rl != 0) {
      /*****/ // We have finished filling the lost buffers with zeros,
      // use the last available one before getting a new one
      memcpy(it->RING_buffer + it->wpos * _mk4array_dfsize,
	     _mk4msg_extract_dataptr(it->msg_flip), _mk4array_nfsize);
      cur_nb = (cur_nb + 1) & 0xffff;
      rl = 0;
    } else {
      /*****/ // Normal run
      int nb;

      // Receive the network data
      _mk4msg_recv_dataframe(it, it->msg_flip, &(it->cerr));
      if (MK4CQEC(it->cerr)) return;

      // Get the network frame number
      nb = _mk4msg_extract_framenumber(it->msg_flip);

      // Have we lost packets ?
      nb_lost = _mk4array_check_nblost(it, nb, expected_nb);
      if (nb_lost > 0) {
	// we will have to recover later (after we wrote all the blanks)
	rl = 1;
	
	// Btw, we have not added any data this time
	inc_wpos = 0;
      } else {
	//Copy the message data to the ring buffer
	memcpy(it->RING_buffer + it->wpos *  _mk4array_dfsize, 
	       _mk4msg_extract_dataptr(it->msg_flip), _mk4array_nfsize);
	cur_nb = nb;
      }
      
      // We can then increase 'expected_nb'
      expected_nb = (nb + 1) & 0xffff;
    }
    
    // Increase 'wpos' (if required) && set nf_nbr
    if ((it->running) && inc_wpos) {
      *access_nfnbr = cur_nb;
      // Before increasing 'wpos' we also do the normalization (if any)
      if (it->do_norm == mk4_true)
	_mk4array_normalize(it, (char *)it->RING_buffer + it->wpos * _mk4array_dfsize, 
			    _mk4array_nfsize);
      
      _mk4array_incwpos(it);
    }
    
  }
  
  it->cerr.error_code = MK4_OK;
}

/*****/

void *_mk4array_capture( void *hook )
{
  mk4array *it = (mk4array *)hook;
  // First wait until requested nfnbr if selected
  _mk4array_capture_wait_nfnbr(it);

  // First drop the requested amount of frames
  _mk4array_capture_dropinit(it);
  if (MK4CQEC(it->cerr)) goto bail_capture;
  
  // Now the real capture
  if (it->fixdelay == mk4_true)
    _mk4array_capture_fixdelay_true(it);
  else
    _mk4array_capture_fixdelay_false(it);
  
  if (MK4CQEC(it->cerr)) goto bail_capture;
  
  it->cerr.error_code = MK4_OK;
  return NULL;
  
 bail_capture:
  it->cerr.error_code = MK4_CAPTURE;
  return NULL;
}

#endif /* #ifdef HAVE_LIBMARKIV */
