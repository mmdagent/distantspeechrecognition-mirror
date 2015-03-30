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

#ifndef __MK4ERROR_H__
#define __MK4ERROR_H__

#ifdef HAVE_LIBMARKIV

/********************/
// MK4 error handler
typedef struct {
  int error_code;
} mk4error;

/********************/
// Error messages

enum {
  MK4_OK,
  MK4_CAPTURE,           // Error during capture
  MK4_RDFPB_ERROR,       // Invalid value for "req data frame per buffer"
  MK4_MB_ERROR,          // Invalid value for 'megabytes'
  MK4_NULL,              // Null value for one handler
  MK4_MALLOC,            // Problem allocating memory
  MK4_SOCKET,            // Could not create socket
  MK4_BIND,              // Could not bind socket
  MK4_DIP_ERROR,         // Destination IP error
  MK4_AUTOMATON_ORDER,   // Operation does not follow authorized order
  MK4_COMM_SEND,         // Error sending data to Mk3
  MK4_COMM_RECV,         // Error receiving data from Mk3
  MK4_GIBBERISH,         // Unexpected data / do not know how to parse
  MK4_SPEED_ERROR,       // Error with 'speed' value
  MK4_SLAVE_ERROR,       // Error with 'slave' value
  MK4_ID_ERROR,          // Error with 'id' value
  MK4_PROMNB_ERROR,      // Error with 'prom_nb' value
  MK4_NFNBR_ERROR,       // Eroor with value of "network frames number"
  MK4_DROPFRAMES_ERROR,  // Error with value of "frames to drop"
  MK4_SAMPLEDELAY_ERROR, // Error with value of "sample delay"
  MK4_INTERNAL,          // Internal error
  MK4_DISPLAY_ERROR,     // Error with value of "display warnings"
  MK4_CURSOR_COUNT,      // Too many cursors created
  MK4_RDFPB_MODULO,      // Value of 'rdfpb' is not a multiple of init value
  MK4_CURSOR_ERROR,      // Error working with this cursor
  MK4_NORM_FERROR,       // Error reading data from normalization files
};

#endif /* #ifdef HAVE_LIBMARKIV */

#endif
