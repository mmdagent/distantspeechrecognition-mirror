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
#ifndef __MK4_COMMON_H__
#define __MK4_COMMON_H__

// Quick err check
#define MK4QEC(X) (X->error_code != MK4_OK)
#define MK4CQEC(X) (X.error_code != MK4_OK)

enum { // Extension to 'True / False'
  mk4_limbo  = 440, // This is only used as an initializer
};

enum { // Default values for various 'mk4_array' components
  _mk4array_speed_default          = 0,
  _mk4array_dip_maxsize            = 16,
  _mk4array_id_default             = -1,
  _mk4array_prom_nb_maxsize        = 9,
  _mk4array_fd_default             = -1,
  _mk4array_slave_default          = mk4_limbo,
  _mk4array_nring_mb_default       = 16,
  _mk4array_nfnbr_default          = -1,
  _mk4array_nfnbr_min              = 0,
  _mk4array_nfnbr_max              = 65535,
  _mk4array_todrop_default         = 0,
};

enum { // Ring Buffer
  _mk4array_dfpnf   = 6,           // 6 data frames per network frame
  _mk4array_nbpc    = 3,           // 3 bytes per channel
  _mk4array_nc      = 64,          // 64 channels
  _mk4array_dfsize  = _mk4array_nbpc * _mk4array_nc,
                      // 1x 64 channels with 3 byts audio sample
  _mk4array_nfsize  = _mk4array_dfpnf*_mk4array_dfsize, // 1 network request
};

enum { // for '_mk4msg'
  _mk4msg_maxsize = 2048, // This is wateful but allows for a small security
    /* over the TCP 1536 standard limit ... yes, I know we using are UDP :) */
};

enum { // for 'mk4cursor'
  _mk4cursor_maxnbr = 256, // maximum number of cursors allowed
};

#endif

