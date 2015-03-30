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

#include "mk4_misc.h"

#include <time.h>

char *_mk4array_showtime()
{
  time_t tmp;

  time(&tmp);
  
  return ctime(&tmp);
}

/*****/

int _mk4array_gcd(int x, int y)
{ // Greatest Common Denominator
  while (y != x)
    if (x < y)
      y = y - x;
    else
      x = x - y;
  
  return y;
}

/*****/
 
int _mk4array_lcm(int x, int y)
{ // Least Common Multiple
  if (x % y == 0)
    return x;

  if (y % x  == 0)
    return y;

  return (x * (y / _mk4array_gcd(x, y)));
}

/*****/

unsigned long long int _mk4array_uint64ts(unsigned int sec, unsigned int nsec)
{
  // We work in 'ns'
  // This is based on the idea that timestamps should only be positive values
  unsigned long long int result;

  result = sec;
  result *= 1000000000;
  result += nsec;

  return (result);
}  
