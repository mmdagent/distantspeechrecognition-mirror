//                              -*- C++ -*-
//
//                           Speech Front End
//                                 (sfe)
//
//  Module:  sfe.common
//  Purpose: Common operations.
//  Author:  Fabian Jakobs and ABC
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or (at
// your option) any later version.
// 
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.


#ifndef _MACH_IND_IO_H_                 /* we only want to include each header file once ! */
#define _MACH_IND_IO_H_

#include <stdio.h>

typedef  unsigned char   UBYTE;         /* type for reading/writing floatbytes */


/*#############################################################################################################################
 #
 #   FUNCTION DECLARATIONS  
 #
 ############################################################################################################################*/

void  init_mach_ind_io( void );

UBYTE float_to_ubyte( float f );
float ubyte_to_float( UBYTE u );

float read_float( FILE *fp );
int   read_floats(FILE *fp, float *whereto, int count );

float read_floatbyte( FILE *fp );
int   read_floatbytes(FILE *fp, float *whereto, int count );

int   read_int( FILE *fp );
int   read_ints(FILE *fp, int *whereto, int count );

short read_short( FILE *fp );
int   read_shorts(FILE *fp, short *whereto, int count);

int   read_string( FILE *f, char *str);

int   read_scaled_vectors( FILE *fp, float **whereto,  int* coeffNP, int *vectorNP );
int   read_scaled_vectors_range( FILE *fp, float **whereto, int *coeffNP, int *vectorNP, int from, int to);
int   write_scaled_vectors(FILE *fp, float *wherefrom, int  coeffN,  int  vectorN );

void  write_float( FILE *fp, float f);
int   write_floats(FILE *fp, float *wherefrom, int count);

void  write_floatbyte( FILE *fp, float f);
int   write_floatbytes(FILE *fp, float *wherefrom, int count);

void  write_int( FILE *fp, int i);
int   write_ints(FILE *fp, int *wherefrom, int count);

void  write_short( FILE *fp, short s);
int   write_shorts(FILE *fp, short *wherefrom, int count);

int   write_string(FILE *f, const char* str);

int   set_machine( int new_machine );

int   check_byte_swap( short *buf, int bufN );       /* See if adc data should be swapped */

void  buf_byte_swap( short *buf, int bufN );         /* Swap bytes for each short in an adc data buffer */

int short_memorychange(short *buf, int bufN);
int float_memorychange(float *buf, int bufN);
int int_memorychange(int *buf, int bufN);
int change_short(short *x);
int change_int(int *x);
int change_float(float *x);


/*************************************************************************************************************************/

#endif   /* _MACH_IND_IO_H_ */
