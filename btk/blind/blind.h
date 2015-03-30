//
//                       Beamforming Toolkit
//                              (btk)
//
//  Module:  btk.modulated
//  Purpose: Cosine modulated analysis and synthesis filter banks.
//  Author:  John McDonough and Andrej Schkarbanenko
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


#ifndef _blind_h_
#define _blind_h_

#include <stdio.h>
#include <assert.h>

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include "common/jexception.h"

#include "stream/stream.h"
#include "stream/pyStream.h"

void blind();

void gradient(const gsl_matrix* Rss, const gsl_matrix* Ryy, gsl_matrix* W);

#endif // _blind_h_

