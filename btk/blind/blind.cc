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


#include <math.h>
#include "modulated/modulated.h"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_fft_complex.h>

#include "blind/blind.h"

void blind() {
  printf("blind\n");
}

void gradient(const gsl_matrix* Rss, const gsl_matrix* Ryy, gsl_matrix* W)
{
  printf("Hello, world\n");
}
