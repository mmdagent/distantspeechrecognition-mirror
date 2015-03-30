//
//                                Millennium
//                    Automatic Speech Recognition System
//                                  (asr)
//
//  Module:  asr.fsm
//  Purpose: Representation and manipulation of finite state machines.
//  Author:  Stephan Kanthak, ported to Millennium by John McDonough
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

#include "fsm/skhash.h"

#if 0

const char *Type<u8>::name("u8");
const u8  Type<u8 >::max(255U);
const u8  Type<u8 >::min(0U);
const char *Type<s8>::name("s8");
const s8  Type<s8 >::max(127);
const s8  Type<s8 >::min(-128);
const char *Type<u16>::name("u16");
const u16 Type<u16>::max(65535U);
const u16 Type<u16>::min(0U);
const char *Type<s16>::name("s16");
const s16 Type<s16>::max(32767);
const s16 Type<s16>::min(-32768);
const char *Type<u32>::name("u32");
const u32 Type<u32>::max(4294967295U);
const u32 Type<u32>::min(0U);
const char *Type<s32>::name("s32");
const s32 Type<s32>::max(2147483647);
const s32 Type<s32>::min(-2147483647 - 1); // gcc warns about too large int when -2147483648
#if defined(HAS_64BIT)
const char *Type<u64>::name("u64");
const u64 Type<u64>::max(18446744073709551615U);
const u64 Type<u64>::min(0U);
const char *Type<s64>::name("s64");
const s64 Type<s64>::max(9223372036854775807LL);
const s64 Type<s64>::min(-9223372036854775808LL);
#endif
const char *Type<f32>::name("f32");
const f32 Type<f32>::max(+3.40282347e+38F);
const f32 Type<f32>::min(-3.40282347e+38F);
const f32 Type<f32>::epsilon(1.19209290e-07F);
const f32 Type<f32>::delta(1.17549435e-38F);
const char *Type<f64>::name("f64");
const f64 Type<f64>::max(+1.7976931348623157e+308);
const f64 Type<f64>::min(-1.7976931348623157e+308);
const f64 Type<f64>::epsilon(2.2204460492503131e-16);
const f64 Type<f64>::delta(2.2250738585072014e-308);

#endif

u32 estimateBytes(u32 x) {
  if (x >= (u32(1) << 24)) return 4;
  else if (x >= (u32(1) << 16)) return 3;
  else if (x >= (u32(1) << 8)) return 2;
  return 1;
}

void setBytes(Vector<u8>::iterator i, u32 x, int nBytes) {
  switch (nBytes) {
  case 4: *(i++) = ((x >> 24) & 0xff);
  case 3: *(i++) = ((x >> 16) & 0xff);
  case 2: *(i++) = ((x >> 8) & 0xff);
  case 1: *(i++) = (x & 0xff);
  case 0: break;
  }
}

void appendBytes(Vector<u8> &v, u32 x, int nBytes) {
  switch (nBytes) {
  case 4: v.push_back((x >> 24) & 0xff);
  case 3: v.push_back((x >> 16) & 0xff);
  case 2: v.push_back((x >> 8) & 0xff);
  case 1: v.push_back(x & 0xff);
  case 0: break;
  }
}

u32 getBytesAndIncrement(Vector<u8>::const_iterator &a, int nBytes) {
  u32 x = 0;
  switch (nBytes) {
  case 4: x |= *(a++); x <<= 8;
  case 3: x |= *(a++); x <<= 8;
  case 2: x |= *(a++); x <<= 8;
  case 1: x |= *(a++);
  case 0: break;
  }
  return x;
}

u32 getBytes(Vector<u8>::const_iterator a, int nBytes) {
  Vector<u8>::const_iterator a_ = a;
  return getBytesAndIncrement(a_, nBytes);
}
