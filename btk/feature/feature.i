//                              -*- C++ -*-
//
//                           Speech Front End
//                                 (btk)
//
//  Module:  btk.feature
//  Purpose: Speech recognition front end.
//  Author:  John McDonough and ABC
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


%module(package="btk") feature

%init {
  // NumPy needs to set up callback functions
  import_array();
}

%{
#include "stream/stream.h"
#include "stream/pyStream.h"
#include <numpy/arrayobject.h>
#include "feature/feature.h"
#include "feature/lpc.h"
#ifdef HAVE_LIBMARKIV
#include "driver/mk4lib.h"
#endif /* HAVE_LIBMARKIV */
using namespace sndfile;
%}

%include btk.h
%include jexception.i
%include typedefs.i
%include vector.i
%include matrix.i

%import stream/stream.i

#ifdef AUTODOC
%section "Feature", before
#endif

enum
{	/* Major formats. */
	SF_FORMAT_WAV			= 0x010000,		/* Microsoft WAV format (little endian). */
	SF_FORMAT_AIFF			= 0x020000,		/* Apple/SGI AIFF format (big endian). */
	SF_FORMAT_AU			= 0x030000,		/* Sun/NeXT AU format (big endian). */
	SF_FORMAT_RAW			= 0x040000,		/* RAW PCM data. */
	SF_FORMAT_PAF			= 0x050000,		/* Ensoniq PARIS file format. */
	SF_FORMAT_SVX			= 0x060000,		/* Amiga IFF / SVX8 / SV16 format. */
	SF_FORMAT_NIST			= 0x070000,		/* NIST Sphere format. */
	SF_FORMAT_VOC			= 0x080000,		/* VOC files. */
	SF_FORMAT_IRCAM			= 0x0A0000,		/* Berkeley/IRCAM/CARL */
	SF_FORMAT_W64			= 0x0B0000,		/* Sonic Foundry's 64 bit RIFF/WAV */
	SF_FORMAT_MAT4			= 0x0C0000,		/* Matlab (tm) V4.2 / GNU Octave 2.0 */
	SF_FORMAT_MAT5			= 0x0D0000,		/* Matlab (tm) V5.0 / GNU Octave 2.1 */
	SF_FORMAT_PVF			= 0x0E0000,		/* Portable Voice Format */
	SF_FORMAT_XI			= 0x0F0000,		/* Fasttracker 2 Extended Instrument */
	SF_FORMAT_HTK			= 0x100000,		/* HMM Tool Kit format */
	// SF_FORMAT_SDS			= 0x110000,		/* Midi Sample Dump Standard */
	// SF_FORMAT_AVR			= 0x120000,		/* Audio Visual Research */
	// SF_FORMAT_WAVEX			= 0x130000,		/* MS WAVE with WAVEFORMATEX */

	/* Subtypes from here on. */

	SF_FORMAT_PCM_S8		= 0x0001,		/* Signed 8 bit data */
	SF_FORMAT_PCM_16		= 0x0002,		/* Signed 16 bit data */
	SF_FORMAT_PCM_24		= 0x0003,		/* Signed 24 bit data */
	SF_FORMAT_PCM_32		= 0x0004,		/* Signed 32 bit data */

	SF_FORMAT_PCM_U8		= 0x0005,		/* Unsigned 8 bit data (WAV and RAW only) */

	SF_FORMAT_FLOAT			= 0x0006,		/* 32 bit float data */
	SF_FORMAT_DOUBLE		= 0x0007,		/* 64 bit float data */
	// SF_FORMAT_PCM_SHORTEN		= 0x0008,		/* Shorten. */

	SF_FORMAT_ULAW			= 0x0010,		/* U-Law encoded. */
	SF_FORMAT_ALAW			= 0x0011,		/* A-Law encoded. */
	SF_FORMAT_IMA_ADPCM		= 0x0012,		/* IMA ADPCM. */
	SF_FORMAT_MS_ADPCM		= 0x0013,		/* Microsoft ADPCM. */

	SF_FORMAT_GSM610		= 0x0020,		/* GSM 6.10 encoding. */
	SF_FORMAT_VOX_ADPCM		= 0x0021,		/* OKI / Dialogix ADPCM */

	SF_FORMAT_G721_32		= 0x0030,		/* 32kbs G721 ADPCM encoding. */
	SF_FORMAT_G723_24		= 0x0031,		/* 24kbs G723 ADPCM encoding. */
	SF_FORMAT_G723_40		= 0x0032,		/* 40kbs G723 ADPCM encoding. */

	SF_FORMAT_DWVW_12		= 0x0040, 		/* 12 bit Delta Width Variable Word encoding. */
	SF_FORMAT_DWVW_16		= 0x0041, 		/* 16 bit Delta Width Variable Word encoding. */
	SF_FORMAT_DWVW_24		= 0x0042, 		/* 24 bit Delta Width Variable Word encoding. */
	SF_FORMAT_DWVW_N		= 0x0043, 		/* N bit Delta Width Variable Word encoding. */

	SF_FORMAT_DPCM_8		= 0x0050,		/* 8 bit differential PCM (XI only) */
	SF_FORMAT_DPCM_16		= 0x0051,		/* 16 bit differential PCM (XI only) */


	/* Endian-ness options. */

	SF_ENDIAN_FILE			= 0x00000000,	/* Default file endian-ness. */
	SF_ENDIAN_LITTLE		= 0x10000000,	/* Force little endian-ness. */
	SF_ENDIAN_BIG			= 0x20000000,	/* Force big endian-ness. */
	SF_ENDIAN_CPU			= 0x30000000,	/* Force CPU endian-ness. */

	SF_FORMAT_SUBMASK		= 0x0000FFFF,
	SF_FORMAT_TYPEMASK		= 0x0FFF0000,
	SF_FORMAT_ENDMASK		= 0x30000000
} ;

enum
{	/* True and false */
	SF_FALSE	= 0,
	SF_TRUE		= 1,

	/* Modes for opening files. */
	SFM_READ	= 0x10,
	SFM_WRITE	= 0x20,
	SFM_RDWR	= 0x30
} ;


// ----- definition for class `FileFeature' -----
// 
%ignore FileFeature;
class FileFeature : public VectorFloatFeatureStream {
public:
  FileFeature(unsigned sz, const String nm = "File");

  void bload(const String fileName, bool old = false) { bload(fileName, old); }

  unsigned size() const;

  const gsl_vector_float* next() const;

  void copy(gsl_matrix_float* matrix);
};

class FileFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    FileFeaturePtr(unsigned sz, const String nm = "File") {
      return new FileFeaturePtr(new FileFeature(sz, nm));
    }

    FileFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  FileFeature* operator->();
};


// ----- definition of Conversion24bit2Short -----
//
class Conversion24bit2Short : public VectorShortFeatureStream {
 public:
  Conversion24bit2Short(VectorCharFeatureStreamPtr& src,
			const String& nm = "Conversion from 24 bit integer to Short");
  virtual void reset() { _src->reset(); VectorShortFeatureStream::reset(); }
  virtual const gsl_vector_short* next(int frameX = -5);
};

class Conversion24bit2ShortPtr : public VectorShortFeatureStreamPtr {
 public:
  %extend {
   Conversion24bit2ShortPtr(VectorCharFeatureStreamPtr& src,
			    const String& nm = "Conversion from 24 bit integer to Short") {
      return new Conversion24bit2ShortPtr(new Conversion24bit2Short(src, nm));
   }
    Conversion24bit2ShortPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  Conversion24bit2ShortPtr* operator->();
};


// ----- definition of Conversion24bit2Float -----
//
class Conversion24bit2Float : public VectorFloatFeatureStream {
 public:
  Conversion24bit2Float(VectorCharFeatureStreamPtr& src,
			const String& nm = "Conversion from 24 bit integer to Float");
  virtual void reset() { _src->reset(); VectorFloatFeatureStream::reset(); }
  virtual const gsl_vector_float* next(int frameX = -5);
};

class Conversion24bit2FloatPtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
   Conversion24bit2FloatPtr(VectorCharFeatureStreamPtr& src,
			    const String& nm = "Conversion from 24 bit integer to Float") {
      return new Conversion24bit2FloatPtr(new Conversion24bit2Float(src, nm));
   }
    Conversion24bit2FloatPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  Conversion24bit2FloatPtr* operator->();
};


#ifdef HAVE_LIBMARKIV

// ----- definition for class `NISTMarkIII' -----
//
%ignore NISTMarkIII;
class NISTMarkIII {
 public:
  NISTMarkIII(VectorShortFeatureStreamPtr& src, unsigned chanN = 64,
              unsigned blkSz = 320 , unsigned shftSz = 160, unsigned triggerX = 0);
  ~NISTMarkIII();

  gsl_vector_float* next(unsigned chanX);

  void reset();

  unsigned blockSize() const;
  unsigned shiftSize() const;
  unsigned overlapSize() const;
};

class NISTMarkIIIPtr {
 public:
  %extend {
    NISTMarkIIIPtr(VectorShortFeatureStreamPtr& src, unsigned chanN = 64,
                   unsigned blkSz = 320, unsigned shftSz = 160, unsigned triggerX = 0) {
      return new NISTMarkIIIPtr(new NISTMarkIII(src, chanN, blkSz, shftSz, triggerX));
    }
  }

  NISTMarkIII* operator->();
};


// ----- definition for class `NISTMarkIIIFeature' -----
//
%ignore NISTMarkIIIFeature;
class NISTMarkIIIFeature : public VectorShortFeatureStream {
 public:
  NISTMarkIIIFeature(NISTMarkIIIPtr& markIII, unsigned chanX, unsigned chanN, const String& nm);
  ~NISTMarkIIIFeature();

  virtual gsl_vector_short* next(int frameX = -5);

  virtual void reset();
};

class NISTMarkIIIFeaturePtr : public VectorShortFeatureStreamPtr {
 public:
  %extend {
    NISTMarkIIIFeaturePtr(NISTMarkIIIPtr& markIII, unsigned chanX, unsigned chanN, const String& nm) {
      return new NISTMarkIIIFeaturePtr(new NISTMarkIIIFeature(markIII, chanX, chanN, nm));
    }

    NISTMarkIIIFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  NISTMarkIIIFeature* operator->();
};


// ----- definition for class `NISTMarkIIIFloat' -----
//
%ignore NISTMarkIIIFloat;
class NISTMarkIIIFloat {
 public:
  NISTMarkIIIFloat(VectorFloatFeatureStreamPtr& src, unsigned chanN = 64,
		   unsigned blkSz = 320 , unsigned shftSz = 160, unsigned triggerX = 0);
  ~NISTMarkIIIFloat();

  gsl_vector_float* next(unsigned chanX);

  void reset();

  unsigned blockSize() const;
  unsigned shiftSize() const;
  unsigned overlapSize() const;
};

class NISTMarkIIIFloatPtr {
 public:
  %extend {
    NISTMarkIIIFloatPtr(VectorFloatFeatureStreamPtr& src, unsigned chanN = 64,
			unsigned blkSz = 320, unsigned shftSz = 160, unsigned triggerX = 0) {
      return new NISTMarkIIIFloatPtr(new NISTMarkIIIFloat(src, chanN, blkSz, shftSz, triggerX));
    }
  }

  NISTMarkIIIFloat* operator->();
};


// ----- definition for class `NISTMarkIIIFloatFeature' -----
//
%ignore NISTMarkIIIFloatFeature;
class NISTMarkIIIFloatFeature : public VectorFloatFeatureStream {
 public:
  NISTMarkIIIFloatFeature(NISTMarkIIIFloatPtr& markIII, unsigned chanX, unsigned chanN, const String& nm);
  ~NISTMarkIIIFloatFeature();

  virtual gsl_vector_float* next(int frameX = -5);

  virtual void reset();
};

class NISTMarkIIIFloatFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    NISTMarkIIIFloatFeaturePtr(NISTMarkIIIFloatPtr& markIII, unsigned chanX, unsigned chanN, const String& nm) {
      return new NISTMarkIIIFloatFeaturePtr(new NISTMarkIIIFloatFeature(markIII, chanX, chanN, nm));
    }

    NISTMarkIIIFloatFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  NISTMarkIIIFloatFeature* operator->();
};

// ----- definition for class `timeOfDay' -----
//
%ignore timeOfDay;
class timeOfDay : public VectorFeatureStream {
  timeOfDay(const String& nm="timeOfDay");
  ~timeOfDay();
  gsl_vector* next(int frameX = -5) ;
  void reset();
};

class timeOfDayPtr : public VectorFeatureStreamPtr {
 public:
  %extend {
    timeOfDayPtr(const String& nm="timeOfDay"){
      return new timeOfDayPtr(new timeOfDay(nm));
    }

    timeOfDayPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

 timeOfDay* operator->();
};

// ----- definition for class `NISTMarkIV' -----
//
%ignore NISTMarkIV;
class NISTMarkIV {
  NISTMarkIV(int blockLen, int mk4speed= 44100);
  ~NISTMarkIV();
  void start();
  void stop();
  bool setOutputFile( const String& fn );
  void popOneBlockData();
  int  *getNthChannelData(int nth);
  void writeOneBlockData();
  int blockSize();
  int IsRecording();
};

class NISTMarkIVPtr {
 public:
  %extend {
    NISTMarkIVPtr(int blockLen, int mk4speed= 44100){
      return new NISTMarkIVPtr(new NISTMarkIV(blockLen,mk4speed));
    }
  }

  NISTMarkIV* operator->();
};

bool convertMarkIVTmpRaw2Wav( const String& ifn, const String& oprefix, int format=0x010000 | 0x0004 /* Microsoft Wave */ );

// ----- definition for class `NISTMarkIVFloatFeature' -----
//
%ignore NISTMarkIVFloatFeature;
class NISTMarkIVFloatFeature : public VectorFloatFeatureStream {
public:
  NISTMarkIVFloatFeature(NISTMarkIVPtr& mark4, unsigned chX, const unsigned firstChanX = 0, const String& nm = "NISTMarkIVFloatFeature");
  virtual ~NISTMarkIVFloatFeature();
  virtual gsl_vector_float* next(int frameX = -5) ;
  virtual void reset();
  gsl_vector_float* data();
};

class NISTMarkIVFloatFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    NISTMarkIVFloatFeaturePtr(NISTMarkIVPtr& mark4, unsigned chX, const unsigned firstChanX = 0, const String& nm = "NISTMarkIVFloatFeature") {
      return new NISTMarkIVFloatFeaturePtr(new NISTMarkIVFloatFeature(mark4, chX, firstChanX, nm));
    }

    NISTMarkIVFloatFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  NISTMarkIVFloatFeature* operator->();
};

// ----- definition for class `NISTMarkIVServer' -----
//
%ignore NISTMarkIVServer;
class NISTMarkIVServer : public VectorFloatFeatureStream {
public:
  NISTMarkIVServer(NISTMarkIVPtr& mark4, int basePort, unsigned clientN=1, const String& nm = "NISTMarkIVServer");
  ~NISTMarkIVServer();
  bool waitForConnections();
  virtual gsl_vector_float* next(int frameX = -5);
  virtual void reset();
  void setLogChanX(unsigned short logChanX){ _logChanX = logChanX;}
  bool setOutputFile( const String& prefix );
  void writeAllBlockData();
};

class NISTMarkIVServerPtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    NISTMarkIVServerPtr(NISTMarkIVPtr& mark4, int basePort, unsigned clientN=1, const String& nm = "NISTMarkIVServer"){
      return new NISTMarkIVServerPtr(new NISTMarkIVServer(mark4, basePort, clientN, nm));
    }

    NISTMarkIVServerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  NISTMarkIVServer* operator->();
};

// ----- definition for class `NISTMarkIVClienet' -----
//
%ignore NISTMarkIVClient;
class NISTMarkIVClient : public VectorFloatFeatureStream {
public:
  NISTMarkIVClient( const String& hostname, int port, unsigned chX, gsl_vector *chanIDs, unsigned blockLen = 320, int shiftLen=-1, unsigned short firstChanX = 0, const String& nm = "NISTMarkIVClient");
  ~NISTMarkIVClient();
  virtual gsl_vector_float* next(int frameX = -5);
  virtual void reset();
};

class NISTMarkIVClientPtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    NISTMarkIVClientPtr( const String& hostname, int port, unsigned chX, gsl_vector *chanIDs, unsigned blockLen = 320, int shiftLen=-1, unsigned short firstChanX = 0, const String& nm = "NISTMarkIVClient"){
      return new NISTMarkIVClientPtr(new NISTMarkIVClient( hostname, port, chX, chanIDs, blockLen, shiftLen, firstChanX, nm));
    }

    NISTMarkIVClientPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  NISTMarkIVClient* operator->();
};

#endif /* #ifdef HAVE_LIBMARKIV */

// ----- definition for class `SampleFeature' -----
// 
%ignore SampleFeature;
class SampleFeature : public VectorFloatFeatureStream {
public:
  SampleFeature(const String fn = "", unsigned blockLen = 320,
		unsigned shiftLen = 160, bool padZeros = false, const String nm = "Sample");

  unsigned read(const String& fn, int format = 0, int samplerate = 16000,
		int chX = 1, int chN = 1, int cfrom = 0, int to = -1, int outsamplerate=-1, float norm = 0.0);

  void write(const String& fn, int format = SF_FORMAT_NIST|SF_FORMAT_PCM_16, int sampleRate = -1);

  void cut(unsigned cfrom, unsigned cto);

  void copySamples(SampleFeaturePtr& src, unsigned cfrom, unsigned to);

  unsigned samplesN() const;

  int getSampleRate() const;

  int getChanN() const;

  const gsl_vector_float* data();

  const gsl_vector* dataDouble();

  virtual void reset();
  
  void exit();

  void zeroMean();

  void addWhiteNoise( float snr );

  void randomize(int startX, int endX, double sigma2);

  virtual const gsl_vector_float* next(int frameX = -5);

  void setSamples(const gsl_vector* samples, unsigned sampleRate);

  int frameX() const;
};

class SampleFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    SampleFeaturePtr(const String fn = "", unsigned blockLen = 320,
                     unsigned shiftLen = 160, bool padZeros = false, const String nm = "Sample") {
      return new SampleFeaturePtr(new SampleFeature(fn, blockLen, shiftLen, padZeros, nm));
    }

    SampleFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SampleFeature* operator->();
};

// ----- definition for class `SampleFeatureRunon' -----
// 
%ignore SampleFeatureRunon;
class SampleFeatureRunon : public SampleFeature {
public:
  SampleFeatureRunon(const String fn = "", unsigned blockLen = 320,
		     unsigned shiftLen = 160, bool padZeros = false, const String nm = "Sample");

  virtual int frameX() const;

  virtual int frameN() const;
};

class SampleFeatureRunonPtr : public SampleFeaturePtr {
 public:
  %extend {
    SampleFeatureRunonPtr(const String fn = "", unsigned blockLen = 320,
			  unsigned shiftLen = 160, bool padZeros = false, const String nm = "Sample") {
      return new SampleFeatureRunonPtr(new SampleFeatureRunon(fn, blockLen, shiftLen, padZeros, nm));
    }

    SampleFeatureRunonPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SampleFeatureRunon* operator->();
};

// ----- definition for class `IterativeSingleChannelSampleFeature' -----
// 
%ignore IterativeSingleChannelSampleFeature;
class IterativeSingleChannelSampleFeature : public VectorFloatFeatureStream {
public:
  IterativeSingleChannelSampleFeature(unsigned blockLen = 320, const String& nm = "IterativeSingleChannelSampleFeature");

  void read(const String& fileName, int format = 0, int samplerate = 44100, int cfrom = 0, int cto = -1 );

  unsigned samplesN() const { return _ttlSamples; }

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();
};

class IterativeSingleChannelSampleFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    IterativeSingleChannelSampleFeaturePtr(unsigned blockLen = 320, const String& nm = "IterativeSingleChannelSampleFeature") {
      return new IterativeSingleChannelSampleFeaturePtr(new IterativeSingleChannelSampleFeature( blockLen, nm ));
    }
    
    IterativeSingleChannelSampleFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  IterativeSingleChannelSampleFeature* operator->();
};

// ----- definition for class `IterativeSampleFeature' -----
// 
%ignore IterativeSampleFeature;
class IterativeSampleFeature : public VectorFloatFeatureStream {
public:
  IterativeSampleFeature(unsigned chX, unsigned blockLen = 320, unsigned firstChanX = 0, const String& nm = "Iterative Sample");

  void read(const String& fileName, int format = SF_FORMAT_NIST|SF_FORMAT_PCM_32, int samplerate = 44100, int chN = 64, int cfrom = 0, int cto = -1 );

  unsigned samplesN() const;

  virtual const gsl_vector_float* next(int frameX = -5);

  void changeFirstChannelID( unsigned firstChanX );
};

class IterativeSampleFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    IterativeSampleFeaturePtr(unsigned chX, unsigned blockLen = 320, unsigned firstChanX = 0, const String& nm = "Iterative Sample") {
      return new IterativeSampleFeaturePtr(new IterativeSampleFeature(chX, blockLen, firstChanX,nm));
    }

    IterativeSampleFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  IterativeSampleFeature* operator->();
};


// ----- definition for class `BlockSizeConversionFeature' -----
// 
%ignore BlockSizeConversionFeature;
class BlockSizeConversionFeature : public VectorFloatFeatureStream {
public:
  BlockSizeConversionFeature(const VectorFloatFeatureStreamPtr& src,
			     unsigned blockLen = 320,
			     unsigned shiftLen = 160, const String& nm = "Block Size Conversion");

  const gsl_vector_float* next(int frameX = -5) const;
};

class BlockSizeConversionFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    BlockSizeConversionFeaturePtr(const VectorFloatFeatureStreamPtr& src,
				  unsigned blockLen = 320,
				  unsigned shiftLen = 160, const String& nm = "Block Size Conversion") {
      return new BlockSizeConversionFeaturePtr(new BlockSizeConversionFeature(src, blockLen, shiftLen));
    }

    BlockSizeConversionFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  BlockSizeConversionFeature* operator->();
};


// ----- definition for class `BlockSizeConversionFeatureShort' -----
// 
%ignore BlockSizeConversionFeatureShort;
class BlockSizeConversionFeatureShort : public VectorShortFeatureStream {
public:
  BlockSizeConversionFeatureShort(VectorShortFeatureStreamPtr& src,
				  unsigned blockLen = 320,
				  unsigned shiftLen = 160, const String& nm = "Block Size Conversion");

  const gsl_vector_short* next() const;
};

class BlockSizeConversionFeatureShortPtr : public VectorShortFeatureStreamPtr {
 public:
  %extend {
    BlockSizeConversionFeatureShortPtr(VectorShortFeatureStreamPtr& src,
				       unsigned blockLen = 320,
				       unsigned shiftLen = 160, const String& nm = "Block Size Conversion") {
      return new BlockSizeConversionFeatureShortPtr(new BlockSizeConversionFeatureShort(src, blockLen, shiftLen));
    }

    BlockSizeConversionFeatureShortPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  BlockSizeConversionFeatureShort* operator->();
};


#ifdef SMARTFLOW

// ----- definition for class `SmartFlowFeature' -----
// 
%ignore SmartFlowFeature;
class SmartFlowFeature : public VectorShortFeatureStream {
public:
  SmartFlowFeature(sflib::sf_flow_sync* sfflow, unsigned blockLen = 320,
		   unsigned shiftLen = 160, const String& nm = "SmartFlowFeature");

  const gsl_vector_short* next() const;
};

class SmartFlowFeaturePtr : public VectorShortFeatureStreamPtr {
 public:
  %extend {
    SmartFlowFeaturePtr(sflib::sf_flow_sync* sfflow, unsigned blockLen = 320,
                        unsigned shiftLen = 160, const String& nm = "SmartFlowFeature") {
      return new SmartFlowFeaturePtr(new SmartFlowFeature(sfflow, blockLen, shiftLen, nm));
    }

    SmartFlowFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SmartFlowFeature* operator->();
};

#endif


// ----- definition for class `PreemphasisFeature' -----
// 
%ignore PreemphasisFeature;
class PreemphasisFeature : public VectorFloatFeatureStream {
public:
  PreemphasisFeature(const VectorFloatFeatureStreamPtr& samp, double mu = 0.95, const String& nm = "Preemphasis");

  const gsl_vector_float* next() const;

  void nextSpeaker();
};

class PreemphasisFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    PreemphasisFeaturePtr(const VectorFloatFeatureStreamPtr& samp, double mu = 0.95, const String& nm = "Preemphasis") {
      return new PreemphasisFeaturePtr(new PreemphasisFeature(samp, mu, nm));
    }

    PreemphasisFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  PreemphasisFeature* operator->();
};


// ----- definition for class `HammingFeatureShort' -----
// 
%ignore HammingFeatureShort;
class HammingFeatureShort : public VectorFloatFeatureStream {
public:
  HammingFeatureShort(const VectorShortFeatureStreamPtr& samp, const String& nm = "Hamming Short");

  const gsl_vector_float* next() const;
};

class HammingFeatureShortPtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    HammingFeatureShortPtr(const VectorShortFeatureStreamPtr& samp, const String& nm = "Hamming Short") {
      return new HammingFeatureShortPtr(new HammingFeatureShort(samp, nm));
    }

    HammingFeatureShortPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  HammingFeatureShort* operator->();
};


// ----- definition for class `HammingFeature' -----
// 
%ignore HammingFeature;
class HammingFeature : public VectorFloatFeatureStream {
public:
  HammingFeature(const VectorFloatFeatureStreamPtr& samp, const String& nm = "Hamming");

  const gsl_vector_float* next() const;
};

class HammingFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    HammingFeaturePtr(const VectorFloatFeatureStreamPtr& samp, const String& nm = "Hamming") {
      return new HammingFeaturePtr(new HammingFeature(samp, nm));
    }

    HammingFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  HammingFeature* operator->();
};


// ----- definition for class `FFTFeature' -----
// 
%ignore FFTFeature;
class FFTFeature : public VectorComplexFeatureStream {
public:
  FFTFeature(const VectorFloatFeatureStreamPtr samp, unsigned fftLen = 512, const String& nm = "FFT");

  const gsl_vector_complex* next(int frameX = -5) const;

  unsigned fftLen()    const;
  unsigned windowLen() const;

  unsigned nBlocks()     const;
  unsigned subSampRate() const;
};

class FFTFeaturePtr : public VectorComplexFeatureStreamPtr {
 public:
  %extend {
    FFTFeaturePtr(const VectorFloatFeatureStreamPtr samp, unsigned fftLen = 512, const String& nm = "FFT") {
      return new FFTFeaturePtr(new FFTFeature(samp, fftLen, nm));
    }

    FFTFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  FFTFeature* operator->();
};

// ----- definition for class `SpectralPowerFloatFeature' -----
// 
%ignore SpectralPowerFloatFeature;
class SpectralPowerFloatFeature : public VectorFloatFeatureStream {
public:
  SpectralPowerFloatFeature(const VectorComplexFeatureStreamPtr& fft, unsigned powN = 0, const String nm = "Power");

  const gsl_vector_float* next() const;
};

class SpectralPowerFloatFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    SpectralPowerFloatFeaturePtr(const VectorComplexFeatureStreamPtr& fft, unsigned powN = 0, const String nm = "Power") {
      return new SpectralPowerFloatFeaturePtr(new SpectralPowerFloatFeature(fft, powN, nm));
    }

    SpectralPowerFloatFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SpectralPowerFloatFeature* operator->();
};


// ----- definition for class `SpectralPowerFeature' -----
// 
%ignore SpectralPowerFeature;
class SpectralPowerFeature : public VectorFeatureStream {
public:
  SpectralPowerFeature(const VectorComplexFeatureStreamPtr& fft, unsigned powN = 0, const String nm = "Power");

  const gsl_vector* next() const;
};

class SpectralPowerFeaturePtr : public VectorFeatureStreamPtr {
 public:
  %extend {
    SpectralPowerFeaturePtr(const VectorComplexFeatureStreamPtr& fft, unsigned powN = 0, const String nm = "Power") {
      return new SpectralPowerFeaturePtr(new SpectralPowerFeature(fft, powN, nm));
    }

    SpectralPowerFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SpectralPowerFeature* operator->();
};


// ----- definition for class `SignalPowerFeature' -----
//
%ignore SignalPowerFeature;
class SignalPowerFeature : public VectorFloatFeatureStream {
public:
  SignalPowerFeature(const VectorFloatFeatureStreamPtr& samp, const String nm = "Signal Power");

  const gsl_vector_float* next() const;
};

class SignalPowerFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    SignalPowerFeaturePtr(const VectorFloatFeatureStreamPtr& samp, const String nm = "Signal Power") {
      return new SignalPowerFeaturePtr(new SignalPowerFeature(samp, nm));
    }

    SignalPowerFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SignalPowerFeature* operator->();
};


// ----- definition for class `ALogFeature' -----
//
%ignore ALogFeature;
class ALogFeature : public VectorFloatFeatureStream {
public:
  ALogFeature(const VectorFloatFeatureStreamPtr& samp, double m = 1.0, double a = 4.0,
	      bool runon = false, const String nm = "ALog Power");

  void nextSpeaker();

  const gsl_vector_float* next() const;
};

class ALogFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    ALogFeaturePtr(const VectorFloatFeatureStreamPtr& samp, double m = 1.0, double a = 4.0,
		   bool runon = false, const String nm = "ALog Power") {
      return new ALogFeaturePtr(new ALogFeature(samp, m, a, runon, nm));
    }

    ALogFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  ALogFeature* operator->();
};


// ----- definition for class `NormalizeFeature' -----
//
%ignore NormalizeFeature;
class NormalizeFeature : public VectorFloatFeatureStream {
public:
  NormalizeFeature(const VectorFloatFeatureStreamPtr& samp, double min = 0.0, double max = 1.0,
		   bool runon = false, const String nm = "Normalize");

  const gsl_vector_float* next() const;

  void nextSpeaker();
};

class NormalizeFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    NormalizeFeaturePtr(const VectorFloatFeatureStreamPtr& samp, double min = 0.0, double max = 1.0,
			bool runon = false, const String nm = "Normalize") {
      return new NormalizeFeaturePtr(new NormalizeFeature(samp, min, max, runon, nm));
    }

    NormalizeFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  NormalizeFeature* operator->();
};


// ----- definition for class `ThresholdFeature' -----
//
%ignore ThresholdFeature;
class ThresholdFeature : public VectorFloatFeatureStream {
public:
  ThresholdFeature(const VectorFloatFeatureStreamPtr& samp, double value = 0.0, double thresh = 1.0,
		   const String& mode = "upper", const String& nm = "Threshold");

  const gsl_vector_float* next() const;
};

class ThresholdFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    ThresholdFeaturePtr(const VectorFloatFeatureStreamPtr& samp, double value = 0.0, double thresh = 1.0,
			const String& mode = "upper", const String& nm = "Threshold") {
      return new ThresholdFeaturePtr(new ThresholdFeature(samp, value, thresh, mode, nm));
    }

    ThresholdFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  ThresholdFeature* operator->();
};


// ----- definition for class `SpectralResamplingFeature' -----
//
%ignore SpectralResamplingFeature;
class SpectralResamplingFeature : public VectorFeatureStream {
 public:
  SpectralResamplingFeature(const VectorFeatureStreamPtr& src, double ratio, unsigned len = 0,
			    const String& nm = "Resampling");

  virtual ~SpectralResamplingFeature();

  virtual const gsl_vector* next(int frameX = -5);
};

class SpectralResamplingFeaturePtr : public VectorFeatureStreamPtr {
 public:
  %extend {
    SpectralResamplingFeaturePtr(const VectorFeatureStreamPtr& src, double ratio, unsigned len = 0,
				 const String& nm = "Resampling") {
      return new SpectralResamplingFeaturePtr(new SpectralResamplingFeature(src, ratio, len, nm));
    }

    SpectralResamplingFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SpectralResamplingFeature* operator->();
};


#ifdef SRCONV

// ----- definition for class `SamplerateConversionFeature' -----
//
%ignore SamplerateConversionFeature;
class SamplerateConversionFeature : public VectorFloatFeatureStream {
 public:
  SamplerateConversionFeature(const VectorFloatFeatureStreamPtr& src, unsigned sourcerate = 22050, unsigned destrate = 16000,
			      unsigned len = 0, const String& method = "fastest", const String& nm = "SamplerateConversion");

  virtual ~SamplerateConversionFeature() { }

  virtual const gsl_vector_float* next(int frameX = -5);
};

class SamplerateConversionFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    SamplerateConversionFeaturePtr(const VectorFloatFeatureStreamPtr& src, unsigned sourcerate = 22050, unsigned destrate = 16000,
				   unsigned len = 0, const String& method = "fastest", const String& nm = "SamplerateConversion") {
      return new SamplerateConversionFeaturePtr(new SamplerateConversionFeature(src, sourcerate, destrate, len, method, nm));
    }

    SamplerateConversionFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SamplerateConversionFeature* operator->();
};

#endif


// ----- definition for class `VTLNFeature' -----
// 
%ignore VTLNFeature;
class VTLNFeature : public VectorFeatureStream {
public:
  VTLNFeature(const VectorFeatureStreamPtr& pow,
              unsigned coeffN = 0, double ratio = 1.0, double edge = 1.0, int version = 1,
              const String& nm = "VTLN");

  virtual const gsl_vector* next(int frameX = -5) const;

  void matrix(gsl_matrix* matrix) const;

  // specify the warp factor
  void warp(double w);
};

class VTLNFeaturePtr : public VectorFeatureStreamPtr {
 public:
  %extend {
    VTLNFeaturePtr(const VectorFeatureStreamPtr& pow,
                         unsigned coeffN = 0, double ratio = 1.0, double edge = 1.0, int version = 1,
		         const String& nm = "VTLN") {
      return new VTLNFeaturePtr(new VTLNFeature(pow, coeffN, ratio, edge, version, nm));
    }

    VTLNFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  VTLNFeature* operator->();
};


// ----- definition for class `MelFeature' -----
// 
%ignore MelFeature;
class MelFeature : public VectorFeatureStream {
public:
  MelFeature(const VectorFeatureStreamPtr mag, int powN = 0,
             float rate = 16000.0, float low = 0.0, float up = 0.0,
             int filterN = 30, int version = 1, const String& nm = "MelFFT");

  const gsl_vector* next() const;

  void read(const String& fileName);

  void matrix(gsl_matrix* matrix) const;
};

class MelFeaturePtr : public VectorFeatureStreamPtr {
 public:
  %extend {
    MelFeaturePtr(const VectorFeatureStreamPtr mag, int powN = 0,
                  float rate = 16000.0, float low = 0.0, float up = 0.0,
                  int filterN = 30, int version = 1, const String& nm = "MelFFT") {
      return new MelFeaturePtr(new MelFeature(mag, powN, rate, low, up, filterN, version, nm));
    }

    MelFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  MelFeature* operator->();
};


// ----- definition for class `SphinxMelFeature' -----
// 
%ignore SphinxMelFeature;
class SphinxMelFeature : public VectorFeatureStream {
public:
  SphinxMelFeature(const VectorFeatureStreamPtr& mag, unsigned fftN = 512, unsigned powerN = 0,
		   float sampleRate = 16000.0, float lowerF = 0.0, float upperF = 0.0,
		   unsigned filterN = 30, const String& nm = "Sphinx Mel Filter Bank");

  const gsl_vector* next() const;
};

class SphinxMelFeaturePtr : public VectorFeatureStreamPtr {
 public:
  %extend {
    SphinxMelFeaturePtr(const VectorFeatureStreamPtr& mag, unsigned fftN = 512, unsigned powerN = 0,
			float sampleRate = 16000.0, float lowerF = 0.0, float upperF = 0.0,
			unsigned filterN = 30, const String& nm = "Sphinx Mel Filter Bank") {
      return new SphinxMelFeaturePtr(new SphinxMelFeature(mag, fftN, powerN, sampleRate, lowerF, upperF, filterN, nm));
    }

    SphinxMelFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SphinxMelFeature* operator->();
};


// ----- definition for class `LogFeature' -----
// 
%ignore LogFeature;
class LogFeature : public VectorFloatFeatureStream {
public:
  LogFeature(const VectorFeatureStreamPtr& mel, double m = 1.0, double a = 1.0,
             bool sphinxFlooring = false, const String& nm = "LogMel");

  const gsl_vector_float* next() const;
};

class LogFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    LogFeaturePtr(const VectorFeatureStreamPtr& mel, double m = 1.0, double a = 1.0,
                  bool sphinxFlooring = false, const String& nm = "LogMel") {
      return new LogFeaturePtr(new LogFeature(mel, m, a, sphinxFlooring, nm));
    }

    LogFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  LogFeature* operator->();
};

// ----- definition for class `FloatToDoubleConversionFeature' -----
//
%ignore FloatToDoubleConversionFeature;
class FloatToDoubleConversionFeature : public VectorFeatureStream {
 public:
  FloatToDoubleConversionFeature(const VectorFloatFeatureStreamPtr& src, const String& nm = "Float to Double Conversion");

  virtual ~FloatToDoubleConversionFeature();

  virtual const gsl_vector* next(int frameX = -5);

  virtual void reset();

};

class FloatToDoubleConversionFeaturePtr : public VectorFeatureStreamPtr {
 public:
  %extend {
  FloatToDoubleConversionFeaturePtr(const VectorFloatFeatureStreamPtr& src, const String& nm = "Float to Double Conversion") {
      return new FloatToDoubleConversionFeaturePtr(new FloatToDoubleConversionFeature(src, nm));
    }

    FloatToDoubleConversionFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  FloatToDoubleConversionFeaturePtr* operator->();
};


// ----- definition for class `CepstralFeature' -----
// 
%ignore CepstralFeature;
// type:
//   0 = 
//   1 = Type 2 DCT
//   2 = Sphinx Legacy
class CepstralFeature : public VectorFloatFeatureStream {
public:
  CepstralFeature(const VectorFloatFeatureStreamPtr mel, unsigned ncep = 13, int type = 1, const String nm = "Cepstral");

  const gsl_vector_float* next() const;

  gsl_matrix* matrix() const;
};

class CepstralFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    CepstralFeaturePtr(const VectorFloatFeatureStreamPtr mel, unsigned ncep = 13, int type = 1, const String nm = "Cepstral") {
      return new CepstralFeaturePtr(new CepstralFeature(mel, ncep, type, nm));
    }

    CepstralFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  CepstralFeature* operator->();
};


// ----- definition for class `WarpMVDRFeature' -----
// 
%ignore WarpMVDRFeature;
class WarpMVDRFeature : public VectorFeatureStream {
public:
  WarpMVDRFeature(const VectorFloatFeatureStreamPtr& src, unsigned order = 60, unsigned correlate = 0, float warp = 0.0,
		  const String& nm = "MVDR");

  const gsl_vector_float* next() const;
};

class WarpMVDRFeaturePtr : public VectorFeatureStreamPtr {
 public:
  %extend {
    WarpMVDRFeaturePtr(const VectorFloatFeatureStreamPtr& src, unsigned order = 60, unsigned correlate = 0, float warp = 0.0,
		       const String& nm = "MVDR") {
      return new WarpMVDRFeaturePtr(new WarpMVDRFeature(src, order, correlate, warp, nm));
    }

    WarpMVDRFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  WarpMVDRFeature* operator->();
};


// ----- definition for class `BurgMVDRFeature' -----
// 
%ignore BurgMVDRFeature;
class BurgMVDRFeature : public VectorFeatureStream {
public:
  BurgMVDRFeature(const VectorFloatFeatureStreamPtr& src, unsigned order = 60, unsigned correlate = 0, float warp = 0.0,
		  const String& nm = "MVDR");

  const gsl_vector_float* next() const;
};

class BurgMVDRFeaturePtr : public VectorFeatureStreamPtr {
 public:
  %extend {
    BurgMVDRFeaturePtr(const VectorFloatFeatureStreamPtr& src, unsigned order = 60, unsigned correlate = 0, float warp = 0.0,
		       const String& nm = "MVDR") {
      return new BurgMVDRFeaturePtr(new BurgMVDRFeature(src, order, correlate, warp, nm));
    }

    BurgMVDRFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  BurgMVDRFeature* operator->();
};


// ----- definition for class `WarpedTwiceFeature' -----
//
%ignore WarpedTwiceMVDRFeature;
class WarpedTwiceMVDRFeature : public VectorFeatureStream {
public:
  WarpedTwiceMVDRFeature(const VectorFloatFeatureStreamPtr& src, unsigned order = 60, unsigned correlate = 0, bool warpFactorFixed=false, float sensibility = 0.1, const String& nm = "WTMVDR");
  virtual ~WarpedTwiceMVDRFeature();

  virtual const gsl_vector* next(int frameX = -5);

  virtual void reset();

};

class WarpedTwiceMVDRFeaturePtr : public VectorFeatureStreamPtr {
 public:
  %extend {
    WarpedTwiceMVDRFeaturePtr(const VectorFloatFeatureStreamPtr& src, unsigned order = 60, unsigned correlate = 0, float warp = 0.0, bool warpFactorFixed=false, float sensibility = 0.1, const String& nm = "WTMVDR"){
      return new WarpedTwiceMVDRFeaturePtr(new WarpedTwiceMVDRFeature( src, order, correlate, warp, warpFactorFixed, sensibility, nm));
    }
    
    WarpedTwiceMVDRFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  WarpedTwiceMVDRFeature* operator->();
};


// ----- definition for class `WarpLPCFeature' -----
// 
%ignore WarpLPCFeature;
class WarpLPCFeature : public VectorFeatureStream {
public:
  WarpLPCFeature(const VectorFloatFeatureStreamPtr& src, unsigned order, unsigned correlate, float warp,
		 const String& nm = "LPC");

  const gsl_vector* next() const;
};

class WarpLPCFeaturePtr : public VectorFeatureStreamPtr {
 public:
  %extend {
    WarpLPCFeaturePtr(const VectorFloatFeatureStreamPtr& src, unsigned order, unsigned correlate, float warp,
		      const String& nm = "LPC") {
      return new WarpLPCFeaturePtr(new WarpLPCFeature(src, order, correlate, warp, nm));
    }

    WarpLPCFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  WarpLPCFeature* operator->();
};


// ----- definition for class `BurgLPCFeature' -----
// 
%ignore BurgLPCFeature;
class BurgLPCFeature : public VectorFeatureStream {
public:
  BurgLPCFeature(const VectorFloatFeatureStreamPtr& src, unsigned order, unsigned correlate, float warp,
		 const String& nm = "LPC");

  const gsl_vector* next() const;
};

class BurgLPCFeaturePtr : public VectorFeatureStreamPtr {
 public:
  %extend {
    BurgLPCFeaturePtr(const VectorFloatFeatureStreamPtr& src, unsigned order, unsigned correlate, float warp,
		      const String& nm = "LPC") {
      return new BurgLPCFeaturePtr(new BurgLPCFeature(src, order, correlate, warp, nm));
    }

    BurgLPCFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  BurgLPCFeature* operator->();
};


// ----- definition for class `SpectralSmoothing' -----
// 
%ignore SpectralSmoothing;
class SpectralSmoothing : public VectorFeatureStream {
public:
  SpectralSmoothing(const VectorFeatureStreamPtr& adjustTo, const VectorFeatureStreamPtr& adjustFrom,
		    const String& nm = "Spectral Smoothing");

  const gsl_vector* next() const;
};

class SpectralSmoothingPtr : public VectorFeatureStreamPtr {
 public:
  %extend {
    SpectralSmoothingPtr(const VectorFeatureStreamPtr& adjustTo, const VectorFeatureStreamPtr& adjustFrom,
			 const String& nm = "Spectral Smoothing") {
      return new SpectralSmoothingPtr(new SpectralSmoothing(adjustTo, adjustFrom, nm));
    }

    SpectralSmoothingPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SpectralSmoothing* operator->();
};


// ----- definition for class `MeanSubtractionFeature' -----
//
%ignore MeanSubtractionFeature;
class MeanSubtractionFeature : public VectorFloatFeatureStream {
 public:
  MeanSubtractionFeature(const VectorFloatFeatureStreamPtr& src, const VectorFloatFeatureStreamPtr& weight = NULL,
			 double devNormFactor = 0.0, bool runon = false, const String& nm = "Mean Subtraction");

  virtual ~MeanSubtractionFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  void nextSpeaker();

  const gsl_vector_float* mean() const;

  void write(const String& fileName, bool variance = false) const;
};

class MeanSubtractionFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    MeanSubtractionFeaturePtr(const VectorFloatFeatureStreamPtr& src, const VectorFloatFeatureStreamPtr& weight = NULL,
			      double devNormFactor = 0.0, bool runon = false, const String nm = "Mean Subtraction") {
      return new MeanSubtractionFeaturePtr(new MeanSubtractionFeature(src, weight, devNormFactor, runon, nm));
    }

    MeanSubtractionFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  MeanSubtractionFeature* operator->();
};


// ----- definition for class `FileMeanSubtractionFeature' -----
//
%ignore FileMeanSubtractionFeature;
class FileMeanSubtractionFeature : public VectorFloatFeatureStream {
 public:
  FileMeanSubtractionFeature(const VectorFloatFeatureStreamPtr& src,
			     double devNormFactor = 0.0, const String& nm = "MeanSubtraction");

  virtual ~FileMeanSubtractionFeature();
  
  virtual const gsl_vector_float* next(int frameX = -5);

  void read(const String& fileName, bool variance = false);
};

class FileMeanSubtractionFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    FileMeanSubtractionFeaturePtr(const VectorFloatFeatureStreamPtr& src,
				  double devNormFactor = 0.0, const String nm = "MeanSubtraction") {
      return new FileMeanSubtractionFeaturePtr(new FileMeanSubtractionFeature(src, devNormFactor, nm));
    }

    FileMeanSubtractionFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  FileMeanSubtractionFeature* operator->();
};


// ----- definition for class `AdjacentFeature' -----
// 
%ignore AdjacentFeature;
class AdjacentFeature : public VectorFloatFeatureStream {
public:
  AdjacentFeature(const VectorFloatFeatureStreamPtr& single, unsigned delta = 5,
		  const String& nm = "Adjacent");

  virtual ~AdjacentFeature();

  const gsl_vector_float* next() const;
};

class AdjacentFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    AdjacentFeaturePtr(const VectorFloatFeatureStreamPtr& single, unsigned delta = 5,
		       const String& nm = "Adjacent") {
      return new AdjacentFeaturePtr(new AdjacentFeature(single, delta, nm));
    }

    AdjacentFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  AdjacentFeature* operator->();
};


// ----- definition for class `LinearTransformFeature' -----
//
%ignore LinearTransformFeature;
class LinearTransformFeature : public VectorFloatFeatureStream {
 public:
#if 0
  LinearTransformFeature(const VectorFloatFeatureStreamPtr& src,
			 gsl_matrix_float* mat = NULL, unsigned sz = 0, const String& nm = "Transform");
#else
  LinearTransformFeature(const VectorFloatFeatureStreamPtr& src, unsigned sz = 0, const String& nm = "Transform");
#endif

  virtual ~LinearTransformFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  gsl_matrix_float* matrix() const;

  void load(const String& fileName, bool old = false);

  void identity();
};

class LinearTransformFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
#if 0
    LinearTransformFeaturePtr(const VectorFloatFeatureStreamPtr& src,
                              gsl_matrix_float* mat = NULL, unsigned sz = 0, const String& nm = "Transform") {

      cout << "Allocating 'LinearTransformFeaturePtr'" << endl;
      return new LinearTransformFeaturePtr(new LinearTransformFeature(src, mat, sz, nm));
    }
#else
    LinearTransformFeaturePtr(const VectorFloatFeatureStreamPtr& src, unsigned sz = 0, const String& nm = "Transform") {

      cout << "Allocating 'LinearTransformFeaturePtr'" << endl;
      return new LinearTransformFeaturePtr(new LinearTransformFeature(src, sz, nm));
    }
#endif

    LinearTransformFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  LinearTransformFeature* operator->();
};


// ----- definition for class `StorageFeature' -----
//
%ignore StorageFeature;
class StorageFeature : public VectorFloatFeatureStream {
  typedef vector<gsl_vector_float*>	_StorageVector;
  static const int MaxFrames;
 public:
  StorageFeature(const VectorFloatFeatureStreamPtr& src, const String& nm = "Storage");

  virtual ~StorageFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  void write(const String& fileName, bool plainText = false) const;

  void read(const String& fileName);

  int evaluate();

  void reset();

};

class StorageFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    StorageFeaturePtr(const VectorFloatFeatureStreamPtr& src = NULL, const String& nm = "Storage") {
      return new StorageFeaturePtr(new StorageFeature(src, nm));
    }

    StorageFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  StorageFeature* operator->();
};

// ----- definition for class `StaticStorageFeature' -----
//
%ignore StaticStorageFeature;
class StaticStorageFeature : public VectorFloatFeatureStream {
  typedef vector<gsl_vector_float*> _StorageVector;
  static const int MaxFrames;
 public:
  StaticStorageFeature(unsigned dim, const String& nm = "Storage");

  virtual ~StaticStorageFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  //void write(const String& fileName) const;

  void read(const String& fileName);

  int evaluate();
  
  unsigned currentNFrames();

  void reset();
};

class StaticStorageFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    StaticStorageFeaturePtr(unsigned dim, const String& nm = "Storage") {
      return new StaticStorageFeaturePtr(new StaticStorageFeature(dim, nm));
    }

    StaticStorageFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  StaticStorageFeature* operator->();
};

// ----- definition for class `CircularStorageFeature' -----
//
%ignore CircularStorageFeature;
class CircularStorageFeature : public VectorFloatFeatureStream {
  typedef vector<gsl_vector_float*>	_StorageVector;
  static const int MaxFrames;
 public:
  CircularStorageFeature(const VectorFloatFeatureStreamPtr& src, unsigned framesN = 3,
			 const String& nm = "Storage");

  virtual ~CircularStorageFeature();

  virtual const gsl_vector_float* next(int frameX = -5);
};

class CircularStorageFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    CircularStorageFeaturePtr(const VectorFloatFeatureStreamPtr& src, int framesN = 3,
			      const String& nm = "Storage") {
      return new CircularStorageFeaturePtr(new CircularStorageFeature(src, framesN, nm));
    }

    CircularStorageFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  CircularStorageFeature* operator->();
};


// ----- definition for class `FilterFeature' -----
//
%ignore FilterFeature;
class FilterFeature : public VectorFloatFeatureStream {
 public:
  FilterFeature(const VectorFloatFeatureStreamPtr& src,	gsl_vector* coeffA,
		const String& nm = "Filter");
  virtual ~FilterFeature();

  virtual const gsl_vector_float* next(int frameX = -5);
};

class FilterFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    FilterFeaturePtr(const VectorFloatFeatureStreamPtr& src,
		     gsl_vector* coeffA, const String& nm = "Filter") {
      return new FilterFeaturePtr(new FilterFeature(src, coeffA, nm));
    }

    FilterFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  FilterFeature* operator->();
};


// ----- definition for class `MergeFeature' -----
//
%ignore MergeFeature;
class MergeFeature : public VectorFloatFeatureStream {
 public:
  MergeFeature(VectorFloatFeatureStreamPtr& stat,
	       VectorFloatFeatureStreamPtr& delta,
	       VectorFloatFeatureStreamPtr& deltaDelta,
	       const String& nm = "Merge");

  virtual ~MergeFeature();

  virtual const gsl_vector_float* next(int frameX = -5);
};

class MergeFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    MergeFeaturePtr(VectorFloatFeatureStreamPtr& stat,
                    VectorFloatFeatureStreamPtr& delta,
                    VectorFloatFeatureStreamPtr& deltaDelta,
                    const String& nm = "Merge") {
      return new MergeFeaturePtr(new MergeFeature(stat, delta, deltaDelta, nm));
    }

    MergeFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  MergeFeature* operator->();
};

// ----- definition for class `MultiModalFeature' -----
//
%ignore MultiModalFeature;
class MultiModalFeature : public VectorFloatFeatureStream {
 public:
  MultiModalFeature(unsigned nModality, unsigned totalVecSize, const String& nm = "MultiModal");

  virtual ~MultiModalFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  void addModalFeature( VectorFloatFeatureStreamPtr &feature, unsigned samplePeriodinNanoSec=1 );
};

class MultiModalFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    MultiModalFeaturePtr(unsigned nModality, unsigned totalVecSize,
			 const String& nm = "MultiModal") {
      return new MultiModalFeaturePtr(new MultiModalFeature(nModality, totalVecSize, nm));
    }

    MultiModalFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  MultiModalFeature* operator->();
};

// ----- definition for class `FeatureSet' -----
// 
%ignore FeatureSet;
class FeatureSet {
public:
  FeatureSet(const String nm = "FeatureSet");

  const String name() const;

  void add(VectorFloatFeatureStreamPtr feat);
  VectorFloatFeatureStreamPtr feature(const String nm);
};

class FeatureSetPtr {
 public:
  %extend {
    FeatureSetPtr(const String nm = "FeatureSet") {
      return new FeatureSetPtr(new FeatureSet(nm));
    }

    // return a codebook
    VectorFloatFeatureStreamPtr __getitem__(const String name) {
      return (*self)->feature(name);
    }
  }

  FeatureSet* operator->();
};

void writeGSLMatrix(const String& fileName, const gsl_matrix* mat);

#ifdef JACK
#include <vector>
#include <jack/jack.h>
#include <jack/ringbuffer.h>

typedef struct {
  jack_port_t *port;
  jack_ringbuffer_t *buffer;
  unsigned buffersize;
  unsigned overrun;
  bool can_process;
} jack_channel_t;

%ignore Jack;
class Jack
{
 public:
  Jack(const String& nm);
  ~Jack();
  jack_channel_t* addPort(unsigned buffersize, const String& connection, const String& nm);
  void start(void) { can_capture = true; };
  unsigned getSampleRate();
};

class JackPtr {
 public:
  %extend {
    JackPtr(const String& nm) {
      return new JackPtr(new Jack(nm));
    }
  }

  Jack* operator->();
};

%ignore JackFeature;
class JackFeature : public VectorFloatFeatureStream {
 public:
  JackFeature(JackPtr& jack, unsigned blockLen, unsigned buffersize,
	      const String& connection, const String& nm);

  virtual ~JackFeature() { };

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();
};

class JackFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    JackFeaturePtr(JackPtr& jack, unsigned blockLen, unsigned buffersize,
		   const String& connection, const String& nm) {
      return new JackFeaturePtr(new JackFeature(jack, blockLen, buffersize, connection, nm));
    }

    JackFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  JackFeature* operator->();
};

#endif


// ----- definition for class `ZeroCrossingRateHammingFeature' -----
// 
%ignore ZeroCrossingRateHammingFeature;
class ZeroCrossingRateHammingFeature : public VectorFloatFeatureStream {
public:
  ZeroCrossingRateHammingFeature(const VectorFloatFeatureStreamPtr& samp, const String& nm = "Zero Crossing Rate Hamming");

  const gsl_vector_float* next() const;
};

class ZeroCrossingRateHammingFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    ZeroCrossingRateHammingFeaturePtr(const VectorFloatFeatureStreamPtr& samp, const String& nm = "Zero Crossing Rate Hamming") {
      return new ZeroCrossingRateHammingFeaturePtr(new ZeroCrossingRateHammingFeature(samp, nm));
    }

    ZeroCrossingRateHammingFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  ZeroCrossingRateHammingFeature* operator->();
};

// ----- definition for class `YINPitchFeature' -----
// 
%ignore YINPitchFeature;
class YINPitchFeature : public VectorFloatFeatureStream {
public:
  YINPitchFeature(const VectorFloatFeatureStreamPtr& samp, unsigned samplerate = 16000, float threshold = 0.5, const String& nm = "YIN Pitch");

  const gsl_vector_float* next() const;
};

class YINPitchFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    YINPitchFeaturePtr(const VectorFloatFeatureStreamPtr& samp, unsigned samplerate = 16000, float threshold = 0.5, const String& nm = "YIN Pitch") {
      return new YINPitchFeaturePtr(new YINPitchFeature(samp, samplerate, threshold, nm));
    }

    YINPitchFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  YINPitchFeature* operator->();
};


// ----- definition for class `SpikeFilter' -----
// 
%ignore SpikeFilter;
class SpikeFilter : public VectorFloatFeatureStream {
public:
  SpikeFilter(VectorFloatFeatureStreamPtr& src, unsigned tapN = 3, const String& nm = "Spike Filter");

  unsigned size() const;

  const gsl_vector_float* next() const;
};

class SpikeFilterPtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    SpikeFilterPtr(VectorFloatFeatureStreamPtr& src, unsigned tapN = 3, const String nm = "Spike Filter") {
      return new SpikeFilterPtr(new SpikeFilter(src, tapN, nm));
    }

    SpikeFilterPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SpikeFilter* operator->();
};


// ----- definition for class `SpikeFilter2' -----
// 
%ignore SpikeFilter2;
class SpikeFilter2 : public VectorFloatFeatureStream {
public:
  SpikeFilter2(VectorFloatFeatureStreamPtr& src,
	       unsigned width = 3, float maxslope = 7000.0, float startslope = 100.0, float thresh = 15.0, float alpha = 0.2, unsigned verbose = 1,
	       const String& nm = "Spike Filter 2");

  unsigned size() const;

  const gsl_vector_float* next() const;

  unsigned spikesN() const { return _count; }
};

class SpikeFilter2Ptr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    SpikeFilter2Ptr(VectorFloatFeatureStreamPtr& src,
		    unsigned width = 3, float maxslope = 7000.0, float startslope = 100.0, float thresh = 15.0, float alpha = 0.2, unsigned verbose = 1,
		    const String& nm = "Spike Filter 2") {
      return new SpikeFilter2Ptr(new SpikeFilter2(src, width, maxslope, startslope, thresh, alpha, verbose, nm));
    }

    SpikeFilter2Ptr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SpikeFilter2* operator->();
};


// ----- definition for class `SoundFile' -----
// 
%ignore SoundFile;
class SoundFile {
 public:
  SoundFile(const String& fn,
	    int mode,
	    int format = 0,
	    int samplerate = 16000,
	    int channels = 1,
	    bool normalize = false);
  ~SoundFile();
  sf_count_t frames() const;
  int samplerate() const;
  int channels() const;
  int format() const;
  int sections() const;
  int seekable() const;
  sf_count_t readf(float *ptr, sf_count_t frames);
  sf_count_t writef(float *ptr, sf_count_t frames);
  sf_count_t read(float *ptr, sf_count_t items);
  sf_count_t write(float *ptr, sf_count_t items);
  sf_count_t seek(sf_count_t frames, int whence = SEEK_SET);
};

class SoundFilePtr {
 public:
  %extend {
    SoundFilePtr(const String& fn,
		 int mode,
		 int format = 0,
		 int samplerate = 16000,
		 int channels = 1,
		 bool normalize = false) {
      return new SoundFilePtr(new SoundFile(fn,
					    mode,
					    format,
					    samplerate,
					    channels,
					    normalize));
    }
  }

  SoundFile* operator->();
};

// ----- definition for class `DirectSampleFeature' -----
// 
%ignore DirectSampleFeature;
class DirectSampleFeature : public VectorFloatFeatureStream {
 public:
  DirectSampleFeature(const SoundFilePtr &sndfile,
		      unsigned blockLen = 320,
		      unsigned start = 0,
		      unsigned end = (unsigned)-1,
		      const String& nm = "DirectSample");
  ~DirectSampleFeature();
  virtual const gsl_vector_float* next(int frameX = -5);
  int sampleRate() const;
  int channels() const;
  void setRegion(unsigned start = 0, unsigned end = (unsigned)-1);
  virtual void reset();
};

class DirectSampleFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    DirectSampleFeaturePtr(const SoundFilePtr &sndfile,
			   unsigned blockLen = 320,
			   unsigned start = 0,
			   unsigned end = (unsigned)-1,
			   const String& nm = "DirectSample") {
      return new DirectSampleFeaturePtr(new DirectSampleFeature(sndfile, blockLen, start, end, nm));
    }

    DirectSampleFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  DirectSampleFeature* operator->();
};

// ----- definition for class `DirectSampleOutputFeature' -----
// 
%ignore DirectSampleOutputFeature;
class DirectSampleOutputFeature : public VectorFloatFeatureStream {
 public:
  DirectSampleOutputFeature(const VectorFloatFeatureStreamPtr& src,
			    const SoundFilePtr &sndfile,
			    const String& nm = "DirectSampleOutput");
  ~DirectSampleOutputFeature();
  virtual const gsl_vector_float* next(int frameX = -5);
  virtual void reset();
};

class DirectSampleOutputFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    DirectSampleOutputFeaturePtr(const VectorFloatFeatureStreamPtr& src,
				 const SoundFilePtr &sndfile,
				 const String& nm = "DirectSampleOutput") {
      return new DirectSampleOutputFeaturePtr(new DirectSampleOutputFeature(src, sndfile, nm));
    }

    DirectSampleOutputFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  DirectSampleOutputFeature* operator->();
};

// ----- definition for class `ChannelExtractionFeature' -----
// 
%ignore ChannelExtractionFeature;
class ChannelExtractionFeature : public VectorFloatFeatureStream {
 public:
  ChannelExtractionFeature(const VectorFloatFeatureStreamPtr& src,
			   unsigned chX = 0,
			   unsigned chN = 1,
			   const String& nm = "ChannelExtraction");
  virtual const gsl_vector_float* next(int frameX = -5);
};

class ChannelExtractionFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    ChannelExtractionFeaturePtr(const VectorFloatFeatureStreamPtr& src,
				unsigned chX = 0,
				unsigned chN = 1,
			        const String& nm = "ChannelExtraction") {
      return new ChannelExtractionFeaturePtr(new ChannelExtractionFeature(src, chX, chN, nm));
    }

    ChannelExtractionFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  ChannelExtractionFeature* operator->();
};


// ----- definition for class 'SampleInterferenceFeature -----
//
%ignore SignalInterferenceFeature;
class SignalInterferenceFeature : public VectorFloatFeatureStream {
public:
  SignalInterferenceFeature(VectorFloatFeatureStreamPtr& signal, VectorFloatFeatureStreamPtr& interference,
			    double dBInterference = 0.0, unsigned blockLen = 512, const String& nm = "Signal Interference");
  
  virtual const gsl_vector_float* next(int frameX = -5);
};

class SignalInterferenceFeaturePtr : public VectorFloatFeatureStreamPtr {
public:
  %extend {
    SignalInterferenceFeaturePtr(VectorFloatFeatureStreamPtr& signal, VectorFloatFeatureStreamPtr& interference,
				 double dBInterference = 0.0, unsigned blockLen = 512, const String& nm = "Signal Interference") {
      return new SignalInterferenceFeaturePtr(new SignalInterferenceFeature(signal, interference, dBInterference, blockLen, nm));
    }
    SignalInterferenceFeaturePtr __iter__() {
      (*self)->reset(); return *self;
    }
  }
  SignalInterferenceFeature* operator->();
};


// ----- definition for class `AmplificationFeature' -----
//
%ignore AmplificationFeature;
class AmplificationFeature : public VectorFloatFeatureStream {
 public:
  AmplificationFeature(const VectorFloatFeatureStreamPtr& src,
		       double amplify = 1.0,
		       const String& nm = "Amplification");
  virtual ~AmplificationFeature();
  virtual const gsl_vector_float* next(int frameX = -5);
  virtual void reset();
};

class AmplificationFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    AmplificationFeaturePtr(const VectorFloatFeatureStreamPtr& src,
			    double amplify = 1.0,
			    const String& nm = "Amplification") {
      return new AmplificationFeaturePtr(new AmplificationFeature(src,
								  amplify,
								  nm));
    }

    AmplificationFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  AmplificationFeature* operator->();
};


// ----- definition for class `CaptureClient' -----
//
%ignore CaptureClient;
class CaptureClient : public VectorFloatFeatureStream {
 public:
  CaptureClient(const char *ip, int port, unsigned blockLen, const String& nm);
  virtual ~CaptureClient();
  virtual const gsl_vector_float* next(int frameX = -5);
  virtual void reset();  
};

class CaptureClientPtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    CaptureClientPtr(const char *ip, int port, unsigned blockLen = 512, const String& nm = "LocalCaptureClient") {    	 
      return new CaptureClientPtr(new CaptureClient(ip, port, blockLen, nm));
    }
      
    CaptureClientPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }
  CaptureClient* operator->();
};


// ----- definition for class `LPCSpectrumEstimator' -----
//
%ignore LPCSpectrumEstimator;
class LPCSpectrumEstimator : public VectorFloatFeatureStream {
 public:
  LPCSpectrumEstimator(const VectorFloatFeatureStreamPtr& source, unsigned order, unsigned fftLen, const String& nm = "LPC Spectrum Estimator");
  virtual ~LPCSpectrumEstimator();

  virtual const gsl_vector_float* next(int frameX = -5);
  const gsl_vector_float* getLPCs() const;
  virtual void reset();  
  const gsl_vector_float* getAutoCorrelationVector();
  float getPredictionError();
};

class LPCSpectrumEstimatorPtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    LPCSpectrumEstimatorPtr(const VectorFloatFeatureStreamPtr& source, unsigned order, unsigned fftLen, const String& nm = "LPC Spectrum Estimator") {
      return new LPCSpectrumEstimatorPtr(new LPCSpectrumEstimator(source, order, fftLen, nm));
    }
      
    LPCSpectrumEstimatorPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  LPCSpectrumEstimator* operator->();
};


// ----- definition for class `CepstralSpectrumEstimator' -----
//
%ignore CepstralSpectrumEstimator;
class CepstralSpectrumEstimator : public VectorFloatFeatureStream {
 public:
  CepstralSpectrumEstimator(const VectorComplexFeatureStreamPtr& source, unsigned order, unsigned fftLen,
			    double logPadding = 1.0, const String& nm = "LPC Spectrum Estimator");
  virtual ~CepstralSpectrumEstimator();

  virtual const gsl_vector_float* next(int frameX = -5);
  virtual void reset();  
};

class CepstralSpectrumEstimatorPtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    CepstralSpectrumEstimatorPtr(const VectorComplexFeatureStreamPtr& source, unsigned order, unsigned fftLen,
				 double logPadding = 1.0, const String& nm = "LPC Spectrum Estimator") {
      return new CepstralSpectrumEstimatorPtr(new CepstralSpectrumEstimator(source, order, fftLen, logPadding, nm));
    }
      
    CepstralSpectrumEstimatorPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  CepstralSpectrumEstimator* operator->();
};


// ----- definition for class `SEMNB' -----
//
%ignore SEMNB;
class SEMNB {
 public:
  SEMNB( unsigned order, unsigned fftLen, const String& nm= "SEMNB");
  ~SEMNB();

  gsl_vector* calcDerivativeOfDeviation(LPCSpectrumEstimatorPtr &lpcSEPtr );
  void reset();
  const gsl_vector* getLPEnvelope();
};

class SEMNBPtr {
 public:
  %extend {
    SEMNBPtr( unsigned order, unsigned fftLen, const String& nm= "SEMNBPtr")
    {
      return new SEMNBPtr(new SEMNB( order, fftLen,  nm ));
    }
  }

  SEMNB* operator->();
};

// ----- definition for class `FloatFeatureTransmitter' -----
//
%ignore FloatFeatureTransmitter;
class FloatFeatureTransmitter : public VectorFloatFeatureStream {
public:
  FloatFeatureTransmitter( const VectorFloatFeatureStreamPtr& src, int port, int sendflags=0, const String& nm="FloatFeatureTransmitter");
  ~FloatFeatureTransmitter();
  const gsl_vector_float* next(int frameX = -5);
  void reset();
  void sendCurrent();

private:
  const VectorFloatFeatureStreamPtr _src;
  float*                           _buffer;
};

class FloatFeatureTransmitterPtr : public VectorFloatFeatureStreamPtr {
public:
  %extend {
    FloatFeatureTransmitterPtr( const VectorFloatFeatureStreamPtr& src, int port, int sendflags=0, const String& nm="FloatFeatureTransmitter"){
      return new FloatFeatureTransmitterPtr(new FloatFeatureTransmitter(src, port, sendflags, nm));
    }
    
    FloatFeatureTransmitterPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }
  FloatFeatureTransmitter* operator->();
};

// ----- definition for class `ComplexFeatureTransmitter' -----
//
%ignore ComplexFeatureTransmitter;
class ComplexFeatureTransmitter : public VectorComplexFeatureStream {
public:
  ComplexFeatureTransmitter( const VectorComplexFeatureStreamPtr& src, int port, int sendflags=0, const String& nm="ComplexFeatureTransmitter");
  ~ComplexFeatureTransmitter();
  const gsl_vector_complex* next(int frameX = -5);
  void reset();
  void sendCurrent();

private:
  const VectorComplexFeatureStreamPtr _src;
  float*                           _buffer;
};

class ComplexFeatureTransmitterPtr : public VectorFloatFeatureStreamPtr {
public:
  %extend {
    ComplexFeatureTransmitterPtr( const VectorComplexFeatureStreamPtr& src, int port, int sendflags=0, const String& nm="ComplexFeatureTransmitter"){
      return new ComplexFeatureTransmitterPtr(new ComplexFeatureTransmitter(src, port, sendflags, nm));
    }
    
    ComplexFeatureTransmitterPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }
  ComplexFeatureTransmitter* operator->();
};

// ----- definition for class `FloatFeatureReceiver' -----
//
%ignore FloatFeatureReceiver;
class FloatFeatureReceiver : public VectorFloatFeatureStream {
public:
  FloatFeatureReceiver( const String& hostname, int port, int blockLen, int shiftLen=-1, const String& nm= "FloatFeatureReceiver");
  ~FloatFeatureReceiver();
  const gsl_vector_float* next(int frameX = -5);
  void reset();
private:
  int                _sockfd2;
  socklen_t          _client;
  float             *_buffer;
};

class FloatFeatureReceiverPtr : public VectorFloatFeatureStreamPtr {
public:
  %extend {
    FloatFeatureReceiverPtr( const String& hostname, int port, int blockLen, int shiftLen=-1, const String& nm= "FloatFeatureReceiver"){
      return new FloatFeatureReceiverPtr(new FloatFeatureReceiver( hostname, port, blockLen, shiftLen, nm));
    }
    
    FloatFeatureReceiverPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }
  FloatFeatureReceiver* operator->();
};

// ----- definition for class `ComplexFeatureReceiver' -----
//
%ignore ComplexFeatureReceiver;
class ComplexFeatureReceiver : public VectorComplexFeatureStream {
public:
  ComplexFeatureReceiver(  const String& hostname, int port, int blockLen, int shiftLen=-1, const String& nm = "ComplexFeatureReceiver" );
  ~ComplexFeatureReceiver();
  const gsl_vector_float* next(int frameX = -5);
  void reset();
private:
  int                _sockfd2;
  socklen_t          _client;
  float             *_buffer;
};

class ComplexFeatureReceiverPtr : public VectorComplexFeatureStreamPtr {
public:
  %extend {
    ComplexFeatureReceiverPtr( const String& hostname, int port, int blockLen, int shiftLen=-1, const String& nm = "ComplexFeatureReceiver" ){
      return new ComplexFeatureReceiverPtr(new ComplexFeatureReceiver( hostname, port, blockLen, shiftLen, nm));
    }
    
    ComplexFeatureReceiverPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }
  ComplexFeatureReceiver* operator->();
};

// ----- definition for class `WriteSoundFile' -----
//
%ignore WriteSoundFile;
class WriteSoundFile {
public:
  WriteSoundFile(const String& fn, int sampleRate, int nChan=1, int format = sndfile::SF_FORMAT_WAV|sndfile::SF_FORMAT_PCM_32);
  ~WriteSoundFile();
  int write( gsl_vector *vector );
  int writeInt( gsl_vector *vector );
  int writeShort( gsl_vector *vector );
  int writeFloat( gsl_vector *vector );
private:
  sndfile::SNDFILE* _sndfile;
  sndfile::SF_INFO _sfinfo;
};

class WriteSoundFilePtr {
public:
  %extend {
    WriteSoundFilePtr(const String& fn, int sampleRate, int nChan=1, int format = sndfile::SF_FORMAT_WAV|sndfile::SF_FORMAT_PCM_32 ){
      return new WriteSoundFilePtr(new WriteSoundFile( fn, sampleRate, nChan, format));
    }
  }
  WriteSoundFile * operator->();
};

// ----- definition for class `WriteHTKFeatureFile' -----
//
%ignore WriteHTKFeatureFile;
class WriteHTKFeatureFile {
 public:
  WriteHTKFeatureFile(const String& outputfile,
		      int nSamples, 
		      int sampPeriod=160 /* 16msec*/, 
		      short sampSize=sizeof(float), 
		      short parmKind=9 /* USER */,
		      bool isBigEndian=false, /* Linux, Windows & Mac are Little endian OSes */
		      const String& nm = "WriteHTKFeatureFile");
  ~WriteHTKFeatureFile();
  void write( gsl_vector *vector );
};

class WriteHTKFeatureFilePtr {
public:
  %extend {
    WriteHTKFeatureFilePtr(const String& outputfile,
			   int nSamples, 
			   int sampPeriod=160 /* 16msec*/, 
			   short sampSize=sizeof(float), 
			   short parmKind=9 /* USER */,
			   bool isBigEndian=false, /* Linux, Windows & Mac are Little endian OSes */
			   const String& nm = "WriteHTKFeatureFile"){
      return new WriteHTKFeatureFilePtr(new WriteHTKFeatureFile(  outputfile, nSamples, sampPeriod, sampSize, parmKind, isBigEndian, nm));
    }
  }
  WriteHTKFeatureFile * operator->();
};
 
// ----- definition for class `HTKFeature' -----
// 
%ignore HTKFeature;
class HTKFeature : public VectorFeatureStream {
 public:
  HTKFeature(const String& inputfile="", int vecSize = 256, 
	     bool isBigEndian=false, const String& nm = "HTKFeature");
  ~HTKFeature();
  virtual const gsl_vector* next(int frameX = -5);
  virtual void reset();
  bool read( const String& inputfile, bool isBigEndian=false );
  int samplesN();
  int samplePeriod();
  short sampleSize();
  short parmKind();
};

class HTKFeaturePtr : public VectorFeatureStreamPtr {
public:
  %extend {
    HTKFeaturePtr(const String& inputfile="", int vecSize = 256, 
		  bool isBigEndian=false, const String& nm = "HTKFeature"){
      return new HTKFeaturePtr(new HTKFeature( inputfile, vecSize, isBigEndian, nm));
    }
    HTKFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }
  HTKFeature * operator->();
};
