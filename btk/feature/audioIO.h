//                              -*- C++ -*-
//
//                            Speech Front End
//                                  (sfe)
//
//  Module:  sfe.feature
//  Purpose: Speech recognition front end.
//  Author:  ABC
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


#ifndef _audioIO_h_
#define _audioIO_h_

#include <math.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_vector_complex.h>

#include "matrix/gslmatrix.h"
#include "stream/stream.h"
#include "common/mlist.h"
#include "feature/feature.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netdb.h>

#include <pthread.h>

#include "btk.h"

#define INTFEATURE_DATA_TYPE     'i'
#define FLOATFEATURE_DATA_TYPE   'f'
#define COMPLEXFEATURE_DATA_TYPE 'C'

#ifdef HAVE_LIBMARKIV

//#define USE_HDD 
//#define USE_HDDCACHE
#if defined(USE_HDD) || defined(USE_HDDCACHE)
#include <deque>
#endif /* USE_HDD OR USE_HDDCACHE */
#include "feature/myList.h"
#include <sys/time.h>
#include "driver/mk4lib.h"

#endif /* #ifdef HAVE_LIBMARKIV */

/* sflib and sndfile both define sf_perror so we put them into seperate namespaces */
namespace sndfile {
#include <sndfile.h>
}

/*@}*/

/**
* \defgroup CaptureClient Capture client for the NIST Mark III
*/
/*@{*/

// ----- definition for class `CaptureClient' -----
//
class CaptureClient;
typedef Inherit<CaptureClient, VectorFloatFeatureStreamPtr> CaptureClientPtr;
class CaptureClient : public VectorFloatFeatureStream {
 
 public:
  CaptureClient(const char *ip, int port, unsigned blockLen, const String& nm);
  
  virtual ~CaptureClient();
  virtual const gsl_vector_float* next(int frameX = -5);
  virtual void reset() { VectorFloatFeatureStream::reset(); }  
  
  int connectToServer(const char *hostip, int port);
  static void* captureFrames(void* arg);
  
  struct sockaddr_in				their_addr;   /* struct containing address of server */
  struct hostent*				he;
  
  int socketHandler;
  int Fs,duration;

#if 0 /*nur für sndfile operation*/

  int channels, bpchannel;
  sndfile::SNDFILE* sndfp;

  struct t_sndfile {
    char name[500];
    sndfile::SF_INFO info; 
    sndfile::SNDFILE* fd;
  
    int first_loop;
    void* zero_space;
  };

  char* char_ts(char const * format);
  sndfile::SNDFILE* sndfile_open();

#endif
  
 private:
  int						th1;
  char*						clientRcvBufffer;
  long						readPtr;
  long						writePtr;
  long						resetPtr;

  pthread_t					rcv_t;	 
  pthread_mutex_t				mutex;
	 
};

/*@}*/

/*@}*/

int connectToServer( char* host, int port, struct sockaddr_in *pServAddr );
int constructServer( int port, struct sockaddr_in *pServAddr, int backlog=5 );

/*@}*/

/**
* \defgroup 
*/
/*@{*/

// ----- definition for class 'BaseFeatureClient' -----
//

class BaseFeatureClient {
public:
  BaseFeatureClient( const String& hostname, int port, const String& nm );
  ~BaseFeatureClient();
  int getSockfd(){
    return _sockfd;
  }
  unsigned short getPort(){
    return _port;
  }
  
protected:
  unsigned short     _port;
  String             _hostname;
  struct sockaddr_in _servAddr;
  int                _sockfd;
};

typedef refcount_ptr<BaseFeatureClient> BaseFeatureClientPtr;

// ----- definition for class 'BaseFeatureServer' -----
//

class BaseFeatureServer {
public:
  BaseFeatureServer( int port, int sendflags, const String& nm="BaseFeatureServer" );
  ~BaseFeatureServer();
  int getSockfd(){
    return _sockfd1;
  }
  
protected:
  unsigned short     _port;
  struct sockaddr_in _servAddr;
  struct sockaddr_in _clieAddr;
  int                _sockfd1;
  int                _sendflags; /* flags for send(). The default is a block mode. */
};

typedef refcount_ptr<BaseFeatureServer> BaseFeatureServerPtr;

/*@}*/

/**
* \defgroup 
*/
/*@{*/

// ----- definition for class 'FloatFeatureTransmitter' -----
//

/**
   @class the server class for the packet communication
   @brief this class iteratively passes a GSL float vector to a server through TCP/IP. 
   @usage
 */
class FloatFeatureTransmitter : protected BaseFeatureServer, public VectorFloatFeatureStream {
public:
  FloatFeatureTransmitter( const VectorFloatFeatureStreamPtr& src, int port, int sendflags=0, const String& nm="FloatFeatureTransmitter");
  ~FloatFeatureTransmitter();
  const gsl_vector_float* next(int frameX = -5);
  void reset();
  void sendCurrent();

private:
  int                               _sockfd2;
  socklen_t                         _client;
  const VectorFloatFeatureStreamPtr _src;
  float*                            _buffer;
};

typedef Inherit<FloatFeatureTransmitter, VectorFloatFeatureStreamPtr> FloatFeatureTransmitterPtr;

/*@}*/

/**
* \defgroup 
*/
/*@{*/

// ----- definition for class 'ComplexFeatureTransmitter' -----
//

class ComplexFeatureTransmitter : protected BaseFeatureServer, public VectorComplexFeatureStream {
public:
  ComplexFeatureTransmitter( const VectorComplexFeatureStreamPtr& src, int port, int sendflags=0, const String& nm="ComplexFeatureTransmitter");
  ~ComplexFeatureTransmitter();
  const gsl_vector_complex* next(int frameX = -5);
  void reset();
  void sendCurrent();

private:
  int                                 _sockfd2;
  socklen_t                           _client;
  const VectorComplexFeatureStreamPtr _src;
  float*                              _buffer;
};

typedef Inherit<ComplexFeatureTransmitter, VectorComplexFeatureStreamPtr> ComplexFeatureTransmitterPtr;


/*@}*/

/**
* \defgroup 
*/
/*@{*/

// ----- definition for class 'FloatFeatureReceiver' -----
//

/**
   @class the client class for the packet communication
   @brief this class iteratively receives a GSL float vector from a client through TCP/IP. 
   @usage The procedures are as follows.
         1. construct an object,
         2. call next() to receive the vector data and         
   @note 
 */
class FloatFeatureReceiver : protected BaseFeatureClient, public VectorFloatFeatureStream {
public:
  FloatFeatureReceiver( const String& hostname, int port, int blockLen, int shiftLen=-1, const String& nm = "FloatFeatureReceiver");
  ~FloatFeatureReceiver();
  const gsl_vector_float* next(int frameX = -5);
  void reset();
private:
  float             *_buffer;
  int                _shiftLen;
};

typedef Inherit<FloatFeatureReceiver, VectorFloatFeatureStreamPtr> FloatFeatureReceiverPtr;

/*@}*/

/**
* \defgroup 
*/
/*@{*/

// ----- definition for class `ComplexFeatureReceiver' -----
//

class ComplexFeatureReceiver : protected BaseFeatureClient, public VectorComplexFeatureStream {
public:
  ComplexFeatureReceiver( const String& hostname, int port, int blockLen, int shiftLen=-1, const String& nm = "ComplexFeatureReceiver" );
  ~ComplexFeatureReceiver();
  const gsl_vector_complex* next(int frameX = -5);
  void reset();
private:
  float             *_buffer;
  int                _shiftLen;
};

typedef Inherit<ComplexFeatureReceiver, VectorComplexFeatureStreamPtr> ComplexFeatureReceiverPtr;

#ifdef HAVE_LIBMARKIV

/*@}*/

/**
* \defgroup NISTMarkIII NIST Mark III
*/
/*@{*/

// ----- definition for class `NISTMarkIII' -----
//
class NISTMarkIII {
 public:
  NISTMarkIII(VectorShortFeatureStreamPtr& src, unsigned chanN = 64,
	      unsigned blkSz = 320, unsigned shftSz = 160, unsigned triggerX = 0);
  ~NISTMarkIII();

  gsl_vector_short* next(unsigned chanX);

  void reset() { _frameX = -1;  gsl_vector_short_set_zero(_bigBlock); }

  unsigned blockSize()   const { return _blockSize; }
  unsigned shiftSize()   const { return _shiftSize; }
  unsigned overlapSize() const { return _overlapSize; }

 private:
  VectorShortFeatureStreamPtr			_src;
  const unsigned				_chanN;
  const unsigned				_blockSize;
  const unsigned				_shiftSize;
  const unsigned				_overlapSize;
  const unsigned				_triggerX;

  gsl_vector_short*				_bigBlock;
  gsl_vector_short*				_block;

  int						_frameX;
};

typedef refcount_ptr<NISTMarkIII> 		NISTMarkIIIPtr;

/*@}*/


// ----- definition for class `NISTMarkIIIFeature' -----
//
class NISTMarkIIIFeature : public VectorShortFeatureStream {
 public:
  NISTMarkIIIFeature(NISTMarkIIIPtr& markIII, unsigned chanX, unsigned chanN, const String& nm)
    : VectorShortFeatureStream(int(markIII->blockSize())/chanN, nm), _markIII(markIII), _chanX(chanX)
    {
      printf("NISTMarkIII Feature Output Size = %d\n", size());
    }

  virtual ~NISTMarkIIIFeature() { }

  virtual gsl_vector_short* next(int frameX = -5) { return _markIII->next(_chanX); }

  virtual void reset() { _markIII->reset(); }

 private:
  NISTMarkIIIPtr				_markIII;
  const unsigned				_chanX;
};

typedef Inherit<NISTMarkIIIFeature, VectorShortFeatureStreamPtr> NISTMarkIIIFeaturePtr;


// ----- definition for class `NISTMarkIIIFloat' -----
//
class NISTMarkIIIFloat {
 public:
  NISTMarkIIIFloat(VectorFloatFeatureStreamPtr& src, unsigned chanN = 64,
		   unsigned blkSz = 320, unsigned shftSz = 160, unsigned triggerX = 0);
  ~NISTMarkIIIFloat();

  gsl_vector_float* next(unsigned chanX);

  void reset() { _frameX = -1;  gsl_vector_float_set_zero(_bigBlock); }

  unsigned blockSize()   const { return _blockSize; }
  unsigned shiftSize()   const { return _shiftSize; }
  unsigned overlapSize() const { return _overlapSize; }

 private:
  VectorFloatFeatureStreamPtr			_src;
  const unsigned				_chanN;
  const unsigned				_blockSize;
  const unsigned				_shiftSize;
  const unsigned				_overlapSize;
  const unsigned				_triggerX;

  gsl_vector_float*				_bigBlock;
  gsl_vector_float*				_block;

  int						_frameX;
};

typedef refcount_ptr<NISTMarkIIIFloat> 		NISTMarkIIIFloatPtr;


// ----- definition for class `NISTMarkIIIFloatFeature' -----
//
class NISTMarkIIIFloatFeature : public VectorFloatFeatureStream {
 public:
  NISTMarkIIIFloatFeature(NISTMarkIIIFloatPtr& markIII, unsigned chanX, unsigned chanN, const String& nm)
    : VectorFloatFeatureStream(int(markIII->blockSize())/chanN, nm), _markIII(markIII), _chanX(chanX)
    {
      printf("NISTMarkIIIFloat Feature Output Size = %d\n", size());
    }

  virtual ~NISTMarkIIIFloatFeature() { }

  virtual gsl_vector_float* next(int frameX = -5) { 
    gsl_vector_float *block = _markIII->next(_chanX); 
    return block;
  }

  virtual void reset() { _markIII->reset(); }

 private:
  NISTMarkIIIFloatPtr				_markIII;
  const unsigned				_chanX;
};

typedef Inherit<NISTMarkIIIFloatFeature, VectorFloatFeatureStreamPtr> NISTMarkIIIFloatFeaturePtr;

// ----- definition for class `timeOfDay' -----
//
class timeOfDay : public VectorFeatureStream {
public:
  timeOfDay( const String& nm="timeOfDay"): 
    VectorFeatureStream(2, nm)
  {}
  ~timeOfDay()
  {}
  const gsl_vector *next(int frameX = -5)
  {
    if (frameX == _frameX) return _vector;
    struct timeval now_time;
    _increment();
    gettimeofday( &now_time, NULL);
    gsl_vector_set(_vector, 0, now_time.tv_sec );
    gsl_vector_set(_vector, 1, now_time.tv_usec);
    return (const gsl_vector *)_vector;
  }
  void reset()
  {
  }
};

typedef Inherit<timeOfDay, VectorFeatureStreamPtr> 		timeOfDayPtr;

// ----- definition for class `NISTMarkIV' -----
//
#define MAX_MARKIVCHAN 64

/**
   @class A wrapper class for the MarkIV library
   @brief This object launches a thread for capturing data with MarkIV. 
          In current implementation, all the data samples will be stored into the buffer 
          unless they are taken out through popOneBlockData().
   @usage
   1. construct an object, NISTMarkIV mark4( 128 ),
   2. set the prefix for output files with mark4.setOutputFile( "/home/out" ) if you want to save the raw data,
   3. launch a thread by calling mark4.start(), 
   4. take internally a block of incoming 64-channel data through mark4.popOneBlockData(), 
   6. get the data block with mark4.getNthChannelData(int chanX),  
   7. write it with mark4.writeOneBlockData() in the case that you did step 2., and 
   8  repeat step 4.to 7. until you call mark4.stop().
   @note
   After step 4., you can also get the n-th channel data through mark4.getNthChannelData(n-th).
*/
class NISTMarkIV {
 public:
  NISTMarkIV(int blockLen, int mk4speed= 44100);
  ~NISTMarkIV();
  bool start();
  void stop();
  bool setOutputFile( const String& fn );  
  bool popOneBlockData();
  int  *getNthChannelData(int nth);
  void writeOneBlockData();
  int blockSize(){ return _blockLen; }
  int IsRecording(){
    return _recording_state;// During the recording, _recording_state > 0
  }

private:
  int initialize_array();
  void clear_array();

  void *record_data();
  int start_capture();
  int halt_capture();

  //pthread_create(&m_thread, 0, &threaded_class::start_thread, this);

  static void *start_thread(void *obj)
  {
    //All we do here is call the do_work() function
    reinterpret_cast<NISTMarkIV *>(obj)->record_data();
    return NULL;
  }
  
#if defined(USE_HDD) || defined(USE_HDDCACHE)
  bool pushNewRFile();
  bool pushNewWFile();
#endif /* USE_HDD OR USE_HDDCACHE */
#if defined(USE_HDDCACHE)
  bool loadCacheData();
  bool writeCacheData( char **pptr );
#endif /* USE_HDDCACHE */

 private:
  static mk4array *_array;
  static mk4cursor *_cursor;  
  mk4error _aerr; /* error code */

  int _nring_mb;  /* byte size per sample */
  int _frames_per_read;
  int _mk4speed;  /* sampling rate */
  char _dip[32];
  int _slave; /*= mk4_false; 101 */
  int _quiet;
  int _drop;
  int _fix;
  int _verbose;
  FILE *_fp[MAX_MARKIVCHAN];
  FILE *_tslogfp; /* for keeping the log of time stamps */

  volatile bool _isDataPopped;
  volatile int  _recording_state;
  pthread_t     _record_thread;
  unsigned long _headcounter;
  int *_block[MAX_MARKIVCHAN];
  int _blockLen;
#if defined(USE_HDD) || defined(USE_HDDCACHE)
  volatile unsigned long    _cachefilewriteposition;
  volatile unsigned long    _cachefilereadposition;
  deque<FILE *> _bufwfpList;
  deque<FILE *> _bufrfpList;
  deque<string> _buffnList;
  int           *_buf4reading;
  int           *_buf4writing;
#else
  volatile unsigned long    _bufferwriteposition;
  volatile unsigned long    _bufferreadposition;
  volatile unsigned int     _bufsize;
  myList _buf[MAX_MARKIVCHAN];
#endif /* NO USE_HDD AND USE_HDDCACHE */ 
#if defined(USE_HDDCACHE)
  volatile unsigned long _bufferwriteposition;
  volatile unsigned long _bufferreadposition;
  volatile unsigned int  _bufsize;
  myList    _buf[MAX_MARKIVCHAN];
#endif /* USE_HDDCACHE */
  unsigned long _intercounter;
};

typedef refcount_ptr<NISTMarkIV> 		NISTMarkIVPtr;

bool convertMarkIVTmpRaw2Wav( const String& ifn, const String& oprefix, int format=0x010000| 0x0004 /* Microsoft Wave */ );

// ----- definition for class `NISTMarkIVFloatFeature' -----
//
/**
   @brief This class takes data samples from mark4 which you have to feed and return it through next(). 
          This was made for keeping the consitency with other feature stream classess.

 */
class NISTMarkIVFloatFeature : public VectorFloatFeatureStream {
public:
  NISTMarkIVFloatFeature(NISTMarkIVPtr& mark4, unsigned chX, const unsigned firstChanX = 0, const String& nm = "NISTMarkIVFloatFeature")
    : VectorFloatFeatureStream( mark4->blockSize(), nm), _mark4(mark4),  _chanX(chX)
    {
      printf("NISTMarkIVFloat Feature chanX = %d, Output Size = %d\n", chX, size());
      _firstChanX = firstChanX;
      _cur = 0;
      if( _chanX >= MAX_MARKIVCHAN ){
	fprintf(stderr,"Channel ID %d must be less than %d\n",_chanX,MAX_MARKIVCHAN);
	throw jindex_error("Channel ID %d must be less than %d\n",_chanX,MAX_MARKIVCHAN);
      }
    }

  ~NISTMarkIVFloatFeature() { 
    //fprintf(stderr,"~NISTMarkIVFloatFeature() %d\n",_chanX);
    if( _mark4->IsRecording() > 0 && _chanX == _firstChanX )
      _mark4->stop();
  }

  virtual gsl_vector_float* next(int frameX = -5) 
  { 
    if (frameX == _frameX) return _vector;

    if( _mark4->IsRecording() <= 0 && 0 == _cur && _chanX == _firstChanX ){
      //fprintf(stderr,"_mark4->start()\n");
      _mark4->start();
    }

    if( _chanX == _firstChanX )
      _mark4->popOneBlockData(); /* take samples from the buffer in the other thread. */

    int  *samples = _mark4->getNthChannelData( _chanX );
    for(int frX=0;frX<_mark4->blockSize();frX++ ){
      gsl_vector_float_set( _vector, frX, samples[frX]);
    }
    _cur++;
    _increment();
    return _vector;
  }

  virtual void reset(){ 
    _cur = 0;  
    VectorFloatFeatureStream::reset();
  }
  
  gsl_vector_float* data()
  {
    return _vector;
  }

 private:
  NISTMarkIVPtr				_mark4;
  const unsigned                        _chanX;
  unsigned                              _cur;
  unsigned                        _firstChanX;
};

typedef Inherit<NISTMarkIVFloatFeature, VectorFloatFeatureStreamPtr> NISTMarkIVFloatFeaturePtr;

// ----- definition for class `NISTMarkIVServer' -----
//
/**
   @class The server class for transmitting Mark IV data to multiple clients.
   @brief This class sends the multichannel data from NISTMarkIVPtr to the client objects of NISTMarkIVClient. 
         Through next() method of the client object, NISTMarkIVClient, each channel data can be obtained.
   @note The number of clients has to be defined when this object is constructed.
 */
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

private:
  bool checkMarkIVHeader(int sockfd2);
private:
  NISTMarkIVPtr	    	      _mark4;
  unsigned                    _clientN;  
  vector<BaseFeatureServer *> _serverList;
  int                        *_sockfd2List;
  unsigned short             *_chanNList; /* _chanNList[clientN]*/ 
  unsigned short            **_chanXList; /* _chanXList[clientN][_chanNList[clX]] */
  unsigned short              _connectionN;
  unsigned short              _logChanX;
};

typedef Inherit<NISTMarkIVServer, VectorFloatFeatureStreamPtr> NISTMarkIVServerPtr;

// ----- definition for class `NISTMarkIVClienet' -----
//
/**
   @class client class for extracting the single-channel data transmitted from the MarkIV server.
   @brief This class receives the data from the "firstChanX" to "lastChanX" channel and
          the method next() returns one of those channels which is fed via a constructor.
 */
class NISTMarkIVClient : public VectorFloatFeatureStream {
public:
  NISTMarkIVClient( const String& hostname, int port, unsigned chX, gsl_vector *chanIDs, unsigned blockLen = 320, int shiftLen=-1, unsigned short firstChanX = 0, const String& nm = "NISTMarkIVClient");
  ~NISTMarkIVClient();
  virtual gsl_vector_float* next(int frameX = -5);
  virtual void reset();
private:
  bool sendMarkIVHeader();

private:
  static BaseFeatureClient *_client;
  static int               *_buffer;
  static size_t             _bufLen;
  static unsigned short    *_chanIDs;
  static unsigned short     _chanN;

  int                _shiftLen;
  const unsigned     _chanX;
  unsigned short     _firstChanX;
  unsigned short     _NthID;
};
typedef Inherit<NISTMarkIVClient, VectorFloatFeatureStreamPtr> NISTMarkIVClientPtr;

#endif /* #ifdef HAVE_LIBMARKIV */

#endif
