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

#include "audioIO.h"

// ----- Methods for class 'CaptureClient' -----
//
CaptureClient::CaptureClient(const char *ip , int port,  unsigned blockLen, const String& nm) : 
	VectorFloatFeatureStream(blockLen , nm) 
{
/*
  channels = 8;
  bpchannel = 4;
*/  
  socketHandler = 0;
  Fs = 44100;
  duration = 10; /*in seconds*/
  
  clientRcvBufffer = (char *) calloc(Fs*duration, sizeof(char));
  readPtr = 0;
  writePtr = 0;
  resetPtr = 0;
  
//  sndfp = sndfile_open();
  
  pthread_mutex_init(&mutex, NULL);
  
  if ( (socketHandler = connectToServer(ip, port)) < 0) {
    printf("connection failed with %s --- %d\n",ip, port);
    exit(1);
  }
  
  // started a thread to receive data frames from the server in parallel
  if (th1 = pthread_create(&rcv_t, NULL, &captureFrames, (void*)this)) {
  	printf(" Thread creation failed: %d \n",th1);
  	exit(1);
  }
}
  
CaptureClient::~CaptureClient()
{
  pthread_mutex_destroy(&mutex); 
//  sf_close(sndfp);
  free(clientRcvBufffer);
}
  
const gsl_vector_float* CaptureClient::next(int frameX)
{
  unsigned short us = 0;
  short s = 0;
  float f = 0.0;
  //float *aux;
  int r2;
  //aux = (float *) calloc(size(),sizeof(float));

  if (frameX == _frameX) return _vector;
  
  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n",name().c_str(), frameX - 1, _frameX);
  
  _increment();

  pthread_mutex_lock(&mutex);
  r2 = (readPtr + size()*2) % (Fs*duration);
  printf("1: writePtr: %d ---- readPtr : %d .. %d \n", writePtr , readPtr, r2);

  if (r2 > readPtr) {
    while ((writePtr >= readPtr) && (writePtr < r2))  {
      pthread_mutex_unlock(&mutex);
      usleep(10000);
      pthread_mutex_lock(&mutex);
    }
  } else {
    while ((writePtr >= readPtr) || (writePtr < r2))  {
      pthread_mutex_unlock(&mutex);
      usleep(10000);
      pthread_mutex_lock(&mutex);
    }
  }
  for (unsigned i=0; i<size(); i++) {

    us = ((unsigned short)((unsigned char)clientRcvBufffer[readPtr])) << 8 | ((unsigned char)clientRcvBufffer[readPtr+1]);
    s = (short) us;
    f = ((float) s) * 3.0517578125e-05; /*2^15*/
		
    gsl_vector_float_set(_vector, i, f);
    //aux[i] = f;
    readPtr = (readPtr + 2) % (int)(Fs*duration);
    us = 0; s = 0; f = 0.0;
/*
	  if (sf_write_float(sndfp, aux, size()) < size())
		  printf("[Error] sndfile write fail bytes:  %d  -- %s \n", size(), sf_strerror(sndfp));
*/
  }
  pthread_mutex_unlock(&mutex);
  //free(aux);
  return _vector;  
}
  
int CaptureClient::connectToServer(const char* host, int port)
{
   int sock;
   if ((he = gethostbyname(host)) == NULL)   {
      perror(" gethostbyname ");
      exit(1);
   }
   if((sock = socket(AF_INET, SOCK_STREAM, 0)) == -1)
   {
      perror("socket");
      exit(1);
   }
   their_addr.sin_family = AF_INET;
   their_addr.sin_port = htons(port);
   their_addr.sin_addr = *((struct in_addr *)he->h_addr);

   memset(&(their_addr.sin_zero), '\0', 8);
   if (connect(sock, (struct sockaddr *)&their_addr, sizeof(struct sockaddr)) == -1)   {
      perror("connect");
      exit(1);
   }
   // printf(" \nGot connection \n"); 
   return sock;
}
  
void* CaptureClient::captureFrames(void* arg)
{
  CaptureClient *cl = (CaptureClient*) arg;
  int bytesToRead = 1024;
  int bytesRcvd = 0, copied = 0, mut , w2;
  char *buffer;
  buffer = (char *)calloc(bytesToRead, sizeof(char));

  while (true) {

    if ( ( bytesRcvd = recv(cl->socketHandler, buffer, bytesToRead, 0) ) < 0) {
      perror("rcv");
      return (void*)-1;
    }

    if (bytesRcvd == 0)
      break;
    
    if ( (mut = pthread_mutex_lock(&cl->mutex)) != 0)
      printf(" Mutex locking failed:  %d \n", mut);

    if (bytesRcvd == bytesToRead) {
    	
      w2 = (cl->writePtr + bytesRcvd) % (cl->Fs * cl->duration);
 	
      if (w2 > cl->writePtr) {
	if ( (cl->readPtr > cl->writePtr) && (cl->readPtr <= w2) ) {
	  printf("Buffer underrun 1 %d %d %d!\r\n", cl->writePtr, w2, cl->readPtr);
	  cl->readPtr = w2+1;
	}
      } else {
	if ( (cl->readPtr > cl->writePtr) || (cl->readPtr <= w2) ) {
	  printf("Buffer underrun 2 %d %d %d!\r\n", cl->writePtr, w2, cl->readPtr);
	  if (cl->readPtr > cl->writePtr)
	    cl->readPtr = w2+1;
	}
      }
   		
      if (cl->writePtr < ((cl->Fs * cl->duration) - bytesRcvd) ) {
		
	memcpy(&cl->clientRcvBufffer[cl->writePtr], buffer, bytesRcvd);
	cl->writePtr += bytesRcvd;

      } else {
	copied = (cl->Fs*cl->duration) - cl->writePtr;
	memcpy(&cl->clientRcvBufffer[cl->writePtr], buffer, copied);
	cl->writePtr = 0;
	cl->resetPtr = 1;
	memcpy(&cl->clientRcvBufffer[cl->writePtr], &buffer[copied], bytesRcvd - copied);
	cl->writePtr = bytesRcvd - copied;
	copied = 0;
      }

      memset(buffer, 0, sizeof(buffer));
      bytesRcvd = 0;
    }

    if ( (mut = pthread_mutex_unlock(&cl->mutex)) != 0)
      printf(" Mutex unlocking failed:  %d \n", mut);
    
    //usleep(10000);
  }

  return 0;
}

/**
   @brief connect the server through the TCP/IP protocol
   @param char* host[in]
   @param int port[in]
   @param struct sockaddr_in *pServAddr[out]
   @return An entry into the file descriptor table for an reference to a socket
 */
int connectToServer( char* host, int port, struct sockaddr_in *pServAddr )
{
   int sockfd;
   struct hostent *server;
   int ret;

   server = gethostbyname(host);
   if ( NULL == server ) {
     fprintf(stderr,"gethostbyname(): no such host %s\n", host);
     return -1;
   }
   
   sockfd = socket(AF_INET, SOCK_STREAM, 0);
   if( sockfd== -1){     
      perror("ERROR opening socket");
      return -1;
   }

   bzero((char *) pServAddr, sizeof(*pServAddr));
   pServAddr->sin_family = AF_INET;
   pServAddr->sin_port = htons(port);
   bcopy((char *)server->h_addr, 
         (char *)&(pServAddr->sin_addr.s_addr),
         server->h_length);

   ret = connect( sockfd, (struct sockaddr *)pServAddr, sizeof(*pServAddr));
   if( ret < 0 ){
      perror("connect");
      return ret;
   }
   //printf(" \nGot connection \n");
   return sockfd;
}

int constructServer( int port, struct sockaddr_in *pServAddr, int backlog )
{
  int ret;

  int  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    perror("ERROR opening socket");
    return -1;
  }

  bzero((char *) pServAddr, sizeof(*pServAddr));

  pServAddr->sin_family      = AF_INET;
  pServAddr->sin_addr.s_addr = INADDR_ANY;
  pServAddr->sin_port        = htons(port);

  ret = bind(sockfd, (struct sockaddr *) pServAddr,sizeof(*pServAddr));
  if( ret < 0 ){
    perror("ERROR on binding");
    return ret;
  }

  listen(sockfd,backlog);
  return sockfd;
}

bool checkHeaderInfo( int sockfd2, const char dataType, int blockLen )
{
  /* receive the header information */
  int ret;
  
  char recvDataType;
  ret = recv( sockfd2, (char *)&recvDataType, sizeof(char), MSG_WAITALL );
  if( ret < 0 ){
    perror("failed to get the data type\n");
    return false;
  }
  if( dataType != recvDataType ) {
    fprintf(stderr,"received the invalid data type %c which should be %c\n",recvDataType,dataType);
    return false;
  }
  
  int recvBlockLen;
  ret = recv( sockfd2, (char *)&recvBlockLen, sizeof(int), MSG_WAITALL );
  if( ret < 0 ){
    perror("failed to get the data length\n");
    return false;
  }
  if( blockLen != recvBlockLen ) {
    fprintf(stderr,"The length of the recevied data is invalid.\nIt has to be %d but %d\n",blockLen,recvBlockLen);
    return false;
  }  
  return true;
}

/*
  @brief send a server header information
  @param int sockfd[in]
  @param char dataType[in] 'f' Float type, 
                           'c' Complex type (sets of float values)
  @param int blockLen[in] the length of the data block which will be sent
 */
bool sendHeaderInfo( int sockfd, char dataType, int blockLen )
{
  /* send the header information */
  ssize_t ret;

  ret = send( sockfd, (const void *)&dataType, sizeof(char), 0 );
  if( ret < sizeof(char) ){
    return false;
  }

  ret = send( sockfd, (const void *)&blockLen, sizeof(int), 0 );
  if( ret < sizeof(int) ){
    return false;
  }

  return true;
}

// ----- methods for class 'BaseFeatureClient' -----
//

BaseFeatureClient::BaseFeatureClient( const String& hostname, int port, const String& nm):
  _hostname(hostname), _port(port)
{
  for(unsigned i=0;i<100000;i++){
    _sockfd = connectToServer( (char *)hostname.c_str(), _port, &_servAddr );
    if( _sockfd < 0 ) { // == -1 ){
      fprintf(stderr,"trying to connect to the server %s:%d...\n", (char *)hostname.c_str(), port);
    } 
    else
      break;
    usleep(5000);
  }
  if( _sockfd < 0 ){
    fprintf(stderr,"failed to connect to the server %s:%d\n", (char *)hostname.c_str(), port);
    throw  j_error("failed to connect to the server %s:%d\n", (char *)hostname.c_str(), port);
  } 
  fprintf(stderr,"connected %s:%d\n", (char *)hostname.c_str(), port );
}

BaseFeatureClient::~BaseFeatureClient()
{
  close(_sockfd);
}

// ----- methods for class 'BaseFeatureServer' -----
//

BaseFeatureServer::BaseFeatureServer( int port, int sendflags, const String& nm ):
  _port(port),_sendflags(sendflags)
{
  _sockfd1 = constructServer( _port, &_servAddr );
  if( _sockfd1 == -1 ){
    fprintf(stderr,"failed to construct the server\n");
    throw  j_error("failed to construct the server\n");
  }  
  if( _sendflags != 0 ){
    _sendflags = MSG_DONTWAIT;
    fprintf(stderr,"flag %d\n",_sendflags);
  }
}

BaseFeatureServer::~BaseFeatureServer()
{
  close(_sockfd1);
}

// ----- methods for class 'FloatFeatureTransmitter' -----
//

FloatFeatureTransmitter::FloatFeatureTransmitter( const VectorFloatFeatureStreamPtr& src, int port, int sendflags, const String& nm ):
  BaseFeatureServer( port, sendflags, nm ), VectorFloatFeatureStream( src->size(), nm ), _src(src)
{
  _buffer  = new float[_src->size()];
  
  _client  = sizeof(_clieAddr);
  _sockfd2 = accept(_sockfd1,
                    (struct sockaddr *) &_clieAddr,
                     &_client);
   if ( _sockfd2 < 0 ) {
     perror("ERROR on accept");
     throw j_error("could not accept the client\n");
   }

   if( false==checkHeaderInfo( _sockfd2, (const char)FLOATFEATURE_DATA_TYPE, (int)src->size() ) ){
     fprintf(stderr,"FloatFeatureTransmitter: received the invalid header\n");
     throw  j_error("FloatFeatureTransmitter: received the invalid header\n");
   }
}

FloatFeatureTransmitter::~FloatFeatureTransmitter()
{
  close(_sockfd2);
  delete [] _buffer;
}

const gsl_vector_float* FloatFeatureTransmitter::next( int frameX )
{
  if (frameX == _frameX) return  _src->current();// _vector;
  
  const gsl_vector_float* vec = _src->next();
  for(unsigned n=0;n<vec->size;n++)
    _buffer[n] = gsl_vector_float_get( vec, n );

  size_t bufLen = _src->size() * sizeof(float);
  //ssize_t ret = write( _sockfd2, (const void *)_buffer, bufLen );
  ssize_t ret = send( _sockfd2, (const void *)_buffer, bufLen, _sendflags );
  if( ret < bufLen ){
    fprintf(stderr,"FloatFeatureTransmitter: Lost packets\n" );
  }
  _increment();

  return vec; //_vector;
}

void FloatFeatureTransmitter::reset()
{ 
  VectorFloatFeatureStream::reset(); 
}

void FloatFeatureTransmitter::sendCurrent()
{
  const gsl_vector_float* vec = _src->current();
  for(unsigned n=0;n<vec->size;n++)
    _buffer[n] = gsl_vector_float_get( vec, n );
  
  size_t bufLen = _src->size() * sizeof(float);
  //ssize_t ret = write( _sockfd2, (const void *)_buffer, bufLen );
  ssize_t ret = send( _sockfd2, (const void *)_buffer, bufLen, _sendflags );
  if( ret < bufLen ){
    fprintf(stderr,"FloatFeatureTransmitter: Lost packets\n" );
  }
}

// ----- methods for class 'ComplexFeatureTransmitter' -----
//

ComplexFeatureTransmitter::ComplexFeatureTransmitter( const VectorComplexFeatureStreamPtr& src, int port, int sendflags, const String& nm ): 
  BaseFeatureServer( port, sendflags, nm ), VectorComplexFeatureStream( src->size(), nm ), _src(src)
{
  _buffer = new float[_src->size()*2];

  _client  = sizeof(_clieAddr);
  _sockfd2 = accept(_sockfd1,
                    (struct sockaddr *) &_clieAddr,
                     &_client);
   if ( _sockfd2 < 0 ) {
     perror("ERROR on accept");
     throw j_error("could not accept the client\n");
   }

  if( false==checkHeaderInfo( _sockfd2, (const char)COMPLEXFEATURE_DATA_TYPE, (int)src->size() ) ){
    fprintf(stderr,"ComplexFeatureTransmitter: received the invalid header\n");
    throw  j_error("ComplexFeatureTransmitter: received the invalid header\n");
  }
}

ComplexFeatureTransmitter::~ComplexFeatureTransmitter()
{
  close(_sockfd2);
  delete [] _buffer;
}

const gsl_vector_complex* ComplexFeatureTransmitter::next( int frameX )
{
  if (frameX == _frameX) return  _src->current();// _vector;
  
  const gsl_vector_complex* vec = _src->next();
  for(unsigned n=0;n<vec->size;n++){
    gsl_complex val = gsl_vector_complex_get( vec, n );
    _buffer[2*n]   = GSL_REAL(val);
    _buffer[2*n+1] = GSL_IMAG(val);
  }

  size_t bufLen = 2 * _src->size() * sizeof(float);
  //ssize_t ret = write( _sockfd2, (const void *)_buffer, bufLen );
  ssize_t ret = send( _sockfd2, (const void *)_buffer, bufLen, _sendflags );
  if( ret < bufLen ){
    fprintf(stderr,"ComplexFeatureTransmitter: Lost packets\n" );
  }
  _increment();

  return vec; //_vector;
}

void ComplexFeatureTransmitter::reset()
{ 
  VectorComplexFeatureStream::reset(); 
}

void ComplexFeatureTransmitter::sendCurrent()
{
  const gsl_vector_complex* vec = _src->current();
  for(unsigned n=0;n<vec->size;n++){
    gsl_complex val = gsl_vector_complex_get( vec, n );
    _buffer[2*n]   = GSL_REAL(val);
    _buffer[2*n+1] = GSL_IMAG(val);
  }

  size_t bufLen = 2 * _src->size() * sizeof(float);
  //ssize_t ret = write( _sockfd2, (const void *)_buffer, bufLen );
  ssize_t ret = send( _sockfd2, (const void *)_buffer, bufLen, _sendflags );
  if( ret < bufLen ){
    fprintf(stderr,"ComplexFeatureTransmitter: Lost packets\n" );
  }
}

// ----- methods for class 'FloatFeatureReceiver' -----
//

FloatFeatureReceiver::FloatFeatureReceiver( const String& hostname, int port, int blockLen, int shiftLen, const String& nm ):
  BaseFeatureClient( hostname, port, nm ), VectorFloatFeatureStream( blockLen, nm ),
  _shiftLen(shiftLen)
{
   _buffer = new float[_vector->size];
   if( shiftLen < 0 )
     _shiftLen = blockLen;

   /* send the header information */
   if( false==sendHeaderInfo( _sockfd, (char)FLOATFEATURE_DATA_TYPE, blockLen ) ){
     fprintf(stderr,"FloatFeatureReceiver: failed to send the header\n");
     throw  j_error("FloatFeatureReceiver: failed to send the header\n");
   }
}

FloatFeatureReceiver::~FloatFeatureReceiver()
{
  delete [] _buffer;
}

const gsl_vector_float* FloatFeatureReceiver::next( int frameX )
{
  if (frameX == _frameX) return  _vector;// _vector;
  int ret;

  //int ret = read( _sockfd, (char *)_buffer, _vector->size * sizeof(float) );
  if( FrameResetX == _frameX ){ // the first frame
    ret = recv( _sockfd, (char *)_buffer, _vector->size * sizeof(float), MSG_WAITALL );
    if (ret < 0)
      perror("ERROR reading from socket");
    for(size_t n=0;n<_vector->size;n++)
      gsl_vector_float_set( _vector, n, _buffer[n] );
  }
  else{
    int blockLen = (int)_vector->size;

    for(int n=0;n<blockLen-_shiftLen;n++){
      gsl_vector_float_set( _vector, n, gsl_vector_float_get( _vector, n+_shiftLen ) );
    }
    ret = recv( _sockfd, (char *)&_buffer[blockLen-_shiftLen], _shiftLen * sizeof(float), MSG_WAITALL );
    if (ret < 0)
      perror("ERROR reading from socket");    
    for(int n=(blockLen-_shiftLen);n<blockLen;n++){
	gsl_vector_float_set( _vector, n, _buffer[n] );
    }
  }

  _increment();
  return _vector; //_vector;
}

void FloatFeatureReceiver::reset()
{ 
  VectorFloatFeatureStream::reset(); 
}

// ----- methods for class 'ComplexFeatureReceiver' -----
//

ComplexFeatureReceiver::ComplexFeatureReceiver( const String& hostname, int port, int blockLen, int shiftLen, const String& nm):
  BaseFeatureClient( hostname, port, nm ), VectorComplexFeatureStream( blockLen, nm ),
  _shiftLen(shiftLen)
{  
  _buffer = new float[ 2 * _vector->size ];
  if( shiftLen < 0 )
    _shiftLen = blockLen;
  
  /* send the header information */
  if( false==sendHeaderInfo( _sockfd, (char)COMPLEXFEATURE_DATA_TYPE, blockLen ) ){
    fprintf(stderr,"ComplexFeatureReceiver: failed to send the header\n");
    throw  j_error("ComplexFeatureReceiver: failed to send the header\n");
  }
}

ComplexFeatureReceiver::~ComplexFeatureReceiver()
{
  delete [] _buffer;
}

const gsl_vector_complex* ComplexFeatureReceiver::next( int frameX )
{
  if (frameX == _frameX) return  _vector;// _vector;
  int ret;
  
  //int ret = read( _sockfd, (char *)_buffer, 2 * _vector->size * sizeof(float) );
  if( FrameResetX == _frameX ){ // the first frame
    ret = recv( _sockfd, (char *)_buffer, 2 * _vector->size * sizeof(float), MSG_WAITALL );
    if (ret < 0){
      perror("ERROR reading from socket");    
    }
    for(size_t n=0;n<_vector->size;n++){
      gsl_vector_complex_set( _vector, n, gsl_complex_rect(_buffer[2*n],_buffer[2*n+1]) );
    }
  }
  else{
    int blockLen = (int)_vector->size;

    for(int n=0;n<blockLen-_shiftLen;n++){
      gsl_vector_complex_set( _vector, n, gsl_vector_complex_get( _vector, n+_shiftLen ) );
    }
    ret = recv( _sockfd, (char *)&_buffer[2*(blockLen-_shiftLen)], 2 * _shiftLen * sizeof(float), MSG_WAITALL );
    if (ret < 0)
      perror("ERROR reading from socket");    
    for(int n=(blockLen-_shiftLen);n<blockLen;n++){
	gsl_vector_complex_set( _vector, n, gsl_complex_rect(_buffer[2*n],_buffer[2*n+1]) );
    }
  }

  _increment();
  return _vector; //_vector;
}

void ComplexFeatureReceiver::reset()
{ 
  VectorComplexFeatureStream::reset(); 
}

#ifdef HAVE_LIBMARKIV

// ----- methods for class `NISTMarkIII' -----
//
NISTMarkIII::
NISTMarkIII(VectorShortFeatureStreamPtr& src, unsigned chanN, unsigned blkSz, unsigned shftSz, unsigned triggerX)
  : _src(src), _chanN(chanN),
    _blockSize(blkSz * chanN), _shiftSize(shftSz * chanN), _overlapSize(_blockSize - _shiftSize),
    _triggerX(triggerX),
    _bigBlock(gsl_vector_short_calloc(_blockSize)),
    _block(new gsl_vector_short[_chanN])
{
  if (shftSz > blkSz)
    throw jdimension_error("Shift size (%d) must be <= block size (%d).\n", shftSz, blkSz);

  if (src->size() != _shiftSize)
    throw jdimension_error("Data sizes do not match (%d vs. %d).\n", src->size(), _shiftSize);

  if (_overlapSize > _shiftSize)
    throw jdimension_error("Overlap and shift sizes do not match (%d vs. %d).\n", _overlapSize, _shiftSize);

  printf("NISTMarkIII Block Size   = %d\n", _blockSize);
  printf("NISTMarkIII Shift Size   = %d\n", _shiftSize);
  printf("NISTMarkIII Overlap Size = %d\n", _overlapSize);

  reset();
  for (unsigned chanX = 0; chanX < _chanN; chanX++) {
    _block[chanX].size   = blkSz;
    _block[chanX].stride = _chanN;
    _block[chanX].data   = _bigBlock->data + chanX;
    _block[chanX].block  = NULL;
    _block[chanX].owner  = 0;
  }
}

NISTMarkIII::~NISTMarkIII()
{
  gsl_vector_short_free(_bigBlock);  delete[] _block;
}

gsl_vector_short* NISTMarkIII::next(unsigned chanX)
{
  if (chanX == _triggerX) {
    memcpy(_bigBlock->data, _bigBlock->data + _shiftSize, _overlapSize * sizeof(short));
    const gsl_vector_short* newBlock = _src->next();
    assert( newBlock->size == _shiftSize );
    memcpy(_bigBlock->data + _overlapSize, newBlock->data, _shiftSize * sizeof(short));
    _frameX++;

#ifdef DEBUG
    printf("Got buffer %d\n", _frameX);
#endif
  }

#ifdef DEBUG
  printf("Returning data for Channel %d\n", chanX);
#endif

  return _block + chanX;
}


// ----- methods for class `NISTMarkIIIFloat' -----
//
NISTMarkIIIFloat::
NISTMarkIIIFloat(VectorFloatFeatureStreamPtr& src, unsigned chanN, unsigned blkSz, unsigned shftSz, unsigned triggerX)
  : _src(src), _chanN(chanN),
    _blockSize(blkSz * chanN), _shiftSize(shftSz * chanN), _overlapSize(_blockSize - _shiftSize),
    _triggerX(triggerX),
    _bigBlock(gsl_vector_float_calloc(_blockSize)),
    _block(new gsl_vector_float[_chanN])
{
  if (shftSz > blkSz)
    throw jdimension_error("Shift size (%d) must be <= block size (%d).\n", shftSz, blkSz);

  if (src->size() != _shiftSize)
    throw jdimension_error("Data sizes do not match (%d vs. %d).\n", src->size(), _shiftSize);

  if (_overlapSize > _shiftSize)
    throw jdimension_error("Overlap and shift sizes do not match (%d vs. %d).\n", _overlapSize, _shiftSize);

  printf("NISTMarkIIIFloat Block Size   = %d\n", _blockSize);
  printf("NISTMarkIIIFloat Shift Size   = %d\n", _shiftSize);
  printf("NISTMarkIIIFloat Overlap Size = %d\n", _overlapSize);

  reset();
  for (unsigned chanX = 0; chanX < _chanN; chanX++) {
    _block[chanX].size   = blkSz;
    _block[chanX].stride = _chanN;
    _block[chanX].data   = _bigBlock->data + chanX;
    _block[chanX].block  = NULL;
    _block[chanX].owner  = 0;
  }
}

NISTMarkIIIFloat::~NISTMarkIIIFloat()
{
  gsl_vector_float_free(_bigBlock);  delete[] _block;
}

gsl_vector_float* NISTMarkIIIFloat::next(unsigned chanX)
{
  if (chanX == _triggerX) {
    if (_overlapSize > 0) {
      memcpy(_bigBlock->data, _bigBlock->data + _shiftSize, _overlapSize * sizeof(float));
    }
    const gsl_vector_float* newBlock = _src->next();
    assert( newBlock->size == _shiftSize );
    memcpy(_bigBlock->data + _overlapSize, newBlock->data, _shiftSize * sizeof(float));
    _frameX++;

#ifdef DEBUG
    printf("Got buffer %d\n", _frameX);
#endif
  }

#ifdef DEBUG
  printf("Returning data for Channel %d\n", chanX);
#endif

  return _block + chanX;
}


// ----- definition for class `NISTMarkIV' -----
//
mk4array  *NISTMarkIV::_array  = NULL;
mk4cursor *NISTMarkIV::_cursor = NULL;

// Internal function using the 'mk4error' print method                                                                                   
void mk4errquit(mk4error *err, const char *msg)
{
    if (err->error_code != MK4_OK) {
        mk4array_perror(err, msg);
        // equit("Quitting");                                                                                                            
        fprintf(stderr,"Quitting!\n"); fflush(stderr);
        exit(-1);
    }
}

#define FRAME_PER_READ     128 /*5 */
#define MAX_FRAME_PER_READ 640

NISTMarkIV::NISTMarkIV(int blockLen, int mk4speed):
  _blockLen(blockLen),
  _isDataPopped(false),
  _mk4speed(mk4speed), /* mk4array_speed_44K; */
  _intercounter(0),
  _tslogfp(NULL)
{
  // mk4speed = mk4array_speed_22K; // 22050
  // fprintf(stderr,"NISTMarkIV()\n");
  _nring_mb = 16;

  _frames_per_read = FRAME_PER_READ;
  _verbose = true;
  strcpy( _dip, "10.0.0.2" );

  _slave = mk4_false; /* 101 */
  _quiet = 0;
  _drop  = 0;
  _fix   = mk4_false;  /* 101 */
  _recording_state = 0;

  for (unsigned chanX=0;chanX<MAX_MARKIVCHAN;chanX++) {
    _fp[chanX] = NULL;
    _block[chanX] = new int[blockLen];
  }
#if defined(USE_HDD) || defined(USE_HDDCACHE)
  _bufwfpList.clear();
  _bufrfpList.clear();
  _buffnList.clear();
  _buf4writing = new int[MAX_MARKIVCHAN*_frames_per_read];
#else
  _bufsize = 0;
#endif /* USE_HDD OR USE_HDDCACHE */
#if defined(USE_HDD)
  _buf4reading = new int[MAX_MARKIVCHAN*blockLen];
#endif /* USE_HDD */
#if defined(USE_HDDCACHE)
  _buf4reading = new int[MAX_MARKIVCHAN*MAX_FRAME_PER_READ];
  _bufsize = 0;
#endif /* USE_HDDCACHE */
}

NISTMarkIV::~NISTMarkIV()
{
  for (unsigned chanX=0;chanX<MAX_MARKIVCHAN;chanX++) {
    delete [] _block[chanX];
    if( NULL != _fp[chanX] )
      fclose(_fp[chanX]);    
  }

#if defined(USE_HDD) || defined(USE_HDDCACHE)
  deque<FILE *>::iterator it;

  for(it=_bufwfpList.begin();it!=_bufwfpList.end();it++)
    fclose( *it );
  for(it=_bufrfpList.begin();it!=_bufrfpList.end();it++)
    fclose( *it );
  delete [] _buf4reading;
  delete [] _buf4writing;
#endif /* USE_HDD OR USE_HDDCACHE */
  if( NULL != _tslogfp )
    fclose(_tslogfp);
  //fprintf(stderr,"~NISTMarkIV() 1\n");
}

bool NISTMarkIV::start()
{
  //fprintf(stderr,"initialize_array()\n");
  initialize_array();
  //fprintf(stderr,"start_capture()\n");
  if( start_capture() < 0 )  
   return false;
  {
    time_t rawtime;
    struct timeval tv;
    time ( &rawtime );
    gettimeofday( &tv, NULL);
    fprintf(stderr,"MarkIV : Start recording %s", ctime(&rawtime));
    fprintf(stderr,"       : %ld %ld\n", tv.tv_sec, tv.tv_usec);
  }
  return true;
}

void NISTMarkIV::stop()
{
  {
    time_t rawtime;   
    struct timeval tv;
    time ( &rawtime );
    gettimeofday( &tv, NULL);
    fprintf(stderr,"MarkIV : stop recording %s", ctime(&rawtime));
    fprintf(stderr,"       : %ld %ld\n", tv.tv_sec, tv.tv_usec);
  }
  _isDataPopped = false;
  _recording_state = 0;
  halt_capture();
  clear_array();

  if( NULL != _tslogfp ){
    fclose(_tslogfp);
    _tslogfp = NULL;
  }
  //fprintf(stderr,"NISTMarkIV::stop() finished\n");
}

bool NISTMarkIV::setOutputFile( const String& prefix )
{
  char fn[FILENAME_MAX];
  bool ret = true;

  for (unsigned chanX=0;chanX<MAX_MARKIVCHAN;chanX++) {
    sprintf(fn,"%s-%03d.raw",prefix.c_str(),chanX);

    if( NULL!=_fp[chanX] ){
      fprintf(stderr,"closing the file %s\n",fn);
      fclose(_fp[chanX]);
      _fp[chanX] = NULL;
    }

    _fp[chanX] = fopen( fn, "wb" );
    if( NULL==_fp[chanX]){
      fprintf(stderr,"cannot open file %s\n",fn);
      ret = false;
    }
  }

  sprintf(fn,"%s.log",prefix.c_str());
  _tslogfp = fopen( fn, "w" );
  if( NULL == _tslogfp ){
    fprintf(stderr,"cannot open file %s\n",fn );
    ret = false;
  }

  return ret;
}

inline static int cvt_to_int(char *ptr)
{
    return ((int)((signed char)ptr[0]))<<16 |
           ((int)((unsigned char)ptr[1]))<<8  |
           ((int)((unsigned char)ptr[2]));
}


#if defined(USE_HDD) || defined(USE_HDDCACHE)

bool NISTMarkIV::pushNewRFile()
{  
  FILE *fp;

  if( _bufrfpList.size() > 0 )
    fclose( _bufrfpList.front() );
  
  fp = fopen( _buffnList.front().c_str(), "rb");
  if( NULL==fp ){
    fprintf(stderr,"could not open work file %s for reading\n",_buffnList.front().c_str());
    return false;
  }
  
  if( _bufrfpList.size() > 0 )
    _bufrfpList[0] = fp;
  else
    _bufrfpList.push_back(fp);

  //fprintf(stderr,"pushNewRFile() %s %d\n", _buffnList.front().c_str(), _cachefilereadposition );
  if( 0 != fseek( _bufrfpList.front(), _cachefilereadposition * sizeof(int) * MAX_MARKIVCHAN, SEEK_SET ) ){
    fprintf(stderr,"cannot seek %s\n", _buffnList.front().c_str() );
    return false;
  }
  return true;
}

#define MAX_KEPT_DATABYTE  ( FRAME_PER_READ * 5000000 ) /* must be a multiple number of FRAME_PER_READ */
bool NISTMarkIV::pushNewWFile()
{
  time_t     now;
  struct tm  *ts;
  char    fn[FILENAME_MAX];
  FILE    *wfp;

  now = time(NULL);
  ts  = localtime(&now);
  sprintf( fn, "%03d%02d%02d%02d-%d.r", ts->tm_yday, ts->tm_hour, ts->tm_min, ts->tm_sec, _buffnList.size() );
  /* tm_sec  seconds after the minute  0-61 
     tm_min  minutes after the hour    0-59
     tm_hour hours since midnight      0-23
     tm_yday days since January 1      0-365 */
  _buffnList.push_back( fn );
  wfp = fopen( fn, "wb" );
  if( NULL==wfp ){
    fprintf(stderr,"could not open work file %s with the writing mode\n",fn);
    return false;
  }
  _bufwfpList.push_back( wfp );
  fprintf(stderr,"%s is created\n",fn);
  return true;
}
#endif /* USE_HDD OR USE_HDDCACHE */ 


#if defined(USE_HDDCACHE)

/*
  @brief load the array data from the cache file
 */
bool NISTMarkIV::loadCacheData()
{
  if( false==pushNewRFile() )
    return false;

  int frame;
  int ret;
  int counter  = 0;
  int sampleN  =  _cachefilewriteposition - _cachefilereadposition;

  if( sampleN > MAX_FRAME_PER_READ )
    sampleN = MAX_FRAME_PER_READ;
  
  ret = fread( (void *)&_buf4reading[0], sizeof(int), sampleN * MAX_MARKIVCHAN, _bufrfpList.front() );
  if( ret != ( sampleN * MAX_MARKIVCHAN ) ){
    fprintf(stderr,"cannot read %d samples from %s\n",MAX_MARKIVCHAN, _buffnList.front().c_str() );
    return false;
  }
  for (frame=0;frame<sampleN;frame++){
    for (unsigned chanX=0;chanX<MAX_MARKIVCHAN;chanX++){
      _buf[chanX].push_back( _buf4reading[counter++] );
    }
  }
  _bufsize               += frame;
  _bufferwriteposition   += frame;
  _cachefilereadposition += frame;
  if( _cachefilereadposition == _cachefilewriteposition || _cachefilereadposition >= MAX_KEPT_DATABYTE ){
    fprintf(stderr,"loadCacheData()::close %s : %d >= %d\n", _buffnList.front().c_str(), _cachefilereadposition, MAX_KEPT_DATABYTE );
    _cachefilereadposition = 0;
    fclose( _bufrfpList.front() );
    _buffnList.pop_front();
    _bufrfpList.pop_front();
  }

  //fprintf(stderr,"%s : %d %d, %d %d %d %d\n", _buffnList.front().c_str(), _cachefilereadposition, _cachefilewriteposition, _bufrfpList.size(), _bufwfpList.size(), _buffnList.size(), _bufsize );

  return true;
}

bool NISTMarkIV::writeCacheData( char **pptr )
{
  unsigned int counter = 0;
  int frame;
  int offset;

  if( 0 == _buffnList.size() ){
    _cachefilewriteposition = 0;
    if( false==pushNewWFile() ){
      fprintf(stderr,"NISTMarkIV: pushNewWFile() failed\n");
      return NULL;
    }
  }

  for (frame=0,offset=0; frame<_frames_per_read; frame++) {
    for (unsigned chanX=0; chanX<MAX_MARKIVCHAN; chanX++, offset+=3)
      _buf4writing[counter++] = cvt_to_int((*pptr)+offset);
  }
  if( counter != fwrite( (const void *)&_buf4writing[0], sizeof(int), counter, _bufwfpList.back() ) )
    fprintf(stderr,"NISTMarkIV: lost a sample\n");
  _cachefilewriteposition += frame;
  if( _cachefilewriteposition >= MAX_KEPT_DATABYTE ){// create a new work file
    fprintf(stderr,"writeCacheData()::close %s : %d >= %d\n", _buffnList.back().c_str(), _cachefilewriteposition, MAX_KEPT_DATABYTE );
    fclose(_bufwfpList.back());
    _bufwfpList.pop_front();
    _cachefilewriteposition = 0;
  }
  else
    fflush(_bufwfpList.back());
  
  return true;
}
#endif /* USE_HDDCACHE */

/**
   @breif copy data in the buffer of the recording thread into _block[chanX]
          so that we can process the input data in this thread.
*/
bool NISTMarkIV::popOneBlockData()
{
  if (_recording_state==0 ) return false;
  _isDataPopped = true;

#if defined(USE_HDD)
  while( _cachefilewriteposition <= ( _cachefilereadposition + _blockLen ) )
     usleep(1000);

  if( false==pushNewRFile() )
    return false;
  int counter  = 0;
  int sampleN  = _blockLen * MAX_MARKIVCHAN;
  if( sampleN != fread( (void *)&_buf4reading[0], sizeof(int), sampleN, _bufrfpList.front() ) ){
    fprintf(stderr,"cannot read %d samples from %s\n",MAX_MARKIVCHAN, _buffnList.front().c_str() );
    return false;
  }
  for(int blX=0;blX<_blockLen;blX++){
    for (unsigned chanX=0;chanX<MAX_MARKIVCHAN;chanX++)
      _block[chanX][blX] = _buf4reading[counter++];
    _cachefilereadposition++;
    if( _cachefilereadposition == MAX_KEPT_DATABYTE ){
      _cachefilereadposition = 0;
      fclose( _bufrfpList.front() );
      _bufrfpList.pop_front();
      _buffnList.pop_front();
      //fclose( _bufwfpList.front() );
      _bufwfpList.pop_front();
    }
    //fprintf(stderr,"%s : %d %d, %d %d %d\n", _buffnList.front().c_str(), _cachefilereadposition, _cachefilewriteposition, _bufrfpList.size(), _bufwfpList.size(), _buffnList.size() );
  }
#else /* USE_HDD */

  {
    while( _bufsize < 4 * _blockLen )
      usleep(1000);

    //fprintf(stderr,"Main B %d < %d\n", _bufsize, _blockLen );
    for(int blX=0;blX<_blockLen;blX++){    
      for (unsigned chanX=0;chanX<MAX_MARKIVCHAN;chanX++) {
	_block[chanX][blX] = _buf[chanX].front();
	_buf[chanX].pop_front();
      }
      _bufferreadposition++;    
    }
    _bufsize -= _blockLen;
    //fprintf(stderr,"Main A %d < %d\n", _bufsize, _blockLen );
  }
#endif /* NO USE_HDD */ 

  return true;
}

void NISTMarkIV::writeOneBlockData()
{
  if( NULL != _fp ){
    for (unsigned chanX=0;chanX<MAX_MARKIVCHAN;chanX++) {
      fwrite(&_block[chanX][0], _blockLen, sizeof(int), _fp[chanX]);
    }
    if( ( _intercounter % 256 ) == 0 ){
      for (unsigned chanX=0;chanX<MAX_MARKIVCHAN;chanX++)
	fflush(_fp[chanX]);
      fflush(_tslogfp);
    }
    _intercounter++;
  }
}

int  *NISTMarkIV::getNthChannelData(int chanX)
{
  return (int *)&_block[chanX][0];
}

int NISTMarkIV::initialize_array()
{
  char prom[9];
  int id;
    
  // Notes: All information related to the parameters and notes
  //        of these functions are explained in "mk4lib.h"  
  // 0: initialize the error handler
  mk4array_init_mk4error(&_aerr);
  mk4errquit(&_aerr, "during comminit");
  
  // 1: create the 'mk4array' handler 
  _array = mk4array_create(_nring_mb, _frames_per_read, &_aerr);
  mk4errquit(&_aerr, "during initialization");
  if (_verbose) printf("Mk3 init done\n");
  
  // 2: initialize the communication with the Mk3
  mk4array_comminit(_array, _dip, &_aerr);
  mk4errquit(&_aerr, "during comminit");
  if (_verbose) printf("Mk3 comminit done\n");
  
  // 3: ask the mk4 for both its 'id' and its 'prom nb'
  id = mk4array_ask_id(_array, &_aerr);
  mk4errquit(&_aerr, "asking id");
  mk4array_ask_promnb(_array, prom, &_aerr);
  mk4errquit(&_aerr, "asking prom nb");
  
  // 4: initialize the Mk3 array with its 'speed' and 'slave' mode
  mk4array_initparams_wp(_array, _mk4speed, _slave, &_aerr);
  mk4errquit(&_aerr, "during initparams");
  if (_verbose) printf("Mk3 paraminit done\n");
  
  // 5: set up misc paramaters (warning display, drop frames, fix...)
  if (_quiet)
    mk4array_display_warnings(_array, mk4_false, &_aerr);
  else
    mk4array_display_warnings(_array, mk4_true, &_aerr);
  mk4errquit(&_aerr, "setting \'warnings\'");
  
  mk4array_fix_drop_X_first_frames(_array, _drop, &_aerr);
  mk4errquit(&_aerr, "setting \'drop X first frames\'");
  
  mk4array_fix_one_sample_delay(_array, _fix, &_aerr);
  mk4errquit(&_aerr, "setting \'one sample delay fix\'");

  // 6: Start capture
  mk4array_capture_on(_array, &_aerr);
  mk4errquit(&_aerr, "starting capture");
  if (_verbose) printf("Mk3 capture started\n");
  
  _cursor = mk4cursor_create(_array, _frames_per_read, &_aerr);
  mk4errquit(&_aerr, "creating cursor");
  
  // Wait for capture to be started
  // (we can not read from the cursor's pointer before that)
  mk4array_wait_capture_started(_array, &_aerr);
  mk4errquit(&_aerr, "waiting on capture started");

  return 0;
}

void NISTMarkIV::clear_array()
{
  mk4array_capture_off(_array, &_aerr);
  mk4errquit(&_aerr, "stopping capture");
  if (_verbose) printf("Mk3 capture stopped\n");
  
  // 8: Clean memory (including the misc cursors)                                                                                      
  mk4array_delete(_array, &_aerr);
  mk4errquit(&_aerr, "cleaning");
  if (_verbose) printf("Mk3 deleted\n");
  
  return;
}

#define SKIPPED_START_SAMPLEN 6400
void *NISTMarkIV::record_data()
{
#if defined(USE_HDD)
  unsigned int counter;
#endif /* USE_HDD */
  mk4error err;
  unsigned chanX;
  // Capture (Main thread code)                                                                                                        
  while (_recording_state == 1) {
    char *ptr = NULL;
    struct timespec ts;
    int frame, nfnbr;
    int offset;

    // Get the cursor pointer and associated capture timestamp                                                                       
    mk4array_get_cursorptr_with_nfnbr(_array, _cursor, &ptr, &ts, &nfnbr, &err);
    mk4errquit(&err, "getting cursor pointer");
#if 0   // Simply wait a bit for initial rubbish to flush
    if( _headcounter < SKIPPED_START_SAMPLEN ){
      _headcounter++;
      continue;
    }
#endif
    if( false == _isDataPopped )
      continue;
    if( NULL != _tslogfp ){
      struct timespec cur_time;
      clock_gettime(CLOCK_REALTIME, &cur_time);
      fprintf(_tslogfp, "%ld %ld %ld\n", cur_time.tv_sec, cur_time.tv_nsec, _bufferwriteposition );
    }
#if defined(USE_HDD)
    counter = 0;
    for (frame=0,offset=0; frame<_frames_per_read; frame++) {
      for (chanX=0; chanX<MAX_MARKIVCHAN; chanX++, offset+=3)
	_buf4writing[counter++] = cvt_to_int(ptr+offset);
    }
    if( counter != fwrite( (const void *)&_buf4writing[0], sizeof(int), counter, _bufwfpList.back() ) )
      fprintf(stderr,"NISTMarkIV: lost a sample\n");
    fflush(_bufwfpList.back());
    _cachefilewriteposition += frame;
    if( _cachefilewriteposition >= MAX_KEPT_DATABYTE ){// create a new work file
      fclose(_bufwfpList.back());
      pushNewWFile();
      _cachefilewriteposition = 0;
    }
#elif defined(USE_HDDCACHE)
#define MAX_RAMCACHE          65536 /* 32768 */
    //fprintf(stderr,"Before RTS   : %d\n",_bufferwriteposition);
    if( _bufsize < MAX_RAMCACHE ){
      if( _buffnList.size() == 0 ){// no cache file
	//fprintf(stderr,"keep data in RAM 1 %d\n", _bufsize );
	for (frame=0,offset=0; frame<_frames_per_read; frame++) {
	  for (chanX=0; chanX<MAX_MARKIVCHAN; chanX++, offset+=3) {
	    _buf[chanX].push_back( cvt_to_int(ptr+offset) );
	  }
	}
	_bufferwriteposition += _frames_per_read;
	_bufsize             += _frames_per_read;
      }
      else{// there is a cache file
	//fprintf(stderr,"load data from the cache file %d\n", _bufsize);
	if( false==loadCacheData() )
	  return NULL;
	if( _buffnList.size() == 0 ){// load all the data in the cache file
	  //fprintf(stderr,"keep data in RAM 2 %d\n", _bufsize );
	  for (frame=0,offset=0; frame<_frames_per_read; frame++) {
	    for (chanX=0; chanX<MAX_MARKIVCHAN; chanX++, offset+=3) {
	      _buf[chanX].push_back( cvt_to_int(ptr+offset) );
	    }
	  }
	  _bufferwriteposition += _frames_per_read;
	  _bufsize             += _frames_per_read;
	}
	else{// dump the data into the cache file
	  //fprintf(stderr,"dump data into the cache file 1 %d\n", _bufsize );
	  if( false==writeCacheData(&ptr) )
	    return NULL;
	}
      }
    }
    else{// dump the data into the cache file
      fprintf(stderr,"Use disk cache %d\n", _bufsize );
      if( false==writeCacheData(&ptr) )
	return NULL;
    }
    //fprintf(stderr,"After RTS : %d\n",_bufferwriteposition);
#else 
    for (frame=0,offset=0; frame<_frames_per_read; frame++) {
      for (chanX=0; chanX<MAX_MARKIVCHAN; chanX++, offset+=3) {
	//buf[bufferwriteposition][i] = cvt_to_int(ptr+offset);
	_buf[chanX].push_back( cvt_to_int(ptr+offset) );
      }
      _bufferwriteposition++;
      _bufsize++;
    }
#endif /* NO USE_HDD */ 
  }
  
  return NULL;
}

int NISTMarkIV::start_capture()
{
#if defined(USE_HDD) || defined(USE_HDDCACHE)
  _cachefilewriteposition = _cachefilereadposition = 0;
#endif /* USE_HDD OR USE_HDDCACHE */
#ifdef  USE_HDD
  if( false==pushNewWFile() )
    return -1;
#else 
  // Initialize all counters
  for (unsigned chanX=0;chanX<MAX_MARKIVCHAN;chanX++){
    _buf[chanX].clear();
  }
  _bufferwriteposition = _bufferreadposition = 0;
#endif /* NO USE_HDD */ 
  _headcounter = 0;
  _recording_state = 1; 
  // Create a recording thread.  Passing NULL since there are no args   
  //pthread_create(&_record_thread, NULL, record_data, (void*)NULL);
  pthread_create(&_record_thread, NULL, &NISTMarkIV::start_thread, this);
  pthread_detach(_record_thread);

  return 0;
}

int NISTMarkIV::halt_capture()
{
  void *ret;
  
  pthread_join(_record_thread,&ret);
  
  return 0;
}

bool convertMarkIVTmpRaw2Wav( const String& ifn, const String& oprefix, int format )
{
  using namespace sndfile;
  SNDFILE** sndfileL;
  SF_INFO* sfinfoL;
  int samplerate = 44100;
  FILE *fp;
  int frX;
  char filename[FILENAME_MAX];

  sndfileL = (SNDFILE**)malloc(MAX_MARKIVCHAN*sizeof(SNDFILE*));
  sfinfoL  = (SF_INFO* )malloc(MAX_MARKIVCHAN*sizeof(SF_INFO));
  if( sndfileL==NULL || sfinfoL==NULL ){
    fprintf(stderr,"Could not allocate memory\n");
    return false;
  }

  for (unsigned chanX=0;chanX<MAX_MARKIVCHAN;chanX++){
    sfinfoL[chanX].format     = format;
    sfinfoL[chanX].samplerate = samplerate;
    sfinfoL[chanX].channels   = 1;
    sprintf(filename,"%s-%02d.wav",oprefix.c_str(),chanX);
    sndfileL[chanX] = sf_open( filename, SFM_WRITE, &sfinfoL[chanX]);
    if (!sndfileL[chanX]){
      fprintf(stderr,"Could not open file %s.\n", filename);
      return false;
    }

    if (sf_error(sndfileL[chanX])) {
      sf_close(sndfileL[chanX]);
      fprintf(stderr,"sndfile error: %s.", sf_strerror(sndfileL[chanX]));
      return false;
    }
  }
  
  fp = fopen( ifn.c_str(), "rb");
  if( NULL==fp ){
    fprintf(stderr,"could not open work file %s for reading\n",ifn.c_str());
    return false;
  }

  for(frX=0;;frX++){
    int buf[MAX_MARKIVCHAN];
    if( MAX_MARKIVCHAN != fread( (void *)&buf[0], sizeof(int), MAX_MARKIVCHAN, fp ) ){
      fprintf(stderr,"cannot read %d samples from %s\n",MAX_MARKIVCHAN, ifn.c_str() );
      return false;
    }
    for (unsigned chanX=0;chanX<MAX_MARKIVCHAN;chanX++){
      sf_write_int(sndfileL[chanX], &buf[chanX], 1 );
    }
  }

  fclose(fp);
  for (unsigned chanX=0;chanX<MAX_MARKIVCHAN;chanX++)
    sf_close(sndfileL[chanX]);
  free(sndfileL);
  free(sfinfoL);

  return true;
}

// ----- definition for class `NISTMarkIVServer' -----
//
/**
   @brief 
   @param NISTMarkIVPtr& mark4[in] MarkIV object
   @param int basePort[in] the least port number. 
   @param unsigned clientN[in] The number of clients which this server class will take.
   @note The ports from 'basePort' to 'basePort + clientN -1' will be used.
 */
NISTMarkIVServer::NISTMarkIVServer( NISTMarkIVPtr& mark4, int basePort, unsigned clientN, const String& nm ):
  VectorFloatFeatureStream( mark4->blockSize(), nm), _mark4(mark4),  _clientN(clientN), _connectionN(0), _logChanX(0)
{
  printf("NISTMarkIVServer : constructing the server for %d clients\n",_clientN);
  if( _clientN >= MAX_MARKIVCHAN ){
    fprintf(stderr,    "The number of channels or clients must be less than %d\n",MAX_MARKIVCHAN);
    throw jindex_error("The number of channels or clients must be less than %d\n",MAX_MARKIVCHAN);
  }

  _sockfd2List = new int[_clientN];
  _chanNList   = new unsigned short[_clientN];
  _chanXList   = (unsigned short **)malloc(_clientN*sizeof(unsigned short *));
  if( NULL == _chanXList ){
    fprintf(stderr,         "NISTMarkIVServer : cannot allocate memory\n");
    throw jallocation_error("NISTMarkIVServer : cannot allocate memory\n");
  }

  _serverList.resize(_clientN);
  for(unsigned clX=0;clX<_clientN;clX++){
    int port = basePort + (int)clX;
    _serverList[clX] = new BaseFeatureServer(port,0);
    printf("NISTMarkIVServer: opened port %d\n",port);
    _sockfd2List[clX]= -1;
    _chanXList[clX]  = NULL;
  }
}


NISTMarkIVServer::~NISTMarkIVServer()
{
  //fprintf(stderr,"~NISTMarkIVServer()\n");
  for(unsigned clX=0;clX<_clientN;clX++){
    delete _serverList[clX];
    _serverList[clX] = NULL;
    if( _sockfd2List[clX] >=0 )
      close(_sockfd2List[clX]);
    if( NULL != _chanXList[clX] )
      free( _chanXList[clX] );
  }
  //delete [] _bChanXList; delete [] _eChanXList;
  delete [] _sockfd2List;
  delete [] _chanNList;
  free( _chanXList );

  if( _mark4->IsRecording() > 0 )
    _mark4->stop();
  /* fprintf(stderr,"~NISTMarkIVServer()\n"); */
}

bool NISTMarkIVServer::setOutputFile( const String& prefix )
{
  return _mark4->setOutputFile(  prefix );
}

void NISTMarkIVServer::writeAllBlockData()
{
  return  _mark4->writeOneBlockData();
}

gsl_vector_float* NISTMarkIVServer::next(int frameX)
{
   if (frameX == _frameX) return _vector;

   size_t bufLen = _mark4->blockSize() * sizeof(int);
   ssize_t ret;

   if( _mark4->IsRecording() <= 0  && _frameX == FrameResetX ){
     if( false==waitForConnections() ){
       fprintf(stderr,    "failed to get clients\n");
       throw jindex_error("failed to get clients\n");
     }
     _mark4->start();
   }
   _mark4->popOneBlockData(); /* take samples from the buffer in the other thread. */

   for(unsigned clX=0;clX<_clientN;clX++){
     /* fprintf(stderr,"%d:",clX); */
     for(unsigned short n=0;n<_chanNList[clX];n++){
       unsigned short chanX = _chanXList[clX][n];
       /* fprintf(stderr," %d (%d)",chanX, _sockfd2List[clX]); */
       int *samples = _mark4->getNthChannelData( chanX );
       /*for(int t=0;t<bufLen/sizeof(int);t++)
	 fprintf(stderr,"%d ",samples[t]);
	 fprintf(stderr,"\n");*/
       ret    = send( _sockfd2List[clX], (const void *)samples, bufLen, 0 );
       if( ret < bufLen ){
	 fprintf(stderr,"NISTMarkIVServer: Lost packets\n" );
       }
       if( _logChanX == chanX){
	 for(int frX=0;frX<_mark4->blockSize();frX++ )
	   gsl_vector_float_set( _vector, frX, samples[frX]);
       }
     }
     /* fprintf(stderr,"\n"); */
   }

   _increment();
   return _vector;
}

void NISTMarkIVServer::reset()
{
  VectorFloatFeatureStream::reset();
}

bool NISTMarkIVServer::checkMarkIVHeader( int sockfd2 )
{
  unsigned short chanN;
  int ret;

  if( false==checkHeaderInfo( sockfd2, (const char)INTFEATURE_DATA_TYPE, (int)_mark4->blockSize() ) ){
    fprintf(stderr,"NISTMarkIVServer: received the invalid header\n");
    return false;
  }

  ret = recv( sockfd2, (char *)&chanN, sizeof(unsigned short), MSG_WAITALL );
  if( ret < 0 ){
    perror("NISTMarkIVServer: failed to get the beginning of the channel index\n");
    return false;
  }
  _chanXList[_connectionN] = (unsigned short *)malloc(chanN*sizeof(unsigned short));
  if( NULL == _chanXList[_connectionN] ){
    fprintf(stderr, "NISTMarkIVServer : cannot allocate memory\n");
    return false;
  }
  
  printf("NISTMarkIVServer: got the %d-th connection for channel ",_connectionN );
  for(unsigned short chanX=0;chanX<chanN;chanX++){
    unsigned short chanID;
    ret = recv( sockfd2, (char *)&chanID, sizeof(unsigned short), MSG_WAITALL );
    if( ret < 0 ){
      perror("NISTMarkIVServer: failed to get the end of the channel index\n");
      return false;
    }
    _chanXList[_connectionN][chanX] = chanID;
    printf("%d ", chanID );
  }
  printf("\n");

  _sockfd2List[_connectionN] = sockfd2;
  _chanNList[_connectionN]   = chanN;
  _connectionN++;

  return true;
}

bool NISTMarkIVServer::waitForConnections()
{ 
  int sockfd1, sockfd2;
  struct sockaddr_in cli_addr;
  socklen_t clientSize = sizeof(cli_addr);

  while (1) {
    //fprintf(stderr,"wait for connections %d, %d\n", _connectionN,_clientN );
    if( _connectionN >= _clientN )
      break;
    for(unsigned clX=0;clX<_clientN;clX++){
      if( _sockfd2List[clX] >= 0 )
	continue;
      
      sockfd1 = _serverList[clX]->getSockfd();
      sockfd2 = accept( sockfd1,
			(struct sockaddr *) &cli_addr,
			&clientSize);
      if ( sockfd2 < 0 ) {
	perror("NISTMarkIVServer: ERROR on accept");
	continue;
      }
      if( false==checkMarkIVHeader(sockfd2) )
	return false;
      usleep(1000);    
#if 0 /* use fork (need to fix a bug) */
      int pid = fork(); /* create a child process */
      if (pid < 0){
	perror("NISTMarkIVServer: ERROR on fork");
	return false;
      }
      if (pid == 0)  {
	if( false==checkMarkIVHeader(sockfd2) )
	  return false;
	//_exit(0);
      }
      else{
	close(sockfd2);
	usleep(1000);     
      }
#endif
    }
  }/* end of while */
     
   return true;
}


// ----- definition for class `NISTMarkIVClienet' -----
//
int*               NISTMarkIVClient::_buffer  = NULL;
size_t             NISTMarkIVClient::_bufLen  = 0;
BaseFeatureClient *NISTMarkIVClient::_client  = NULL;
unsigned short    *NISTMarkIVClient::_chanIDs = NULL;
unsigned short     NISTMarkIVClient::_chanN   = 0;

NISTMarkIVClient::NISTMarkIVClient( const String& hostname, int port, unsigned chX, gsl_vector *chanIDs, unsigned blockLen, int shiftLen, unsigned short firstChanX, const String& nm ):
  VectorFloatFeatureStream( blockLen, nm ),
  _chanX(chX),_firstChanX(firstChanX)
{
  if( _chanX < _firstChanX ){
    fprintf(stderr,"The channel index %d must be more than %d\n",_chanX,_firstChanX);
    throw jindex_error("The channel index %d must be more than %d\n",_chanX,_firstChanX);
  }

  if( _chanX == _firstChanX ){
    _chanN = chanIDs->size;
    _client  = new BaseFeatureClient( hostname, port, nm );
    _bufLen  = _chanN * _vector->size * sizeof(int);
    _buffer  = new int[_bufLen];
    _chanIDs = new unsigned short[_chanN];
    for(unsigned short chanX=0;chanX<_chanN;chanX++)
      _chanIDs[chanX] = (unsigned short)gsl_vector_get( chanIDs, chanX );
  }

  for(_NthID=0;_NthID<_chanN;_NthID++)
    if( _chanX == _chanIDs[_NthID] )
      break;

  if( shiftLen <= 0 || _shiftLen == blockLen ){
    _shiftLen = blockLen;
  }
  else{
    fprintf(stderr,"NISTMarkIVClient: Not yet supported\n");
    _shiftLen = blockLen;
  }

  if( false==sendMarkIVHeader() ){
    fprintf(stderr,"NISTMarkIVClient: failed to send the header\n");
    throw  j_error("NISTMarkIVClient: failed to send the header\n");
  }

  // confirm that the address for static variable is the same in a process.
  // fprintf(stderr,"%d (%d-%d): %ld %ld %ld %ld\n", _chanX, _firstChanX, _lastChanX, (long)&_buffer, (long)&_bufLen, (long)_client, (long)&_lastChanX) ;
}

NISTMarkIVClient::~NISTMarkIVClient()
{
  if( _client != NULL ){
    delete _client;
    _client = NULL;
  }
  if( _buffer != NULL ){
    delete [] _buffer;
    _buffer = NULL;
  }
  if( _chanIDs != NULL ){
    delete [] _chanIDs;
    _chanIDs = NULL;
  }
}

bool NISTMarkIVClient::sendMarkIVHeader()
{
  ssize_t ret;
  int sockfd = _client->getSockfd();
  
  /* send the header information */
  if( false==sendHeaderInfo( sockfd, (char)INTFEATURE_DATA_TYPE, size() ) ){
    return false;
  }

  ret = send( sockfd, (char *)&_chanN, sizeof(unsigned short), 0 );
  if( ret < sizeof(unsigned short) ){
    return false;
  }

  for(unsigned short chanX=0;chanX<_chanN;chanX++){
    unsigned short chanID = _chanIDs[chanX];
    ret = send( sockfd, (char *)&chanID, sizeof(unsigned short), 0 );
    if( ret < sizeof(unsigned short) ){
      return false;
    }
  }

  return true;
}

gsl_vector_float* NISTMarkIVClient::next(int frameX )
{
  if (frameX == _frameX) return _vector;

  if( _chanX == _firstChanX ){
    int sockfd = _client->getSockfd();

    int ret = recv( sockfd, (char *)_buffer, _bufLen, MSG_WAITALL );
    if (ret < 0){
      perror("ERROR reading from socket");
    }
  }
  
  int idx = _NthID * size();
  for(int frX=0;frX<size();frX++){
    gsl_vector_float_set( _vector, frX, _buffer[idx] );
    idx++;
  }
  
  _increment();
  return _vector;
}

void NISTMarkIVClient::reset()
{
   VectorFloatFeatureStream::reset();
}

#endif /* #ifdef HAVE_LIBMARKIV */
