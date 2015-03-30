//
//                            Speech Front End
//                                  (sfe)
//
//  Module:  sfe.feature
//  Purpose: Speech recognition front end.
//  Author:  John McDonough, Tobias Gehrig and ABC
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

#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "common/mach_ind_io.h"
#include "feature/feature.h"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_sf.h>

#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_complex.h>
#include "matrix/gslmatrix.h"

#include "common/jpython_error.h"

#include <sndfile.h>
#include <getline.h>


// unpack the data from half- to standard complex form
//
void halfComplexUnpack(gsl_vector_complex* tgt, const double* src)
{
  int len  = tgt->size;
  int len2 = (len + 1) / 2;

  gsl_vector_complex_set(tgt, 0,    gsl_complex_rect(src[0],    0));
  if ((len & 1) == 0)
    gsl_vector_complex_set(tgt, len2, gsl_complex_rect(src[len2], 0));
  for (int m = 1; m < len2; m++) {
    gsl_vector_complex_set(tgt, m,
			   gsl_complex_rect(src[m],  src[len-m]));
    gsl_vector_complex_set(tgt, len-m,
			   gsl_complex_rect(src[m], -src[len-m]));
  }
}


// pack the data from standard to half-complex form
//
void halfComplexPack(double* tgt, const gsl_vector_complex* src, unsigned size)
{
  unsigned len  = (size == 0)?(src->size):size;
  unsigned len2 = (len+1) / 2;

  gsl_complex entry = gsl_vector_complex_get(src, 0);
  tgt[0]    = GSL_REAL(entry);

  for (unsigned m = 1; m < len2; m++) {
    entry      = gsl_vector_complex_get(src, m);
    tgt[m]     = GSL_REAL(entry);
    tgt[len-m] = GSL_IMAG(entry);
  }

  if ((len & 1) == 0) {
    entry     = gsl_vector_complex_get(src, len2);
    tgt[len2] = GSL_REAL(entry);
  }
}

#ifdef HAVE_LIBFFTW3

// unpack the data from fftw format to standard complex form
//
void fftwUnpack(gsl_vector_complex* tgt, const fftw_complex* src)
{
  int len  = tgt->size;
  int len2 = len / 2;

  for (unsigned i = 0; i <= len2; i++) {
    gsl_vector_complex_set(tgt, i, gsl_complex_rect(src[i][0], src[i][1]));
    if (i != 0 && i != len2)
      gsl_vector_complex_set(tgt, len - i , gsl_complex_rect(src[i][0], -src[i][1]));
  }
}

#endif


// ----- methods for class `FileFeature' -----
//
const gsl_vector_float* FileFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frameX - 1, _frameX);

  _increment();
  if (_frameX >= _feature->size1) {
    // printf("FileFeature: Throwing end of samples ...\n");  fflush(stdout);
    throw jiterator_error("end of samples!");
  }

  float* feat = gsl_matrix_float_ptr(_feature, _frameX, 0);
  for (int i = 0; i < _feature->size2; i++)
    gsl_vector_float_set(_vector, i, feat[i]);

  return _vector;
}

void FileFeature::bload(const String& fileName, bool old)
{
  cout << "Loading features from " << fileName << "." << endl;

  if (_feature != NULL) {
    gsl_matrix_float_free(_feature);
    _feature = NULL;
  }

  _feature = gsl_matrix_float_load(_feature, fileName.chars(), old);

  reset();

  printf("Matrix is %d x %d\n", _feature->size1, _feature->size2);
}

void FileFeature::copy(gsl_matrix_float* matrix)
{
  if ((_feature == NULL) || (_feature->size1 != matrix->size1) || (_feature->size2 != matrix->size2)) {
    gsl_matrix_float_free(_feature);
    _feature = gsl_matrix_float_calloc(matrix->size1, matrix->size2);
  }

  gsl_matrix_float_memcpy(_feature, matrix);
}

FileFeature& FileFeature::operator=(const FileFeature& f)
{
  //  fmatrixCopy(_feature, f._feature);
  gsl_matrix_float_memcpy(_feature, f._feature);

  return *this;
}


// ----- methods for Conversion24bit2Short -----
//
const gsl_vector_short* Conversion24bit2Short::next(int frameX) {
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  _increment();

  const gsl_vector_char* newBlock = _src->next(_frameX);
  unsigned short ii;
  int offset = 0;
  for (int i=0; i<_size; i++) {

    // this is how array3conversion_dsp.c from NSFS does it:
    // *buffer++ = (inbuf[(i*64+j)*3+1])|(inbuf[(i*64+j)*3]<<8);
    
    ii = ((unsigned char)gsl_vector_char_get(newBlock, i*3+1) | 
	  ((unsigned char)gsl_vector_char_get(newBlock, i*3)) << 8);
    gsl_vector_short_set(_vector, i, ii);
  }
  return _vector;
}


// ----- methods for Conversion24bit2Float -----
//
const gsl_vector_float* Conversion24bit2Float::next(int frameX) {
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frameX - 1, _frameX);

  _increment();

  const gsl_vector_char* newBlock = _src->next(_frameX);
  signed int ii;
  unsigned char *p = (unsigned char*) &ii;
  unsigned char sample;
  for (int i=0; i<_size; i++) {
    sample = (unsigned char)gsl_vector_char_get(newBlock, i*3);
    /* Code to put on a 32 bits range */
    /* From Big Endian (CMA's data) to Linux Little Endian numbers */
    if (sample & 128) 
      *(p+3) = 0xFF;
    else 
      *(p+3) = 0x00;
    *(p+2) = sample;
    *(p+1) = (unsigned char)gsl_vector_char_get(newBlock, i*3+1);
    *(p)   = (unsigned char)gsl_vector_char_get(newBlock, i*3+2);
    gsl_vector_float_set( _vector, i, ii*1.0 );
  }
  return _vector;
}


// ----- methods for class `SampleFeature' -----
//
SampleFeature::SampleFeature(const String& fn, unsigned blockLen,
			    unsigned shiftLen, bool padZeros, const String& nm) :
  VectorFloatFeatureStream(blockLen, nm),
  _samples(NULL), _ttlSamples(0), _shiftLen(shiftLen), _cur(0), _padZeros(padZeros), _endOfSamples(false),
  _cpSamplesF(NULL),_cpSamplesD(NULL), _T(NULL), _r(NULL)
{
  if (fn != "") read(fn);

  cout << "Sample Feature Block Length " << blockLen << endl;
  cout << "Sample Feature Shift Length " << shiftLen << endl;
}

SampleFeature::~SampleFeature()
{ 
    if (_samples != NULL) delete[] _samples; 
    if (_cpSamplesF != NULL) gsl_vector_float_free(_cpSamplesF); 
    if (_cpSamplesD != NULL) gsl_vector_free(_cpSamplesD);
    if (_r != NULL) gsl_rng_free (_r);
}

unsigned SampleFeature::
read(const String& fn, int format, int samplerate, int chX, int chN, int cfrom, int to, int outsamplerate, float norm)
{
  using namespace sndfile;
  SNDFILE* sndfile;
  SF_INFO sfinfo;
  float* tmpsamples;
  int nsamples;

  _norm = norm;
  _ttlSamples = 0;

  if ( NULL != _samples ){ delete[] _samples;}
  _samples = NULL;	// avoid double deletion if file cannot be read

  sfinfo.format = format;
  sfinfo.samplerate = samplerate;
  sfinfo.channels = chN;
  sndfile = sf_open(fn.c_str(), SFM_READ, &sfinfo);
  if (!sndfile)
    throw jio_error("Could not open file %s.", fn.c_str());

  if (sf_error(sndfile)) {
    sf_close(sndfile);
    throw jio_error("sndfile error: %s.", sf_strerror(sndfile));
  }

  if (norm == 0.0) {
#ifdef DEBUG
    cout << "Disabling SNDFILE normalization option" << endl;
#endif
    sf_command (sndfile, SFC_SET_NORM_FLOAT, NULL, SF_FALSE);
  } else {
#ifdef DEBUG
    cout << "Enabling SNDFILE normalization option" << endl;
#endif
    sf_command (sndfile, SFC_SET_NORM_FLOAT, NULL, SF_TRUE);
  }

  if (outsamplerate == -1) outsamplerate = sfinfo.samplerate;

#ifdef DEBUG
  cout << "channels: " << sfinfo.channels << endl;
  cout << "frames: " << sfinfo.frames << endl;
  cout << "samplerate: " << sfinfo.samplerate << endl;
#endif

  if ((to < 0) || (to >= sfinfo.frames))
    to = sfinfo.frames - 1;
  if (cfrom < 0)
    cfrom = 0;
  if ((cfrom > to) || (cfrom > sfinfo.frames)) {
    sf_close(sndfile);
    throw jio_error("Cannot load samples from %d to %d.", cfrom, to);
  }
  nsamples = to - cfrom + 1;

  // Allocating memory for samples
  tmpsamples = new float[nsamples*sfinfo.channels];
  if (tmpsamples == NULL) {
    sf_close(sndfile);
    throw jallocation_error("Error when allocating memory for samples.");
  }

  if (sf_seek(sndfile, cfrom, SEEK_SET) == -1)
    throw jio_error("Error seeking to %d", cfrom);

  _ttlSamples = sf_readf_float(sndfile, tmpsamples, nsamples);
#ifdef DEBUG
  cout << "read frames: " << _ttlSamples << endl;
#endif

  _sampleRate = sfinfo.samplerate;
  chN = sfinfo.channels;
  _nChan = chN;

  if (chX > sfinfo.channels || chX < 1) {
    if (chX == 0)
      throw jconsistency_error("Multi-channel read is not yet supported.");
      
    // for now just allow one channel to be loaded
    throw jconsistency_error("Selected channel out of range of available channels.");
    //    chX = 1; 
  }
  chX--;
  if (chX < 0 || chN == 1 )
    _samples = tmpsamples;
  else {
    _samples = new float[nsamples];
    if (_samples == NULL) {
      delete[] tmpsamples;
      sf_close(sndfile);
      throw jallocation_error("Error when allocating memory for samples");
    }
    // Copy the selected channel to _samples
    for (int i=0; i < nsamples; i++)
      _samples[i] = tmpsamples[i*sfinfo.channels + chX];
    delete[] tmpsamples;
  }

  if (_sampleRate <= outsamplerate)
    if (norm != 1.0 && norm != 0.0)
      for (int i=0; i < nsamples; i++)
	_samples[i] *= norm;

#ifdef SRCONV
  if (_sampleRate != outsamplerate) {
#ifdef DEBUG
    cout << "sample rate converting to " << outsamplerate<<endl;
#endif
    SRC_DATA data;
    data.input_frames = _ttlSamples;
    data.src_ratio = (float)outsamplerate / (float)_sampleRate;
    data.output_frames = (long)ceil(data.src_ratio*(float)_ttlSamples);
#ifdef DEBUG
    cout << "src_ratio: " << data.src_ratio << endl;
    cout << "output_frames: " << data.output_frames << endl;
    char normalisation = sf_command (sndfile, SFC_GET_NORM_FLOAT, NULL, 0) ;
    cout << "norm: " << (normalisation?"true":"false") << endl;
#endif
    data.data_in = _samples;
    data.data_out = new float[data.output_frames];
#ifdef DEBUG
    cout << "channels "<<((chX<0)?sfinfo.channels:1) <<endl;
#endif
    if (src_simple(&data, SRC_SINC_BEST_QUALITY, (chX<0)?sfinfo.channels:1)) {    
      sf_close(sndfile);
      throw jconsistency_error("Error during samplerate conversion.");
    }
    delete[] _samples;
    _samples = data.data_out;
    _ttlSamples = data.output_frames_gen;
#ifdef DEBUG
    cout << "output_frames_gen: " << _ttlSamples << endl;
#endif
  }
#endif

  if (_sampleRate > outsamplerate)
    if (norm != 1.0 && norm != 0.0)
      for (int i=0; i < nsamples; i++)
	_samples[i] *= norm;

  _format = sfinfo.format;

  _cur        = 0;
  reset();
  _endOfSamples = false;

  sf_close(sndfile);
  return _ttlSamples;
}

void SampleFeature::addWhiteNoise(float snr)
{
  double desiredNoA;
  double avgSig = 0.0, avgNoi = 0.0;
  struct timeval now_time;
  short *noiseSamp = new short[ _ttlSamples];
  int max = INT_MIN;

  for(int i=0; i < _ttlSamples; i++)
    avgSig += fabsf(_samples[i]);
  avgSig = avgSig / (double)_ttlSamples;

  gettimeofday( &now_time, NULL);
  srand( now_time.tv_usec );

  for(int i=0;i<_ttlSamples;i++){
    noiseSamp[i] = rand();
    if( noiseSamp[i] > max )
      max = noiseSamp[i];
  }
  for(int i=0;i<_ttlSamples;i++){
    noiseSamp[i] = ( noiseSamp[i] / (float)max ) - 0.5;
    avgNoi += fabsf(noiseSamp[i]);
  }
  avgNoi = avgNoi / (double)_ttlSamples;

  //desiredNoA = pow( 10, ( log10(avgSig) - ( snr / 20.0 ) ) );
  desiredNoA = avgSig / pow( 10.0, snr/20.0 );

  //printf("Avg of Signal Amp %f dB\n", 20 * log10( avgSig ) );
  //printf("Add white noise with %f dB\n", 20 * log10( desiredNoA ) );
  for(int i=0;i<_ttlSamples;i++)
    noiseSamp[i] = desiredNoA * noiseSamp[i] / avgNoi;
  
  for(int i=0;i<_ttlSamples;i++)
    _samples[i] += noiseSamp[i];

  delete [] noiseSamp;
}

void SampleFeature::write(const String& fn, int format, int sampleRate)
{
  using namespace sndfile;
  SNDFILE* sndfile;
  SF_INFO sfinfo;
  int nsamp, frames = _ttlSamples;
  float norm;
  float *samplesorig = NULL;

  if (sampleRate == -1) sampleRate = _sampleRate;
#ifdef SRCONV
  sfinfo.samplerate = sampleRate;
#else
  sfinfo.samplerate = _sampleRate;
#endif
  sfinfo.format = format;
  sfinfo.channels = 1;
  sfinfo.frames = 0;
  sfinfo.sections = 0;
  sfinfo.seekable = 0;

  sndfile = sf_open(fn.c_str(), SFM_WRITE, &sfinfo);
  if (!sndfile)
    throw jio_error("Error opening file %s.", fn.c_str());

  if (_norm == 0.0) {
#ifdef DEBUG
    cout << "Disabling SNDFILE normalization option" << endl;
#endif
    sf_command (sndfile, SFC_SET_NORM_FLOAT, NULL, SF_FALSE);
  } else {
#ifdef DEBUG
    cout << "Enabling SNDFILE normalization option" << endl;
#endif
    sf_command (sndfile, SFC_SET_NORM_FLOAT, NULL, SF_TRUE);
  }
 
  if (_norm != 1.0 && _norm != 0.0) {
    norm = 1/_norm;
    samplesorig = _samples;
    _samples = new float[_ttlSamples];
    for (int i=0; i < _ttlSamples; i++)
      _samples[i] = samplesorig[i]*norm;
  }

#ifdef SRCONV
  if (_sampleRate != sfinfo.samplerate) {
    SRC_DATA data;
#ifdef DEBUG
    cout << "sample rate converting to " << sfinfo.samplerate << endl;
#endif
    data.input_frames = _ttlSamples;
    data.src_ratio = (float)sfinfo.samplerate / (float)_sampleRate;
    data.output_frames = (long)ceil(data.src_ratio*(float)_ttlSamples);
#ifdef DEBUG
    cout << "src_ratio: " << data.src_ratio << endl;
    cout << "output_frames: " << data.output_frames << endl;
    char normalisation = sf_command (sndfile, SFC_GET_NORM_FLOAT, NULL, 0) ;
    cout << "norm: " << (normalisation?"true":"false") << endl;
#endif
    data.data_in = _samples; //new float[data.input_frames];
    data.data_out = new float[data.output_frames];
    /*
    for (unsigned int i=0; i < _ttlSamples ; i++)
      data.data_in[i] = (float)_samples[i] / (float)0x8000;
    */
    if (src_simple(&data, SRC_SINC_BEST_QUALITY, 1)) {    
      sf_close(sndfile);
      if (samplesorig != NULL) {
	delete[] _samples;
	_samples = samplesorig;
      }
      throw jconsistency_error("Error during samplerate conversion.");
    }
    nsamp = sf_writef_float(sndfile, data.data_out, data.output_frames_gen);
    frames = data.output_frames_gen;
#ifdef DEBUG
    cout << "output_frames_gen: " << frames << endl;
#endif
  } else
#endif
    nsamp = sf_writef_float(sndfile, _samples, _ttlSamples);
  if(nsamp != int(frames))
    cerr << "unable to write " << (frames - nsamp) << " samples" << endl;
  sf_close(sndfile);
  if (samplesorig != NULL) {
    delete[] _samples;
    _samples = samplesorig;
  }
}

const gsl_vector_float* SampleFeature::data()
{
  if (NULL == _cpSamplesF) {
    _cpSamplesF = gsl_vector_float_calloc(_ttlSamples);
  } else {
    gsl_vector_float_free(_cpSamplesF);
    _cpSamplesF = gsl_vector_float_calloc(_ttlSamples);
  }

  for (unsigned i = 0; i < _ttlSamples; i++)
    gsl_vector_float_set(_cpSamplesF, i, _samples[i]);
  
  return _cpSamplesF;
}

const gsl_vector* SampleFeature::dataDouble()
{
  if( NULL == _cpSamplesD )
    _cpSamplesD = gsl_vector_calloc(_ttlSamples);
  else{
    gsl_vector_free( _cpSamplesD );
    _cpSamplesD = gsl_vector_calloc(_ttlSamples);
  }

  for (unsigned i = 0; i < _ttlSamples; i++)
    gsl_vector_set(_cpSamplesD, i, _samples[i]);

  return _cpSamplesD;
}


#define SLOW    -32768
#define SHIGH    32767
#define SLIMIT(x) \
  ((((x) < SLOW)) ? (SLOW) : (((x) < (SHIGH)) ? (x) : (SHIGH)))

void SampleFeature::zeroMean()
{
  double mean = 0.0;

  if (_samples == NULL)
    throw jconsistency_error("Must first load data before setting mean to zero.");

  for (unsigned i = 0; i < _ttlSamples; i++)
    mean += _samples[i];

  mean /= _ttlSamples;

  for (unsigned i = 0; i < _ttlSamples; i++)
    _samples[i] = int(SLIMIT(_samples[i] - mean));
}

void SampleFeature::cut(unsigned cfrom, unsigned cto)
{
  if (cfrom >= cto)
    throw j_error("Cut bounds (%d,%d) do not match.", cfrom, cto);

  if (cto >= _ttlSamples)
    throw j_error("Do not have enough samples (%d,%d).", cto, _ttlSamples);

  _ttlSamples = cto - cfrom + 1;
  float* newSamples = new float[_ttlSamples];
  memcpy(newSamples, _samples + cfrom, _ttlSamples * sizeof(float));

  delete[] _samples;
  _samples = newSamples;
}

// create a generator chosen by the environment variable GSL_RNG_TYPE
void SampleFeature::randomize(int startX, int endX, double sigma2)
{
  if (_r == NULL) {
    gsl_rng_env_setup();
    _T = gsl_rng_default;
    _r = gsl_rng_alloc(_T);
  }

  printf("Randomizing from %6.2f to %6.2f\n", startX / 16000.0, endX / 16000.0);
     
  for (int n = startX; n <= endX; n++) {
    _samples[n] = gsl_ran_gaussian(_r, sigma2);
  }
}

const gsl_vector_float* SampleFeature::next(int frameX)
{

  if (_endOfSamples) {
    // fprintf(stderr,"end of samples!");
    //if( NULL != _samples ){ delete [] _samples; }
    //_samples = NULL;
    throw jiterator_error("end of samples!");
  }

  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX){
    fprintf(stderr,"Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);
  }

  if (_cur >= _ttlSamples) {
    _endOfSamples = true;
    
    // cout << "Read " << _frameX << " blocks of audio data." << endl;
    //if( NULL != _samples ){ delete [] _samples; }
    //_samples = NULL;
    throw jiterator_error("end of samples!");
  }
    
  if (_cur + size() >= _ttlSamples) {
    if (_padZeros) {
      gsl_vector_float_set_zero(_vector);
      unsigned remainingN = _ttlSamples - _cur;
      for (unsigned i = 0; i < remainingN; i++)
	gsl_vector_float_set(_vector, i, _samples[_cur + i]);
    } else {
      _endOfSamples = true;
      
      // cout << "Read " << _frameX << " blocks of audio data." << endl;
      //if( NULL != _samples ){ delete [] _samples; }
      //_samples = NULL;
      throw jiterator_error("end of samples!");
    }
  } else {
    for (unsigned i = 0; i < size(); i++)
      gsl_vector_float_set(_vector, i, _samples[_cur + i]);
  }

  _cur += _shiftLen;

  _increment();
  return _vector;
}

void SampleFeature::copySamples(SampleFeaturePtr& src, unsigned cfrom, unsigned to)
{
  if( NULL != _samples ){ delete [] _samples; }
  if (to == 0) {
    _ttlSamples = src->_ttlSamples;
  } else {
    if (to <= cfrom)
      throw jindex_error("cfrom = %d and to = %d are inconsistent.", cfrom, to);
    if (to >= src->_ttlSamples)
      to = src->_ttlSamples - 1;

    _ttlSamples = to - cfrom;
  }
    
  _samples = new float[_ttlSamples];
  memcpy(_samples, src->_samples + cfrom, _ttlSamples * sizeof(float));
}

void SampleFeature::setSamples(const gsl_vector* samples, unsigned sampleRate)
{
  if( NULL != _samples ){ delete [] _samples; }
  _sampleRate = sampleRate;
  _ttlSamples = samples->size;
  _samples = new float[_ttlSamples];
  for (unsigned sampleX = 0; sampleX < _ttlSamples; sampleX++)
    _samples[sampleX] = float(gsl_vector_get(samples, sampleX));

  reset();
}

// ----- methods for class `IterativeSingleChannelSampleFeature' -----
//
IterativeSingleChannelSampleFeature::IterativeSingleChannelSampleFeature( unsigned blockLen, const String& nm )
  : VectorFloatFeatureStream(blockLen, nm), _blockLen(blockLen), _cur(0)
{ 
  _interval	 = 30;
  _sndfile = NULL;
  _samples = NULL;
  _last = false;
}

IterativeSingleChannelSampleFeature::~IterativeSingleChannelSampleFeature()
{
  if (_sndfile != NULL) 
    sf_close(_sndfile);
  if( _samples != NULL )
    delete[] _samples;
  _sndfile = NULL;
  _samples = NULL;
}

void IterativeSingleChannelSampleFeature::reset()
{
  _ttlSamples = _cur = 0;  _last = false;  VectorFloatFeatureStream::reset();
}

void IterativeSingleChannelSampleFeature::read(const String& fileName, int format, int samplerate, int cfrom, int cto )
{
  using namespace sndfile;

  delete[] _samples;  _samples = NULL;
  if (_sndfile != NULL) {  sf_close(_sndfile); _sndfile = NULL; }

  _sfinfo.channels   = 1;
  _sfinfo.samplerate = samplerate;
  _sfinfo.format     = format;

  _sndfile = sf_open(fileName.c_str(), SFM_READ, &_sfinfo);
  if (!_sndfile)
    throw jio_error("Could not open file %s.", fileName.c_str());

  if (sf_error(_sndfile)) {
    sf_close(_sndfile);
    throw jio_error("sndfile error: %s.", sf_strerror(_sndfile));
  }

  if( _sfinfo.channels > 1 ){
    sf_close(_sndfile);
    fprintf(stderr,"IterativeSingleChannelSampleFeature is for the single channel file only\n");
    throw j_error("IterativeSingleChannelSampleFeature is for the single channel file only\n");
  }
#ifdef DEBUG
  cout << "channels: "   << _sfinfo.channels   << endl;
  cout << "frames: "     << _sfinfo.frames     << endl;
  cout << "samplerate: " << _sfinfo.samplerate << endl;
#endif

  sf_command(_sndfile, SFC_SET_NORM_FLOAT, NULL, SF_FALSE);
  _blockN     = _interval * _sfinfo.samplerate / _blockLen + 1;
  _sampleN    = _blockN   * _blockLen;
  _samples = new float[_sampleN];

  for (unsigned i = 0; i < _sampleN; i++)
    _samples[i] = 0.0;

  if (sf_seek(_sndfile, cfrom, SEEK_SET) == -1)
    throw jio_error("Error seeking to %d", cfrom);

  if (cto > 0 and cto < cfrom)
    throw jconsistency_error("Segment cannot start at %d and end at %d", cfrom, cto);
  _cto = cto - cfrom;
}

const gsl_vector_float* IterativeSingleChannelSampleFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  unsigned currentFrame = _cur % _blockN;
  if ( currentFrame == 0) {

    if (_last || (_cto > 0 && _cur * _blockLen > _cto)){
      // fprintf(stderr,"end of samples!");
      delete[] _samples;  _samples = NULL;
      if (_sndfile != NULL) {  sf_close(_sndfile); _sndfile = NULL; }
      throw jiterator_error("end of samples!");
    }

    for (unsigned i = 0; i < _sampleN; i++)
      _samples[i] = 0.0;
    unsigned readN = sf_readf_float(_sndfile, _samples, _sampleN);

    _ttlSamples += readN;

    if (readN < _sampleN) _last = true;
  }

  unsigned offset = currentFrame * _blockLen;
  for (unsigned i = 0; i < _blockLen; i++)
    gsl_vector_float_set(_vector, i, _samples[offset+i]);

  _cur++;
  _increment();
  return _vector;
}


// ----- methods for class `IterativeSampleFeature' -----
//
float*            IterativeSampleFeature::_allSamples    = NULL;
sndfile::SNDFILE* IterativeSampleFeature::_sndfile       = NULL;
sndfile::SF_INFO  IterativeSampleFeature::_sfinfo;
unsigned          IterativeSampleFeature::_interval	 = 30;
unsigned          IterativeSampleFeature::_blockN;
unsigned          IterativeSampleFeature::_sampleN;
unsigned          IterativeSampleFeature::_allSampleN;
unsigned          IterativeSampleFeature::_ttlSamples;

IterativeSampleFeature::IterativeSampleFeature(unsigned chX, unsigned blockLen, unsigned firstChanX, const String& nm)
  : VectorFloatFeatureStream(blockLen, nm), _blockLen(blockLen), _chanX(chX),  _firstChanX(firstChanX), _cur(0){ }

IterativeSampleFeature::~IterativeSampleFeature()
{
  if (_sndfile != NULL) 
    sf_close(_sndfile);
  _sndfile = NULL;
  if( _allSamples != NULL )
    delete[] _allSamples;
  _allSamples = NULL;
}

void IterativeSampleFeature::reset()
{
  _ttlSamples = _cur = 0;  _last = false;  VectorFloatFeatureStream::reset();
}

void IterativeSampleFeature::read(const String& fileName, int format, int samplerate, int chN, int cfrom, int cto )
{
  using namespace sndfile;

  if ( _chanX != _firstChanX ) return;

  delete[] _allSamples;  _allSamples = NULL;
  if (_sndfile != NULL) { sf_close(_sndfile); _sndfile = NULL; }

  _sfinfo.channels   = chN;
  _sfinfo.samplerate = samplerate;
  _sfinfo.format     = format;

  _sndfile = sf_open(fileName.c_str(), SFM_READ, &_sfinfo);
  if (!_sndfile)
    throw jio_error("Could not open file %s.", fileName.c_str());

  if (sf_error(_sndfile)) {
    sf_close(_sndfile);
    throw jio_error("sndfile error: %s.", sf_strerror(_sndfile));
  }

#ifdef DEBUG
  cout << "channels: "   << _sfinfo.channels   << endl;
  cout << "frames: "     << _sfinfo.frames     << endl;
  cout << "samplerate: " << _sfinfo.samplerate << endl;
#endif

  sf_command(_sndfile, SFC_SET_NORM_FLOAT, NULL, SF_FALSE);
  _blockN     = _interval * _sfinfo.samplerate / _blockLen + 1;
  _sampleN    = _blockN   * _blockLen;
  _allSampleN = _sampleN  * _sfinfo.channels;
  _allSamples = new float[_allSampleN];

  if (sf_seek(_sndfile, cfrom, SEEK_SET) == -1)
    throw jio_error("Error seeking to %d", cfrom);

  if (cto > 0 and cto < cfrom)
    throw jconsistency_error("Segment cannot start at %d and end at %d", cfrom, cto);
  _cto = cto - cfrom;
}

const gsl_vector_float* IterativeSampleFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  unsigned currentFrame = _cur % _blockN;
  if (_chanX == _firstChanX && currentFrame == 0) {

    if (_last || (_cto > 0 && _cur * _blockLen > _cto)){
      // fprintf(stderr,"end of samples!");
      throw jiterator_error("end of samples!");
    }

    for (unsigned i = 0; i < _allSampleN; i++)
      _allSamples[i] = 0.0;
    unsigned readN = sf_readf_float(_sndfile, _allSamples, _sampleN);

    _ttlSamples += readN;

    if (readN < _sampleN) _last = true;
  }

  unsigned offset = currentFrame * _sfinfo.channels * _blockLen;
  for (unsigned i = 0; i < _blockLen; i++)
    gsl_vector_float_set(_vector, i, _allSamples[offset + i * _sfinfo.channels + _chanX]);

  _cur++;
  _increment();
  return _vector;
}


// ----- methods for class `BlockSizeConversionFeature' -----
//
BlockSizeConversionFeature::
BlockSizeConversionFeature(const VectorFloatFeatureStreamPtr& src,
			   unsigned blockLen,
			   unsigned shiftLen, const String& nm)
  : VectorFloatFeatureStream(blockLen, nm), _src(src),
    _inputLen(_src->size()), _blockLen(blockLen), _shiftLen(shiftLen),
    _overlapLen(_blockLen - _shiftLen), _curIn(0), _curOut(0), _srcFrameX(-1)
{
  if (_blockLen < _shiftLen)
    throw jdimension_error("Block length (%d) is less than shift length (%d).\n",
			   _blockLen, _shiftLen);

  //if (_inputLen < _shiftLen)
  //  throw jdimension_error("Input length (%d) is less than shift length (%d).\n",
  //			   _inputLen, _shiftLen);

  printf("Block Size Conversion Input Size  = %d\n", _src->size());
  printf("Block Size Conversion Shift Size  = %d\n", _shiftLen);
  printf("Block Size Conversion Output Size = %d\n", size());
}

void BlockSizeConversionFeature::_inputLonger()
{
  if (_frameX == FrameResetX) {
    _srcFrameX++;
    _srcFeat = _src->next(_srcFrameX);
    memcpy(_vector->data, _srcFeat->data, _blockLen * sizeof(float));
    _curIn += _blockLen;
    return;
  }

  if (_overlapLen > 0)
    memmove(_vector->data, _vector->data + _shiftLen, _overlapLen * sizeof(float));

  if (_curIn + _shiftLen < _inputLen) {
    memcpy(_vector->data + _overlapLen, _srcFeat->data + _curIn, _shiftLen * sizeof(float));
    _curIn += _shiftLen;
  } else {
    int remaining = _inputLen - _curIn;

    if (remaining < 0)
	throw jconsistency_error("Remaining sample (%d) cannot be negative.\n", remaining);

    if (remaining > 0)
      memcpy(_vector->data + _overlapLen, _srcFeat->data + _curIn, remaining * sizeof(float));
    
    _curIn = 0;
    _srcFrameX++;
    _srcFeat = _src->next(_srcFrameX);
    unsigned fromNew = _shiftLen - remaining;

    memcpy(_vector->data + _overlapLen + remaining, _srcFeat->data + _curIn, fromNew * sizeof(float));
    _curIn += fromNew;
  }
}

void BlockSizeConversionFeature::_outputLonger()
{
  if (_frameX == FrameResetX) {
    while (_curOut + _inputLen <= _blockLen) {
      _srcFrameX++;
      _srcFeat = _src->next(_srcFrameX);
      memcpy(_vector->data + _curOut, _srcFeat->data, _inputLen * sizeof(float));
      _curOut += _inputLen;
    }

    int remaining = _blockLen - _curOut;
    if (remaining > 0) {
      _srcFrameX++;
      _srcFeat = _src->next(_srcFrameX);
      memcpy(_vector->data + _curOut, _srcFeat->data, remaining * sizeof(float));
      _curIn += remaining;
    }
    _curOut = 0;
    return;
  }

  if (_overlapLen > 0) {
    memmove(_vector->data, _vector->data + _shiftLen, _overlapLen * sizeof(float));
    _curOut += _overlapLen;
  }

  if (_curIn > 0) {
    int remaining = _inputLen - _curIn;
    if (remaining > 0) {
      memcpy(_vector->data + _curOut, _srcFeat->data + _curIn, remaining * sizeof(float));
      _curOut += remaining;
    }
    _curIn = 0;
  }

  while (_curOut + _inputLen <= _blockLen) {
    _srcFrameX++;
    _srcFeat = _src->next(_srcFrameX);
    memcpy(_vector->data + _curOut, _srcFeat->data, _inputLen * sizeof(float));
    _curOut += _inputLen;
  }

  int remaining = _blockLen - _curOut;
  if (remaining > 0) {
    _srcFrameX++;
    _srcFeat = _src->next(_srcFrameX);
    memcpy(_vector->data + _curOut, _srcFeat->data, remaining * sizeof(float));
    _curIn += remaining;
  }
  _curOut = 0;
}

const gsl_vector_float* BlockSizeConversionFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (_inputLen > _shiftLen)
    _inputLonger();
  else
    _outputLonger();

  _increment();
  return _vector;
}


// ----- methods for class `BlockSizeConversionFeatureShort' -----
//
BlockSizeConversionFeatureShort::
BlockSizeConversionFeatureShort(VectorShortFeatureStreamPtr& src,
			   unsigned blockLen,
			   unsigned shiftLen, const String& nm)
  : VectorShortFeatureStream(blockLen, nm), _src(src),
    _inputLen(_src->size()), _blockLen(blockLen), _shiftLen(shiftLen),
    _overlapLen(_blockLen - _shiftLen), _curIn(0), _curOut(0)
{
  if (_blockLen < _shiftLen)
    throw jdimension_error("Block length (%d) is less than shift length (%d).\n",
			   _blockLen, _shiftLen);

  //if (_inputLen < _shiftLen)
  //  throw jdimension_error("Input length (%d) is less than shift length (%d).\n",
  //			   _inputLen, _shiftLen);

  printf("Block Size Conversion Short Input Size  = %d\n", _src->size());
  printf("Block Size Conversion Short Shift Size  = %d\n", _shiftLen);
  printf("Block Size Conversion Short Output Size = %d\n", size());
}

void BlockSizeConversionFeatureShort::_inputLonger()
{
  if (_frameX == FrameResetX) {
    _srcFeat = _src->next();
    memcpy(_vector->data, _srcFeat->data, _blockLen * sizeof(short));
    _curIn += _blockLen;
    return;
  }

  if (_overlapLen > 0)
    memmove(_vector->data, _vector->data + _shiftLen, _overlapLen * sizeof(short));

  if (_curIn + _shiftLen < _inputLen) {
    memcpy(_vector->data + _overlapLen, _srcFeat->data + _curIn, _shiftLen * sizeof(short));
    _curIn += _shiftLen;
  } else {
    int remaining = _inputLen - _curIn;

    if (remaining < 0)
	throw jconsistency_error("Remaining sample (%d) cannot be negative.\n", remaining);

    if (remaining > 0)
      memcpy(_vector->data + _overlapLen, _srcFeat->data + _curIn, remaining * sizeof(short));
    
    _curIn = 0; _srcFeat = _src->next();
    unsigned fromNew = _shiftLen - remaining;

    memcpy(_vector->data + _overlapLen + remaining, _srcFeat->data + _curIn, fromNew * sizeof(short));
    _curIn += fromNew;
  }
}

void BlockSizeConversionFeatureShort::_outputLonger()
{
  if (_frameX == FrameResetX) {
    while (_curOut + _inputLen <= _blockLen) {
      _srcFeat = _src->next();
      memcpy(_vector->data + _curOut, _srcFeat->data, _inputLen * sizeof(short));
      _curOut += _inputLen;
    }

    int remaining = _blockLen - _curOut;
    if (remaining > 0) {
      _srcFeat = _src->next();
      memcpy(_vector->data + _curOut, _srcFeat->data, remaining * sizeof(short));
      _curIn += remaining;
    }
    _curOut = 0;
    return;
  }

  if (_overlapLen > 0) {
    memmove(_vector->data, _vector->data + _shiftLen, _overlapLen * sizeof(short));
    _curOut += _overlapLen;
  }

  if (_curIn > 0) {
    int remaining = _inputLen - _curIn;
    if (remaining > 0) {
      memcpy(_vector->data + _curOut, _srcFeat->data + _curIn, remaining * sizeof(short));
      _curOut += remaining;
    }
    _curIn = 0;
  }

  while (_curOut + _inputLen <= _blockLen) {
    _srcFeat = _src->next();
    memcpy(_vector->data + _curOut, _srcFeat->data, _inputLen * sizeof(short));
    _curOut += _inputLen;
  }

  int remaining = _blockLen - _curOut;
  if (remaining > 0) {
    _srcFeat = _src->next();
    memcpy(_vector->data + _curOut, _srcFeat->data, remaining * sizeof(short));
    _curIn += remaining;
  }
  _curOut = 0;
}

const gsl_vector_short* BlockSizeConversionFeatureShort::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (_inputLen > _shiftLen)
    _inputLonger();
  else
    _outputLonger();

  _increment();
  return _vector;
}


// ----- methods for class `PreemphasisFeature' -----
//
PreemphasisFeature::PreemphasisFeature(const VectorFloatFeatureStreamPtr& samp, double mu, const String& nm)
  : VectorFloatFeatureStream(samp->size(), nm), _samp(samp), _prior(0.0), _mu(mu)
{
  printf("Preemphasis Feature Input Size  = %d\n", _samp->size());
  printf("Preemphasis Feature Output Size = %d\n", size());
}

void PreemphasisFeature::nextSpeaker()
{
  _prior = 0.0;
}

const gsl_vector_float* PreemphasisFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector_float* block = _samp->next(_frameX + 1);
  _increment();

  for (unsigned i = 0; i < size(); i++) {
    gsl_vector_float_set(_vector, i, gsl_vector_float_get(block, i) - _mu * _prior);
    _prior = gsl_vector_float_get(block, i);
  }

  return _vector;
}


// ----- methods for class `HammingFeatureShort' -----
//
HammingFeatureShort::HammingFeatureShort(const VectorShortFeatureStreamPtr& samp, const String& nm)
  : VectorFloatFeatureStream(samp->size(), nm), _samp(samp), _windowLen(samp->size()),
    _window(new double[_windowLen])
{
  double temp = 2. * M_PI / (double)(_windowLen - 1);
  for ( unsigned i = 0 ; i < _windowLen; i++ )
    _window[i] = 0.54 - 0.46*cos(temp*i);

  printf("Hamming Feature Input Size  = %d\n", _samp->size());
  printf("Hamming Feature Output Size = %d\n", size());
}

const gsl_vector_float* HammingFeatureShort::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector_short* block = _samp->next(_frameX + 1);
  _increment();

  for (unsigned i = 0; i < _windowLen; i++)
    gsl_vector_float_set(_vector, i, _window[i] * gsl_vector_short_get(block, i));

  return _vector;
}


// ----- methods for class `HammingFeature' -----
//
HammingFeature::HammingFeature(const VectorFloatFeatureStreamPtr& samp, const String& nm)
  : VectorFloatFeatureStream(samp->size(), nm), _samp(samp), _windowLen(samp->size()),
    _window(new double[_windowLen])
{
  double temp = 2. * M_PI / (double)(_windowLen - 1);
  for ( unsigned i = 0 ; i < _windowLen; i++ )
    _window[i] = 0.54 - 0.46*cos(temp*i);

  printf("Hamming Feature Input Size  = %d\n", _samp->size());
  printf("Hamming Feature Output Size = %d\n", size());
}

const gsl_vector_float* HammingFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector_float* block = _samp->next(_frameX + 1);
  _increment();

  for (unsigned i = 0; i < _windowLen; i++)
    gsl_vector_float_set(_vector, i, _window[i] * gsl_vector_float_get(block, i));

  return _vector;
}


// ----- methods for class `FFTFeature' -----
//
FFTFeature::FFTFeature(const VectorFloatFeatureStreamPtr& samp, unsigned fftLen, const String& nm)
  : VectorComplexFeatureStream(fftLen, nm), _samp(samp), _fftLen(fftLen), _windowLen(_samp->size()),
#ifdef HAVE_LIBFFTW3
    _samples(static_cast<double*>(fftw_malloc(sizeof(double) * _fftLen))), _output(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * (_fftLen / 2 + 1))))
#else
  _samples(new double[_fftLen])
#endif
{
  for (unsigned i = 0; i < _fftLen; i++)
    _samples[i] = 0.0;

  printf("FFT Feature Input Size  = %d\n", _samp->size());
  printf("FFT Feature Output Size = %d\n", size());

#ifdef HAVE_LIBFFTW3
  fftw_init_threads();
  fftw_plan_with_nthreads(2);
  _fftwPlan = fftw_plan_dft_r2c_1d(_fftLen, _samples, _output, FFTW_MEASURE);
#endif
}

FFTFeature::~FFTFeature()
{
#ifdef HAVE_LIBFFTW3
  fftw_destroy_plan(_fftwPlan);
  fftw_free(_samples);
  fftw_free(_output);
#else
  delete[] _samples;
#endif
}

const gsl_vector_complex* FFTFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector_float* block = _samp->next(_frameX + 1);
  _increment();

  for (unsigned i = 0; i < _windowLen; i++)
    _samples[i] = gsl_vector_float_get(block, i);
  for (unsigned i = _windowLen; i < _fftLen; i++)
    _samples[i] = 0.0;

#ifdef HAVE_LIBFFTW3
  fftw_execute(_fftwPlan);
  fftwUnpack(_vector, _output);
#else
  gsl_fft_real_radix2_transform(_samples, /*stride=*/ 1, _fftLen);
  halfComplexUnpack(_vector, _samples);
#endif

  return _vector;
}


// ----- methods for class `SpectralPowerFloatFeature' -----
//
SpectralPowerFloatFeature::
SpectralPowerFloatFeature(const VectorComplexFeatureStreamPtr& fft, unsigned powN, const String& nm) :
  VectorFloatFeatureStream(powN == 0 ? fft->size() : powN, nm), _fft(fft)
{
  if (size() != fft->size() && size() != (fft->size() / 2) + 1)
    throw jconsistency_error("Number of power coefficients %d does not match FFT length %d.",
			     size(), fft->size());

  printf("Spectral Power Feature Input Size  = %d\n", _fft->size());
  printf("Spectral Power Feature Output Size = %d\n", size());
}

const gsl_vector_float* SpectralPowerFloatFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector_complex* fftVec = _fft->next(_frameX + 1);
  _increment();

  for (unsigned i = 0; i < size(); i++)
    gsl_vector_float_set(_vector, i, gsl_complex_abs2(gsl_vector_complex_get(fftVec, i)));

  return _vector;
}


// ----- methods for class `SpectralPowerFeature' -----
//
SpectralPowerFeature::
SpectralPowerFeature(const VectorComplexFeatureStreamPtr& fft, unsigned powN, const String& nm) :
  VectorFeatureStream(powN == 0 ? fft->size() : powN, nm), _fft(fft)
{
  if (size() != fft->size() && size() != (fft->size() / 2) + 1)
    throw jconsistency_error("Number of power coefficients %d does not match FFT length %d.",
			     size(), fft->size());

  printf("Spectral Power Feature Input Size  = %d\n", _fft->size());
  printf("Spectral Power Feature Output Size = %d\n", size());
}

const gsl_vector* SpectralPowerFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector_complex* fftVec = _fft->next(_frameX + 1);
  _increment();

  for (unsigned i = 0; i < size(); i++)
    gsl_vector_set(_vector, i, gsl_complex_abs2(gsl_vector_complex_get(fftVec, i)));

  return _vector;
}


// ----- methods for class `SignalPowerFeature' -----
//
const gsl_vector_float* SignalPowerFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector_float* block = _samp->next(_frameX + 1);
  _increment();

  double power = 0.0;
  for (unsigned i = 0; i < _samp->size(); i++) {
    double val = gsl_vector_float_get(block, i);
    power += val * val;
  }
  gsl_vector_float_set(_vector, 0, power / _samp->size() / _range);

  return _vector;
}


// ----- methods for class `ALogFeature' -----
//
const gsl_vector_float* ALogFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector_float* block = _samp->next(_frameX + 1);
  _increment();

  if (_minMaxFound == false)
    _findMinMax(block);
  float b = _max / pow(10.0, _a);

  for (unsigned i = 0; i < size(); i++) {
    double val = b + gsl_vector_float_get(block, i);

    if (val <= 0.0) val = 1.0;

    gsl_vector_float_set(_vector, i, _m * log10(val));
  }

  return _vector;
}

void ALogFeature::reset()
{
  _samp->reset(); VectorFloatFeatureStream::reset();

  if (_runon) return;

  _min = HUGE; _max = -HUGE; _minMaxFound = false;
}

void ALogFeature::_findMinMax(const gsl_vector_float* block)
{
  if (_runon) {
    for (unsigned i = 0; i < block->size; i++) {
      float val = gsl_vector_float_get(block, i);
      if (val < _min) _min = val;
      if (val > _max) _max = val;
    }
    return;
  }

  int frameX = 0;
  while (true) {
    try {
      block = _samp->next(frameX);
      for (unsigned i = 0; i < block->size; i++) {
	float val = gsl_vector_float_get(block, i);
	if (val < _min) _min = val;
	if (val > _max) _max = val;
      }
      frameX++;
    } catch (jiterator_error& e) {
      _minMaxFound = true;
      return;
    }
  }
}


// ----- methods for class `NormalizeFeature' -----
//
NormalizeFeature::
NormalizeFeature(const VectorFloatFeatureStreamPtr& samp, double min, double max, bool runon, const String& nm)
  : VectorFloatFeatureStream(samp->size(), nm), _samp(samp), _min(min), _max(max), _range(_max - _min),
    _xmin(HUGE), _xmax(-HUGE), _minMaxFound(false), _runon(runon) { }

const gsl_vector_float* NormalizeFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector_float* block = _samp->next(_frameX + 1);
  _increment();

  /*------------------------------------------------------------
  ;  y = ((x - xmin)/xrange) * yrange + ymin
  ;    =   x * (yrange/xrange)  - xmin*yrange/xrange + ymin
  ;    =   x * factor           - xmin*factor + ymin
  ;-----------------------------------------------------------*/
  if (_minMaxFound == false) _findMinMax(block);
  double xrange = _xmax - _xmin;
  double factor = _range / xrange;
  double add    = _min - _xmin * factor;

  for (unsigned i = 0; i < size(); i++)
    gsl_vector_float_set(_vector, i, gsl_vector_float_get(block, i) * factor + add);

  return _vector;
}

void NormalizeFeature::reset()
{
  _samp->reset();  VectorFloatFeatureStream::reset(); 

  if (_runon) return;

  _xmin = HUGE; _xmax = -HUGE; _minMaxFound = false;
}

void NormalizeFeature::_findMinMax(const gsl_vector_float* block)
{
  if (_runon) {
    for (unsigned i = 0; i < block->size; i++) {
      float val = gsl_vector_float_get(block, i);
      if (val < _xmin) _xmin = val;
      if (val > _xmax) _xmax = val;
    }
    return;
  }

  int frameX = 0;
  while (true) {
    try {
      block = _samp->next(frameX);
      for (unsigned i = 0; i < block->size; i++) {
	float val = gsl_vector_float_get(block, i);
	if (val < _xmin) _xmin = val;
	if (val > _xmax) _xmax = val;
      }
      frameX++;
    } catch (jiterator_error& e) {
      _minMaxFound = true;
      return;
    }
  }
}


// ----- methods for class `ThresholdFeature' -----
//
ThresholdFeature::
ThresholdFeature(const VectorFloatFeatureStreamPtr& samp, double value, double thresh,
		 const String& mode, const String& nm)
  : VectorFloatFeatureStream(samp->size(), nm), _samp(samp), _value(value), _thresh(thresh)
{
  if (mode == "upper")
    _compare =  1;
  else if (mode == "lower")
    _compare = -1;
  else if (mode == "both")
    _compare =  0;
  else
    throw jkey_error("Mode %s is not supported", mode.c_str());
}

const gsl_vector_float* ThresholdFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector_float* block = _samp->next(_frameX + 1);
  _increment();

  for (unsigned i = 0; i < size(); i++) {
    double v = gsl_vector_float_get(block, i);

    if (_compare > 0) {
      if (v >= _thresh) v = _value;
    } else if (_compare == 0) {
      if (v >= _thresh) v = _value;
      else if (v <= -_thresh) v = -_value;
    } else if (_compare < 0) {
      if (v <= _thresh) v = _value;
    }

    gsl_vector_float_set(_vector, i, v);
  }

  return _vector;
}


// ----- methods for class `SpectralResamplingFeature' -----
//
const double SpectralResamplingFeature::SampleRatio = 16.0 / 22.05;

SpectralResamplingFeature::
SpectralResamplingFeature(const VectorFeatureStreamPtr& src, double ratio, unsigned len,
		  const String& nm)
  : VectorFeatureStream((len == 0 ? src->size() : len), nm), _src(src), _ratio(ratio * float(src->size()) / float(size())), _len(len)
{
  if (_ratio > 1.0)
    throw jconsistency_error("Must resample the spectrum to a higher rate (ratio = %10.4f < 1.0).",
			     _ratio);
}

SpectralResamplingFeature::~SpectralResamplingFeature() { }

const gsl_vector* SpectralResamplingFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector* srcVec = _src->next(_frameX + 1);
  _increment();

  for (unsigned coeffX = 0; coeffX < size(); coeffX++) {
    float    exact = coeffX * _ratio;
    unsigned low   = unsigned(coeffX * _ratio);
    unsigned high  = low + 1;

    float    wgt   = high - exact;
    float    coeff = wgt * gsl_vector_get(srcVec, low)
      + (1.0 - wgt) * gsl_vector_get(srcVec, high);

    gsl_vector_set(_vector, coeffX, coeff);
  }

  return _vector;
}


#ifdef SRCONV

// ----- methods for class `SamplerateConversionFeature' -----
//
SamplerateConversionFeature::
SamplerateConversionFeature(const VectorFloatFeatureStreamPtr& src, unsigned sourcerate, unsigned destrate,
			    unsigned len, const String& method, const String& nm)
  : VectorFloatFeatureStream((len == 0 ? unsigned(unsigned(src->size() * float(destrate)/ float(sourcerate))) : len), nm),
    _src(src), _dataInSamplesN(0), _dataOutStartX(0), _dataOutSamplesN(0)
{
  if (method == "best")
    _state = src_new(SRC_SINC_BEST_QUALITY, 1, &_error);
  else if (method == "medium")
    _state = src_new(SRC_SINC_MEDIUM_QUALITY, 1, &_error);
  else if (method == "fastest")
    _state = src_new(SRC_SINC_FASTEST, 1, &_error);
  else if (method == "zoh")
    _state = src_new(SRC_ZERO_ORDER_HOLD, 1, &_error);
  else if (method == "linear")
    _state = src_new(SRC_LINEAR, 1, &_error);
  else
    throw jconsistency_error("Cannot recognize type (%s) of sample rate converter.", method.c_str());

  if (_state == NULL)
    throw j_error("Error while initializing the samplerate converter: %s", src_strerror(_error));

  _data.src_ratio     = float(destrate) / float(sourcerate);
  _data.input_frames  = _src->size();
  _data.output_frames = size();
  printf("output frames %d\n", _data.output_frames);
  printf("input frames %d\n",  _data.input_frames);
  _data.end_of_input  = 0;

  _data.data_in  = new float[2 * _data.input_frames];
  _data.data_out = new float[_data.output_frames];
}

SamplerateConversionFeature::~SamplerateConversionFeature()
{
  _state = src_delete(_state);

  delete[] _data.data_in;
  delete[] _data.data_out;
}

void SamplerateConversionFeature::reset()
{
  src_reset(_state);  _src->reset(); VectorFloatFeatureStream::reset();
}

const gsl_vector_float* SamplerateConversionFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  // copy samples remaining in '_data.data_out' from prior iteration
  unsigned outputBufferSamplesN = 0;
  if (_dataOutSamplesN > 0) {
    memcpy(_vector->data, _data.data_out + _dataOutStartX, _dataOutSamplesN * sizeof(float));
    outputBufferSamplesN += _dataOutSamplesN;
    _dataOutStartX = _dataOutSamplesN = 0;
  }

  // iterate until '_vector' is full
  while (outputBufferSamplesN < size()) {

    // copy samples from '_src->next()'
    if (_dataInSamplesN < _src->size()) {
      const gsl_vector_float* srcVec = _src->next();
      memcpy(_data.data_in + _dataInSamplesN, srcVec->data, _src->size() * sizeof(float));
      _dataInSamplesN += _src->size();
    }

    // process the current buffer
    _error = src_process(_state, &_data);
    
    // copy generated samples to '_vector'
    unsigned outputSamplesCopied = size() - outputBufferSamplesN;
    if (outputSamplesCopied > _data.output_frames_gen)
      outputSamplesCopied = _data.output_frames_gen;
    memcpy(_vector->data + outputBufferSamplesN, _data.data_out, outputSamplesCopied * sizeof(float));
    outputBufferSamplesN += outputSamplesCopied;
    _dataOutStartX   = outputSamplesCopied;
    _dataOutSamplesN = _data.output_frames_gen - outputSamplesCopied;

    // copy down remaining samples in '_data.data_in'
    _dataInSamplesN -= _data.input_frames_used;
    memmove(_data.data_in, _data.data_in + _data.input_frames_used, _dataInSamplesN * sizeof(float));
  }
    
  _increment();
  return _vector;
}

#endif // SRCONV

// ----- methods for class 'VTLNFeature' -----
//
VTLNFeature::VTLNFeature(const VectorFeatureStreamPtr& pow,
			 unsigned coeffN, double ratio, double edge, int version, 
			 const String& nm)
  : VectorFeatureStream(coeffN == 0 ? pow->size() : coeffN, nm), _pow(pow), _ratio(ratio),
    _edge(edge), _version(version)
{
  printf("Linear VTLN Size = %d\n", size());
  _auxV = gsl_vector_alloc(coeffN);
}


const gsl_vector* VTLNFeature::nextOrg(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector* powVec = _pow->next(_frameX + 1);
  _increment();

  double yedge = (_edge < _ratio) ? (_edge / _ratio)              : 1.0;
  double b     = (yedge < 1.0)    ? (1.0 - _edge) / (1.0 - yedge) : 0;

  for (unsigned coeffX = 0; coeffX < size(); coeffX++) {

    double Y0 = double(coeffX)   / double(size());
    double Y1 = double(coeffX+1) / double(size());

    double X0 = ((Y0 < yedge) ? (_ratio * Y0) :
		 (b     * Y0 +  1.0 - b)) * size();
    double X1 = ((Y1 < yedge) ? (_ratio * Y1) :
		 (b     * Y1 +  1.0 - b)) * size();

    int    Lower_coeffY1 = int(X1);
    double alpha1        = X1 - Lower_coeffY1;

    int    Lower_coeffY0 = int(X0);
    double alpha0        = int(X0) + 1 - X0;

    double z             =  0.0;

    if (Lower_coeffY0 >= powVec->size)
      Lower_coeffY0 = powVec->size - 1;
         
    if (Lower_coeffY1 > powVec->size)
      Lower_coeffY1 = powVec->size;

    if ( Lower_coeffY0 == Lower_coeffY1) {
      z += (X1-X0) * gsl_vector_get(powVec, Lower_coeffY0);
    } else {
      z += alpha0  * gsl_vector_get(powVec, Lower_coeffY0);

      for (int i = Lower_coeffY0+1; i < Lower_coeffY1; i++)
	z += gsl_vector_get(powVec, i);

      if ( Lower_coeffY1 < int(powVec->size))
	z += alpha1 * gsl_vector_get(powVec, Lower_coeffY1);
    }

    gsl_vector_set(_vector, coeffX, z);
  }

  return _vector;
}

const gsl_vector* VTLNFeature::nextFF(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector* powVec = _pow->next(_frameX + 1);
  _increment();

  unsigned N	= size();  
  float b	= N * _edge;
  float slope1	= _ratio;
  float slope2	= _ratio;
  if (slope1 < 1.0)
    slope2 = (N - slope1 * b) / (N - b);
  //printf("%f %f %f\n", b, slope1, slope2);
  
  gsl_vector_set_zero(_vector);
  gsl_vector_set_zero(_auxV);
  
  for (int sIdx = 0; sIdx < N; sIdx++) {
    
    float sIdx1	= sIdx - 0.5;
    float sIdx2	= sIdx + 0.5;
    float v	= gsl_vector_get(powVec, sIdx);
    float dIdx1 = sIdx1*slope1;
    if (sIdx1 > b)
      dIdx1 = b * slope1 + (sIdx1 - b) * slope2;
    float dIdx2 = sIdx2*slope1;
    if (sIdx2 > b)
      dIdx2 = b * slope1 + (sIdx2 - b) * slope2;
    
    int i1 = int(floor(dIdx1));
    int i2 = int(ceil(dIdx2));
    
    if (i1<=N-1) {
      
      //double alpha = 1.0/(dIdx2 - dIdx1);
      double alpha = 1.0;
      double alpha1 = (1.0 - (dIdx1 - i1)) * alpha;
      double alpha2 = (i2  - dIdx2) * alpha;
      
      for (int j = i1; j <= i2; j++) {
        
        int k = j;
        if (k < 0)
          k = 0;
        if (k >= N)
          break;
        
        double a = alpha;
        if (j == i1)
          a = alpha1;
        if (j == i2)
          a = alpha2;
  
        gsl_vector_set(_vector, k, gsl_vector_get(_vector, k) + a * v);
        gsl_vector_set(_auxV, k, gsl_vector_get(_auxV, k) + a);        
      }      
    }    
  }

  for (unsigned i = 0; i < N; i++) {
    double norm = gsl_vector_get(_auxV, i);
    if (norm > 1E-20)
      gsl_vector_set(_vector, i, gsl_vector_get(_vector, i)/norm);
  }

  return _vector;
}

const gsl_vector* VTLNFeature::next(int frameX) {
  switch(_version) {
    case 1:
      return nextOrg(frameX);
    case 2:
      return nextFF(frameX);
    default:
      printf("[ERROR] VTLNFeature::next >> unknown version number (%d)", _version);
      exit(1);
  }
}

void VTLNFeature::matrix(gsl_matrix* mat) const
{
  if (mat->size1 != size() || mat->size2 != size())
    throw jdimension_error("Matrix (%d x %d) does not match (%d x %d)",
			   mat->size1, mat->size2, size(), size());

  gsl_matrix_set_zero(mat);

  double yedge = (_edge < _ratio) ? (_edge / _ratio)              : 1.0;
  double b     = (yedge < 1.0)    ? (1.0 - _edge) / (1.0 - yedge) : 0;

  for (unsigned coeffX = 0; coeffX < size(); coeffX++) {

    double Y0 = double(coeffX)   / double(size());
    double Y1 = double(coeffX+1) / double(size());

    double X0 = ((Y0 < yedge) ? (_ratio * Y0) :
		 (b     * Y0 +  1.0 - b)) * size();
    double X1 = ((Y1 < yedge) ? (_ratio * Y1) :
		 (b     * Y1 +  1.0 - b)) * size();

    int    Lower_coeffY1 = int(X1);
    double alpha1        = X1 - Lower_coeffY1;

    int    Lower_coeffY0 = int(X0);
    double alpha0        = int(X0) + 1 - X0;

    if (Lower_coeffY0 >= mat->size1)
      Lower_coeffY0 = mat->size1 - 1;

    if (Lower_coeffY1 > mat->size1)
      Lower_coeffY1 = mat->size1;

    if ( Lower_coeffY0 == Lower_coeffY1) {
      gsl_matrix_set(mat, coeffX, Lower_coeffY0, X1-X0);
    } else {
      gsl_matrix_set(mat, coeffX, Lower_coeffY0, alpha0);

      for (int i = Lower_coeffY0+1; i < Lower_coeffY1; i++)
	gsl_matrix_set(mat, coeffX, i, 1.0);

      if ( Lower_coeffY1 < int(size()))
	gsl_matrix_set(mat, coeffX, Lower_coeffY1, alpha1);	
    }
  }
}


// ----- methods for class `MelFeature::_SparseMatrix' -----
//
MelFeature::_SparseMatrix::_SparseMatrix(unsigned m, unsigned n, unsigned version)
  : _data(new float*[m]), _m(m), _n(n), _offset(new unsigned[m]), _coefN(new unsigned[m]), _version(version)
{
  for (unsigned i = 0; i < _m; i++) {
    _data[i]  = NULL;  _offset[i] = _coefN[i] = 0;
  }
}

MelFeature::_SparseMatrix::~_SparseMatrix()
{
  _dealloc();
}

void MelFeature::_SparseMatrix::_alloc(unsigned m, unsigned n)
{
  _dealloc();
  _m = m;  _n = n;

  _data   = new float*[_m];
  _offset = new unsigned[_m];
  _coefN  = new unsigned[_m];

  for (unsigned i = 0; i < _m; i++) {
    _data[i]  = NULL;  _offset[i] = _coefN[i] = 0;
  }
}

void MelFeature::_SparseMatrix::_dealloc()
{
  for (unsigned i = 0; i < _m; i++)
    delete[] _data[i];

  delete[] _data;    _data   = NULL;
  delete[] _offset;  _offset = NULL;
  delete[] _coefN;   _coefN  = NULL;
}

float MelFeature::_SparseMatrix::_mel(float hz)
{
   if (hz>=0) return (float)(2595.0 * log10(1.0 + (double)hz/700.0));
   else return 0.0;
}

float MelFeature::_SparseMatrix::_hertz(float m)
{
   double d = m / 2595.0;
   return (float)(700.0 * (pow(10.0,d) - 1.0));
}

void MelFeature::_SparseMatrix::melScaleOrg(int powN,  float rate, float low, float up, int filterN)
{
  float df = rate / (4.0 * (powN/2));   /* spacing between FFT points in Hz */
  float mlow = _mel(low);
  float mup  = _mel(up);
  float dm   = (mup - mlow)/(filterN+1);    /* delta mel */

  if (low<0.0 || 2.0*up>rate || low>up)
    throw j_error("mel: something wrong with\n");

  /* printf("lower = %fHz (%fmel), upper = %fHz (%fmel)\n",low,mlow,up,mup);*/

  /* -------------------------------------------
     free band matrix, allocate filterN pointer
     ------------------------------------------- */
  if (_data) {
    for (unsigned i = 0; i < _m; i++)
      delete[] _data[i];
    delete[] _data;
  }
  delete[] _offset;  delete[] _coefN;

  _data   = new float*[filterN];
  _coefN  = new unsigned[filterN];
  _offset = new unsigned[filterN];

  /* ---------------------------
     loop over all filters
     --------------------------- */
  for (unsigned x = 0; x < filterN; x++) {

    /* ---- left, center and right edge ---- */
    float left   = _hertz( x     *dm + mlow);
    float center = _hertz((x+1.0)*dm + mlow);
    float right  = _hertz((x+2.0)*dm + mlow);
    /* printf("%3d: left = %fmel, center = %fmel, right = %fmel\n",
       x,x*dm+mlow,(x+1.0)*dm+mlow,(x+2.0)*dm+mlow); */
    /* printf("%3d: left = %fHz, center = %fHz, right = %fHz\n",
       x,left,center,right); */
      
    float height = 2.0 / (right - left);          /* normalized height = 2/width */
    float slope1 = height / (center - left);
    float slope2 = height / (center - right);
    int start    = (int)ceil(left / df);
    int end      = (int)floor(right / df);
      
    _offset[x] = start;
    _coefN[x]  = end - start + 1;
    _n         = end;
    _data[x] = new float[_coefN[x]];
    float freq=start*df;
    for (unsigned i=0; i < _coefN[x]; i++) {
      freq += df;
      if (freq <= center)
	_data[x][i] = slope1*(freq-left);
      else
	_data[x][i] = slope2*(freq-right);
    }
  }
  _rate = rate;
}

void MelFeature::_SparseMatrix::melScaleFF(int powN,  float rate, float low, float up, int filterN)
{
  float df = rate / (4.0 * (powN/2));   /* spacing between FFT points in Hz */
  float mlow = _mel(low);
  float mup  = _mel(up);
  float dm   = (mup - mlow)/(filterN+1);    /* delta mel */

  if (low<0.0 || 2.0*up>rate || low>up)
    throw j_error("mel: something wrong with\n");

  /* printf("lower = %fHz (%fmel), upper = %fHz (%fmel)\n",low,mlow,up,mup);*/

  /* -------------------------------------------
     free band matrix, allocate filterN pointer
     ------------------------------------------- */
  if (_data) {
    for (unsigned i = 0; i < _m; i++)
      delete[] _data[i];
    delete[] _data;
  }
  delete[] _offset;  delete[] _coefN;

  _data   = new float*[filterN];
  _coefN  = new unsigned[filterN];
  _offset = new unsigned[filterN];

  /* ---------------------------
     loop over all filters
     --------------------------- */
  for (unsigned x = 0; x < filterN; x++) {

    /* ---- left, center and right edge ---- */
    float left   = _hertz( x     *dm + mlow);
    float center = _hertz((x+1.0)*dm + mlow);
    float right  = _hertz((x+2.0)*dm + mlow);
    /* printf("%3d: left = %fmel, center = %fmel, right = %fmel\n",
       x,x*dm+mlow,(x+1.0)*dm+mlow,(x+2.0)*dm+mlow); */
    /* printf("%3d: left = %fHz, center = %fHz, right = %fHz\n",
       x,left,center,right); */
      
    float height = 2.0 / (right - left);          /* normalized height = 2/width */
    float slope1 = height / (center - left);
    float slope2 = height / (center - right);
    int start    = (int)ceil(left / df);
    int end      = (int)floor(right / df);
      
    _offset[x] = start;
    _coefN[x]  = end - start + 1;
    _n         = end;
    _data[x] = new float[_coefN[x]];
    float freq=start*df;
    // FF Print MFB --- START
    /*for (unsigned j=0; j<start; j++)
      printf("%f ", 0.0);*/
    // FF Print MFB --- END
    for (unsigned i=0; i < _coefN[x]; i++) {
      //freq += df; // better don't put it here FF fix
      if (freq <= center)
	_data[x][i] = slope1*(freq-left);
      else
	_data[x][i] = slope2*(freq-right);
      // FF Print MFB --- START
      //printf("%f ", _data[x][i]);
      // FF Print MFB --- END
      freq += df; // instead put it here FF fix
    }
    // FF Print MFB --- START
    /*for (unsigned j=end; j<powN; j++)
      printf("%f ", 0.0);
    printf("\r\n");*/
    // FF Print MFB --- END
  }
  _rate = rate;
}

void MelFeature::_SparseMatrix::melScale(int powN,  float rate, float low, float up, int filterN)
{
  switch (_version) {
    case 1:
      melScaleOrg(powN,  rate, low, up, filterN);
    break;
    case 2:
      melScaleFF(powN,  rate, low, up, filterN);
    break;
    default:
      printf("[ERROR] MelFeature::_SparseMatrix::fmatrixBMulot >> Unknown Version.\r\n");
      exit(1);
  }  
}

gsl_vector* MelFeature::_SparseMatrix::fmatrixBMulotOrg( gsl_vector* C, const gsl_vector* A) const
{
  if (C == A)
     throw jconsistency_error("matrix multiplication: result matrix must be different!\n");

  if ( int(A->size) < _n)   // _n can be smaller
     throw jconsistency_error("Matrix columns differ: %d and %d.\n",
            A->size, _n);

  // fmatrixResize(C,A->m,_m);
  for (unsigned j = 0; j < C->size; j++) {
    double  sum = 0.0;
    double*  aP  = A->data + _offset[j];
    double*  eP  = A->data + _offset[j] + _coefN[j] - 3;
    float*   bP  = _data[j];
    while (aP < eP) {
      sum += aP[0]*bP[0] + aP[1]*bP[1] + aP[2]*bP[2] + aP[3]*bP[3];
      aP += 4; bP += 4;
    }
    eP += 3;
    while (aP < eP) sum += *(aP++) * *(bP++);
    gsl_vector_set(C, j, sum);
  }

  return C;
}

gsl_vector* MelFeature::_SparseMatrix::fmatrixBMulotFF( gsl_vector* C, const gsl_vector* A) const
{
  if (C == A)
     throw jconsistency_error("matrix multiplication: result matrix must be different!\n");

  if ( int(A->size) < _n)		// _n can be smaller
     throw jconsistency_error("Matrix columns differ: %d and %d.\n",
			      A->size, _n);

  // fmatrixResize(C,A->m,_m);
  for (unsigned j = 0; j < C->size; j++) {
    double  sum = 0.0;
    double*  aP  = A->data + _offset[j];
    //double*  eP  = A->data + _offset[j] + _coefN[j] - 3; // "are you serious about this" FF fix
    double*  eP  = A->data + _offset[j] + (_coefN[j] & ~3); // "better do this" FF fix
    float*   bP  = _data[j];
    while (aP < eP) {
      sum += aP[0]*bP[0] + aP[1]*bP[1] + aP[2]*bP[2] + aP[3]*bP[3];
      aP += 4; bP += 4;
    }
    //eP += 3; // "are you serious about this" FF fix - part 2
    eP += (_coefN[j] & 3); // "better do this" FF fix - part 2
    while (aP < eP) sum += *(aP++) * *(bP++);
    gsl_vector_set(C, j, sum);
  }

  return C;
}

gsl_vector* MelFeature::_SparseMatrix::fmatrixBMulot( gsl_vector* C, const gsl_vector* A) const {
  
  switch (_version) {
    case 1:
      fmatrixBMulotOrg(C, A);
    break;
    case 2:
      fmatrixBMulotFF(C, A);
    break;
    default:
      printf("[ERROR] MelFeature::_SparseMatrix::fmatrixBMulot >> Unknown Version.\r\n");
      exit(1);
  }
  
}

void MelFeature::_SparseMatrix::readBuffer(const String& fb)
{
  std::list<String> scratch;
  splitList(fb, scratch);

  std::list<String>::iterator sitr = scratch.begin();
  for (unsigned i = 0; i < 2; i++)
    sitr++;

  std::list<String> rows;
  splitList(*sitr, rows);

  unsigned i = 0;
  for (std::list<String>::iterator itr = rows.begin(); itr != rows.end(); itr++) {

    // cout << "All " << *itr  << endl;

    std::list<String> values;
    splitList((*itr), values);

    if (i == 0)
	_alloc(rows.size(), values.size() - 1);

    // --- scan <offset> ---
    std::list<String>::iterator jitr = values.begin();
    if ( values.size() < 1 || sscanf(*jitr, "%d", &(_offset[i])) != 1) {
      _dealloc();
      throw jconsistency_error("expected integer value for <offset> not: %s\n", (*jitr).c_str());
    }

    // --- How many coefficients? Allocate memory! ---
    _coefN[i]      = values.size() - 1;
    unsigned maxN  = _offset[i] + _coefN[i];
    if (_n < maxN) _n = maxN;

    _data[i] = new float[_coefN[i]];

    if (_data[i] == NULL) {
      _dealloc();
      throw jallocation_error("could not allocate float band matrix");
    }

    // --- scan <coef0> <coef1> .. ---
    unsigned j = 0;
    for (jitr++; jitr != values.end(); jitr++) {

      // cout << "Value " << *jitr  << endl;

      float d;
      if ( sscanf((*jitr).c_str(), "%f", &d) != 1) {
	_dealloc();
	throw jconsistency_error("expected 'float' type elements.\n");
      }
      _data[i][j] = d;
      j++;
    }
    i++;
  }
}

void MelFeature::_SparseMatrix::matrix(gsl_matrix* mat) const
{
  if (mat->size1 != _m || mat->size2 != _n)
    throw jdimension_error("Matrix (%d x %d) does not match (%d x %d)",
			   mat->size1, mat->size2, _m, _n);

  gsl_matrix_set_zero(mat);
  for (unsigned i = 0; i < _m; i++)
    for (int j = 0; j < _coefN[i]; j++)
      gsl_matrix_set(mat, i, j + _offset[i], _data[i][j]);
}


// ----- methods for class `MelFeature' -----
//
MelFeature::MelFeature(const VectorFeatureStreamPtr& mag, int powN,
		       float rate, float low, float up,
		       unsigned filterN, unsigned version, const String& nm)
  : VectorFeatureStream(filterN, nm), _nmel(filterN),
    _powN((powN == 0) ? mag->size() : powN),
    _mag(mag), _mel(0, 0, version)
{
  if (up <= 0) up = rate/2.0;
  _mel.melScale(_powN, rate, low, up, _nmel);

  printf("Mel Feature Input Size  = %d\n", _powN);
  printf("Mel Feature Output Size = %d\n", size());
}

MelFeature::~MelFeature()
{
}

void MelFeature::read(const String& fileName)
{
  static size_t n      = 0;
  static char*  buffer = NULL;

  FILE* fp = fileOpen(fileName, "r");
  getline(&buffer, &n, fp);
  fileClose(fileName, fp);

  // cout << "Buffer " << buffer << endl;

  _mel.readBuffer(buffer);
}

const gsl_vector* MelFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector* fftVec = _mag->next(_frameX + 1);
  _increment();
  
  _mel.fmatrixBMulot(_vector, fftVec);

  return _vector;
}


// ----- methods for class `SphinxMelFeature' -----
//
SphinxMelFeature::SphinxMelFeature(const VectorFeatureStreamPtr& mag, unsigned fftN, unsigned powerN,
				   float sampleRate, float lowerF, float upperF,
				   unsigned filterN, const String& nm)
  : VectorFeatureStream(filterN, nm), _fftN(fftN), _filterN(filterN),
    _powerN((powerN == 0) ? mag->size() : powerN), _sampleRate(sampleRate),
    _mag(mag), _filters(gsl_matrix_calloc(_filterN, _powerN))
{
  printf("Sphinx Mel Feature Input Size  = %d\n", _powerN);
  printf("Sphinx Mel Feature Output Size = %d\n", size());

  double dfreq = _sampleRate / _fftN;
  if (upperF > _sampleRate / 2)
    throw j_error("Upper frequency %f exceeds Nyquist %f", upperF, sampleRate / 2.0);

  double melmax = _melFrequency(upperF);
  double melmin = _melFrequency(lowerF);
  double dmelbw = (melmax - melmin) / (_filterN + 1);

  // Filter edges, in Hz
  gsl_vector* edges(gsl_vector_calloc(_filterN + 2));
  for (unsigned n = 0; n < _filterN + 2; n++)
    gsl_vector_set(edges, n, _melInverseFrequency(melmin + dmelbw * n));

  // Set filter triangles, in DFT points
  for (unsigned filterX = 0; filterX < _filterN; filterX++) {
    const double left_freq	= gsl_vector_get(edges, filterX);
    const double center_freq	= gsl_vector_get(edges, filterX + 1);
    const double right_freq	= gsl_vector_get(edges, filterX + 2);

    const double freq_width	= right_freq - left_freq;
    const double freq_height	= 2.0 / freq_width;
    const double left_slope	= freq_height / (center_freq - left_freq);
    const double right_slope	= freq_height / (right_freq - center_freq);
    
    unsigned min_k = 999999;
    unsigned max_k = 0;

    for (unsigned k = 1; k < _powerN; k++) {
      double hz			= k * dfreq;
      if (hz < left_freq) continue;
      if (hz > right_freq) break;
      double left_value		= (hz - left_freq) / (center_freq - left_freq);
      double right_value	= (right_freq - hz) / (right_freq - center_freq);
      double filter_value	=  min(left_value, right_value);
      gsl_matrix_set(_filters, filterX, k, filter_value);
      min_k = min(k, min_k);
      max_k = max(k, max_k);
      _boundaries.push_back(_Boundary(min_k, max_k));
    }
  }
  gsl_vector_free(edges);
}

SphinxMelFeature::~SphinxMelFeature()
{
  gsl_matrix_free(_filters);
}

double SphinxMelFeature::_melFrequency(double frequency)
{
  return (2595.0 * log10(1.0 + (frequency / 700.0)));
}

double SphinxMelFeature::_melInverseFrequency(double mel)
{
  return (700.0 * (pow(10.0, mel / 2595.0) - 1.0));
}


const gsl_vector* SphinxMelFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector* fftVec = _mag->next(_frameX + 1);
  _increment();
  
  gsl_blas_dgemv(CblasNoTrans, 1.0, _filters, fftVec, 0.0, _vector);

  return _vector;
}


// ----- methods for class `LogFeature' -----
//
LogFeature::LogFeature(const VectorFeatureStreamPtr& mel, double m, double a, bool sphinxFlooring,
		       const String& nm) :
  VectorFloatFeatureStream(mel->size(), nm),
  _nmel(mel->size()), _mel(mel), _m(m), _a(a), _SphinxFlooring(sphinxFlooring)
{
  printf("Log Feature Size = %d\n", size());
}

const gsl_vector_float* LogFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector* melVec = _mel->next(_frameX + 1);
  _increment();

  unsigned err = 0;
  for (unsigned i = 0; i < _nmel; i++) {
    double val = gsl_vector_get(melVec, i);
    if (_SphinxFlooring) {

      static const double _floor = 1.0E-05;
      if (val < _floor) {
	val = _floor;  err++; 
      }
    } else {
      val += _a;
      if (val <= 0.0) {
	val = 1.0;  err++; 
      }
    }

    gsl_vector_float_set(_vector, i, _m * log10(val));
  }

  /*
  if (err > 0)
    printf("Warning: %d times the value for log() was <= 0, set result to 0.0\n",
	   err);
  */

  return _vector;
}

// ----- methods for class `CepstralFeature' -----
//
// type:
//   0 = Type 1 DCT ? 
//   1 = Type 2 DCT ? 
//   2 = Sphinx Legacy
CepstralFeature::CepstralFeature(const VectorFloatFeatureStreamPtr& mel,
				 unsigned ncep, int type, const String& nm) :
  VectorFloatFeatureStream(ncep, nm), _ncep(ncep),
  _cos(gsl_matrix_float_calloc(ncep, mel->size())), _mel(mel)
{
  if (type == 0) {
    cout << "Using DCT Type 1." << endl;
    gsl_matrix_float_set_cosine(_cos, ncep, mel->size(), type);
  } else if (type == 1) {
    cout << "Using DCT Type 2." << endl;
    gsl_matrix_float_set_cosine(_cos, ncep, mel->size(), type);
  } else if (type == 2) {
    cout << "Using Sphinx legacy DCT." << endl;
    _sphinxLegacy();
  } else {
    throw jindex_error("Unknown DCT type\n");
  }

  printf("Cepstral Feature Input Size  = %d\n", _mel->size());
  printf("Cepstral Feature Output Size = %d\n", size());
}

void CepstralFeature::_sphinxLegacy()
{
  for (unsigned cepstraX = 0; cepstraX < size(); cepstraX++) {
    double deltaF = M_PI * float(cepstraX) / _mel->size();
    for (unsigned filterX = 0; filterX < _mel->size(); filterX++) {
      double frequency = deltaF * (filterX + 0.5);
      double c	       = cos(frequency) / _mel->size();
      if (filterX == 0) c *= 0.5;
      gsl_matrix_float_set(_cos, cepstraX, filterX, c);
    }
  }
}

const gsl_vector_float* CepstralFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector_float* logVec = _mel->next(_frameX + 1);
  _increment();

  gsl_blas_sgemv(CblasNoTrans, 1.0, _cos, logVec, 0.0, _vector);

  return _vector;
}

gsl_matrix* CepstralFeature::matrix() const
{
  cout << "Allocating DCT Matrix (" << size() << " x " << _mel->size() << ")" << endl;

  gsl_matrix* matrix = gsl_matrix_calloc(size(), _mel->size());

  for(unsigned i=0;i<matrix->size1;i++)
    for(unsigned j=0;j<matrix->size2;j++)
      gsl_matrix_set(matrix, i,j, gsl_matrix_float_get(_cos,i,j) );

  return matrix;
}

// ----- methods for class `FloatToDoubleConversionFeature' -----
//
const gsl_vector* FloatToDoubleConversionFeature::next(int frameX) {
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector_float* srcVec = _src->next(_frameX + 1);
  _increment();

  for (unsigned i=0; i<_vector->size; i++)
    gsl_vector_set(_vector, i, gsl_vector_float_get(srcVec, i));

  return _vector;

}

// ----- methods for class `MeanSubtractionFeature' -----
//
const float     MeanSubtractionFeature::_varianceFloor  = 0.0001;
const float	MeanSubtractionFeature::_beforeWgt      = 0.98;
const float	MeanSubtractionFeature::_afterWgt       = 0.995;
const unsigned	MeanSubtractionFeature::_framesN2Change = 500;

MeanSubtractionFeature::
MeanSubtractionFeature(const VectorFloatFeatureStreamPtr& src, const VectorFloatFeatureStreamPtr& weight,
		       double devNormFactor, bool runon, const String& nm)
  : VectorFloatFeatureStream(src->size(), nm),
    _src(src), _wgt(weight),
    _mean(gsl_vector_float_calloc(size())), _var(gsl_vector_float_calloc(size())),
    _devNormFactor(devNormFactor), _framesN(0), _runon(runon), _meanVarianceFound(false)
{
  gsl_vector_float_set_zero(_mean);
  gsl_vector_float_set_all(_var, 1.0);

  printf("Mean Subtraction Feature Input Size  = %d\n", _src->size());
  printf("Mean Subtraction Feature Output Size = %d\n", size());
}

MeanSubtractionFeature::~MeanSubtractionFeature()
{
  gsl_vector_float_free(_mean);
  gsl_vector_float_free(_var);
}

void MeanSubtractionFeature::write(const String& fileName, bool variance) const
{
  if (_frameX <= 0)
    throw jio_error("Frame count must be > 0.\n", _frameX);

  FILE* fp = fileOpen(fileName, "w");

  const gsl_vector_float* vec = (variance) ? _var : _mean;
  gsl_vector_float_fwrite(fp, vec);

  fileClose(fileName, fp);
}

const gsl_vector_float* MeanSubtractionFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  if (_runon)
    return _nextRunon(frameX);
  else
    return _nextBatch(frameX);
}

const gsl_vector_float* MeanSubtractionFeature::_nextRunon(int frameX)
{
  const gsl_vector_float* srcVec = _src->next(_frameX + 1);
  float 		  wgt    = 1.0;
  if (_wgt.isNull() == false) {
    const gsl_vector_float* wgtVec = _wgt->next(_frameX + 1);
     		            wgt    = gsl_vector_float_get(wgtVec, 0);
  }
  _increment();

  if (wgt > 0.0) {
    // update mean
    float wgt  = (_framesN < _framesN2Change) ? _beforeWgt : _afterWgt;
    for (unsigned i = 0; i < size(); i++) {
      float f    = gsl_vector_float_get(srcVec, i);
      float m    = gsl_vector_float_get(_mean, i);

      float comp = wgt * m + (1.0 - wgt) * f;
      gsl_vector_float_set(_mean, i, comp);
    }

    // update square mean
    if (_devNormFactor > 0.0) {
      for (unsigned i = 0; i < size(); i++) {
	float f    = gsl_vector_float_get(srcVec, i);
	float m    = gsl_vector_float_get(_mean, i);
	float diff = f - m;

	float sm   = gsl_vector_float_get(_var, i);

	float comp = wgt * sm + (1.0 - wgt) * (diff * diff);
	gsl_vector_float_set(_var, i, comp);
      }
    }

    _framesN++;
  }

  _normalize(srcVec);

  return _vector;
}

const gsl_vector_float* MeanSubtractionFeature::_nextBatch(int frameX)
{
  if (_meanVarianceFound == false)
    _calcMeanVariance();

  const gsl_vector_float* srcVec = _src->next(_frameX + 1);
  _increment();

  _normalize(srcVec);

  return _vector;
}

void MeanSubtractionFeature::_calcMeanVariance()
{
  int    frameX = 0;
  double ttlWgt = 0.0;
  gsl_vector_float_set_zero(_mean);
  while (true) {
    try {
      const gsl_vector_float* srcVec = _src->next(frameX);
      float                   wgt    = 1.0;
      if (_wgt.isNull() == false) {
        const gsl_vector_float* wgtVec = _wgt->next(frameX);
        wgt    = gsl_vector_float_get(wgtVec, 0);
      }

      // printf("Frame %4d : Weight %6.2f\n", frameX, wgt);

      // sum for mean
      for (unsigned i = 0; i < size(); i++) {
        float f = gsl_vector_float_get(srcVec, i);
        float m = gsl_vector_float_get(_mean, i);

        gsl_vector_float_set(_mean, i, m + wgt * f);
      }
      frameX++;  ttlWgt += wgt;
    } catch (jiterator_error& e) {
      for (unsigned i = 0; i < size(); i++)
	gsl_vector_float_set(_mean, i, gsl_vector_float_get(_mean, i) / ttlWgt);
      break;
    } catch (j_error& e) {
      if (e.getCode() == JITERATOR) {
        for (unsigned i = 0; i < size(); i++)
          gsl_vector_float_set(_mean, i, gsl_vector_float_get(_mean, i) / ttlWgt);
        break;
      }
    }
  }

  frameX = 0;  ttlWgt = 0.0;
  gsl_vector_float_set_zero(_var);
  while (true) {
    try {
      const gsl_vector_float* srcVec = _src->next(frameX);
      float                   wgt    = 1.0;
      if (_wgt.isNull() == false) {
        const gsl_vector_float* wgtVec = _wgt->next(frameX);
        wgt    = gsl_vector_float_get(wgtVec, 0);
      }

      // sum for covariance
      for (unsigned i = 0; i < size(); i++) {
        float f = gsl_vector_float_get(srcVec, i);
        float v = gsl_vector_float_get(_var, i);

        gsl_vector_float_set(_var, i, v + wgt * f * f);
      }
      frameX++;  ttlWgt += wgt;
    } catch (jiterator_error& e) {
      for (unsigned i = 0; i < size(); i++) {
        float m = gsl_vector_float_get(_mean, i);
        gsl_vector_float_set(_var, i, (gsl_vector_float_get(_var, i) / ttlWgt) - (m * m));
      }
      break;
    } catch (j_error& e) {
      if (e.getCode() == JITERATOR) {
        for (unsigned i = 0; i < size(); i++) {
          float m = gsl_vector_float_get(_mean, i);
          gsl_vector_float_set(_var, i, (gsl_vector_float_get(_var, i) / ttlWgt) - (m * m));
        }
        break;
      }

    }
  }
  _meanVarianceFound = true;
}

void MeanSubtractionFeature::_normalize(const gsl_vector_float* srcVec)
{
  // subtract mean
  for (unsigned i = 0; i < size(); i++) {
    float f     = gsl_vector_float_get(srcVec, i);
    float m     = gsl_vector_float_get(_mean, i);

    gsl_vector_float_set(_vector, i, f - m);
  }

  // normalize standard deviation
  if (_devNormFactor > 0.0) {
    for (unsigned i = 0; i < size(); i++) {
      float f   = gsl_vector_float_get(_vector, i);
      float var = gsl_vector_float_get(_var, i);

      if (var < _varianceFloor) var = _varianceFloor;

      gsl_vector_float_set(_vector, i, f / (_devNormFactor * sqrt(var)));
    }
  }
}

void MeanSubtractionFeature::reset()
{
  _src->reset();  VectorFloatFeatureStream::reset();  _meanVarianceFound = false;

  if (_wgt.isNull() == false) _wgt->reset();
}

void MeanSubtractionFeature::nextSpeaker()
{
  _framesN = 0;
  gsl_vector_float_set_zero(_mean);
  gsl_vector_float_set_all(_var, 1.0);
}


// ----- methods for class `FileMeanSubtractionFeature' -----
//
const float FileMeanSubtractionFeature::_varianceFloor = 0.0001;

FileMeanSubtractionFeature::
FileMeanSubtractionFeature(const VectorFloatFeatureStreamPtr& src, double devNormFactor, const String& nm)
  : VectorFloatFeatureStream(src->size(), nm),
    _src(src), _mean(gsl_vector_float_calloc(size())), _variance(NULL), _devNormFactor(devNormFactor)
{
  gsl_vector_float_set_zero(_mean);

  printf("File Mean Subtraction Feature Input Size  = %d\n", _src->size());
  printf("File Mean Subtraction Feature Output Size = %d\n", size());
}

FileMeanSubtractionFeature::~FileMeanSubtractionFeature()
{
  gsl_vector_float_free(_mean);
  if (_variance != NULL)
    gsl_vector_float_free(_variance);
}

const gsl_vector_float* FileMeanSubtractionFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector_float* srcVec = _src->next(_frameX + 1);
  _increment();

  // subtract mean
  for (unsigned i = 0; i < size(); i++) {
    float f = gsl_vector_float_get(srcVec, i);
    float m = gsl_vector_float_get(_mean, i);

    float comp = f - m;
    gsl_vector_float_set(_vector, i, comp);
  }

  // normalize standard deviation
  if (_variance != NULL && _devNormFactor > 0.0) {
    for (unsigned i = 0; i < size(); i++) {
      float f   = gsl_vector_float_get(_vector, i);
      float var = gsl_vector_float_get(_variance, i);

      if (var < _varianceFloor) var = _varianceFloor;

      gsl_vector_float_set(_vector, i, f / (_devNormFactor * sqrt(var)));
    }
  }

  return _vector;
}

void FileMeanSubtractionFeature::reset()
{
  _src->reset();
  VectorFloatFeatureStream::reset();
}

void FileMeanSubtractionFeature::read(const String& fileName, bool variance)
{
  FILE* fp = fileOpen(fileName, "r");
  if (variance == false) {

    printf("Loading mean from '%s'.\n", fileName.c_str());
    if (_mean->size != int(size())) {
      fileClose(fileName, fp);
      throw jdimension_error("Feature and mean do not have same size (%d vs. %d).\n", _mean->size, size());
    }
    gsl_vector_float_fread(fp, _mean);

  } else {

    if (_variance == NULL)
      _variance = gsl_vector_float_calloc(size());
    printf("Loading covariance from '%s'.\n", fileName.c_str());
    if (_variance->size != int(size())) {
      fileClose(fileName, fp);
      throw jdimension_error("Feature and covariance do not have same size (%d vs. %d).\n", _variance->size, size());
    }
    gsl_vector_float_fread(fp, _variance);

  }
  fileClose(fileName, fp);
}


// ----- methods for class `AdjacentFeature' -----
//
AdjacentFeature::
AdjacentFeature(const VectorFloatFeatureStreamPtr& single, unsigned delta,
		const String& nm) :
  VectorFloatFeatureStream((2 * delta + 1) * single->size(), nm), _delta(delta),
  _single(single), _singleSize(single->size()),
  _plen(2 * _delta * _singleSize), _framesPadded(0)
{
  printf("Adjacent Feature Input Size  = %d\n", _single->size());
  printf("Adjacent Feature Output Size = %d\n", size());  fflush(stdout);
}

void AdjacentFeature::_bufferNextFrame(int frameX)
{
  if (frameX == 0) {				// initialize the buffer
    const gsl_vector_float* singVec = _single->next(/* featX = */ 0);
    for (unsigned featX = 1; featX <= _delta + 1; featX++) {
      unsigned offset = featX * _singleSize;
      for (unsigned sampX = 0; sampX < _singleSize; sampX++)
	gsl_vector_float_set(_vector, offset + sampX, gsl_vector_float_get(singVec, sampX));
    }

    for (unsigned featX = 1; featX < _delta; featX++) {
      singVec = _single->next(featX);
      unsigned offset = (featX + _delta + 1) * _singleSize;
      for (unsigned sampX = 0; sampX < _singleSize; sampX++)
	gsl_vector_float_set(_vector, offset + sampX, gsl_vector_float_get(singVec, sampX));
    }
  }

  // slide down the old values
  for (unsigned sampX = 0; sampX < _plen; sampX++)
    gsl_vector_float_set(_vector, sampX, gsl_vector_float_get(_vector, _singleSize + sampX));

  if (_framesPadded == 0) {			// normal processing
    try {

      const gsl_vector_float* singVec = _single->next(frameX + _delta);
      unsigned offset = 2 * _delta * _singleSize;
      for (unsigned sampX = 0; sampX < _singleSize; sampX++)
	gsl_vector_float_set(_vector, offset + sampX, gsl_vector_float_get(singVec, sampX));

    } catch (jiterator_error& e) {

      for (unsigned sampX = 0; sampX < _singleSize; sampX++)
	gsl_vector_float_set(_vector, _plen + sampX, gsl_vector_float_get(_vector, (_plen - _singleSize) + sampX));
      _framesPadded++;

    } catch (j_error& e) {

      if (e.getCode() != JITERATOR) {
	cout << e.what() << endl;
	throw;
      }
      for (unsigned sampX = 0; sampX < _singleSize; sampX++)
	gsl_vector_float_set(_vector, _plen + sampX, gsl_vector_float_get(_vector, (_plen - _singleSize) + sampX));
      _framesPadded++;

    }
  } else if (_framesPadded < _delta) {		// pad with zeros
    for (unsigned sampX = 0; sampX < _singleSize; sampX++)
      gsl_vector_float_set(_vector, _plen + sampX, gsl_vector_float_get(_vector, (_plen - _singleSize) + sampX));
    _framesPadded++;
  } else {					// end of utterance
    throw jiterator_error("end of samples (FilterFeature)!");
  }
}

const gsl_vector_float* AdjacentFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  _bufferNextFrame(_frameX + 1);
  _increment();

  return _vector;
}

void AdjacentFeature::reset()
{
  _single->reset();

  _framesPadded = 0;
  gsl_vector_float_set_zero(_vector);

  VectorFloatFeatureStream::reset();
}


// ----- methods for class `LinearTransformFeature' -----
//
LinearTransformFeature::
LinearTransformFeature(const VectorFloatFeatureStreamPtr& src, unsigned sz, const String& nm) :
  // VectorFloatFeatureStream((sz == 0) ? mat->size1 : sz, nm),
  VectorFloatFeatureStream(sz, nm),
  _src(src),
  _trans(gsl_matrix_float_calloc(size(), _src->size()))
{
  cout << "Linear Transformation Feature Input Size  = " << _src->size() << endl;
  cout << "Linear Transformation Feature Output Size = " << size()       << endl;
}

const gsl_vector_float* LinearTransformFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector_float* logVec = _src->next(_frameX + 1);
  _increment();
  
  gsl_blas_sgemv(CblasNoTrans, 1.0, _trans, logVec, 0.0, _vector);

  return _vector;
}

gsl_matrix_float* LinearTransformFeature::matrix() const
{
#if 0
  cout << "Allocating Transformation Matrix (" << size() << " x " << _src->size() << ")" << endl;

  gsl_matrix_float* matrix = gsl_matrix_float_calloc(size(), _src->size());

  gsl_matrix_float_memcpy(matrix, _trans);

  return matrix;
#else
  return _trans;
#endif
}

void LinearTransformFeature::load(const String& fileName, bool old)
{
  gsl_matrix_float_load(_trans, fileName, old);
  _trans = gsl_matrix_float_resize(_trans, size(), _src->size());
}

void LinearTransformFeature::identity()
{
  if (size() != _src->size())
    throw jdimension_error("Cannot set an (%d x %d) matrix to identity.", size(), _src->size());

  gsl_matrix_float_set_zero(_trans);
  for (unsigned i = 0; i < size(); i++)
    gsl_matrix_float_set(_trans, i, i, 1.0);
}


// ----- methods for class `StorageFeature' -----
//
StorageFeature::StorageFeature(const VectorFloatFeatureStreamPtr& src,
			       const String& nm) :
  VectorFloatFeatureStream(src->size(), nm), _src(src), _frames(MaxFrames)
{
  for (int i = 0; i < MaxFrames; i++)
    _frames[i] = gsl_vector_float_calloc(size());
}

StorageFeature::~StorageFeature()
{
  for (int i = 0; i < MaxFrames; i++)
    gsl_vector_float_free(_frames[i]);
}

const int StorageFeature::MaxFrames = 100000;
const gsl_vector_float* StorageFeature::next(int frameX)
{
  if (frameX >= 0 && frameX <= _frameX) return _frames[frameX];

  if (frameX >= MaxFrames)
    throw jdimension_error("Frame %d is greater than maximum number %d.",
			   frameX, MaxFrames);

  const gsl_vector_float* singVec = _src->next(_frameX + 1);
  _increment();

  // printf("Storing frame %d in %s\n", frameX, name().c_str());  fflush(stdout);

  gsl_vector_float_memcpy(_frames[_frameX], singVec);

  return _frames[_frameX];
}

void StorageFeature::write(const String& fileName, bool plainText) const
{
  if (_frameX <= 0)
    throw jio_error("Frame count must be > 0.\n", _frameX);

  FILE* fp = fileOpen(fileName, "w");
  if (plainText) {
    int sz = _frames[0]->size;
    fprintf(fp, "%d %d\n", _frameX, sz);

    for (int i = 0; i <= _frameX; i++) {
      for (int j = 0; j < sz; j++) {
	fprintf(fp, "%g", gsl_vector_float_get(_frames[i], j));
	if (j < sz - 1)
	  fprintf(fp, " ");
      }
      fprintf(fp, "\n");
    }

  } else {    
    write_int(fp, _frameX);
    write_int(fp, _frames[0]->size);

    for (int i = 0; i <= _frameX; i++)
      gsl_vector_float_fwrite(fp, _frames[i]);
  }
  fileClose(fileName, fp);
}

void StorageFeature::read(const String& fileName)
{
  FILE* fp = fileOpen(fileName, "r");
  _frameX = read_int(fp);
  int sz  = read_int(fp);

  if (sz != int(_frames[0]->size))
    throw jdimension_error("Feature dimensions (%d vs. %d) do not match.\n",
			   sz, _frames[0]->size);

  for (int i = 0; i < _frameX; i++)
    gsl_vector_float_fread(fp, _frames[i]);
  fileClose(fileName, fp);
}

int StorageFeature::evaluate()
{
  reset();

  int frameX = 0;
  try {
    while(true)
      next(frameX++);
   } catch (jiterator_error& e) {
   } catch (j_error& e) {
     if (e.getCode() != JITERATOR) {
       throw;
     }
   }

  return _frameX;
}

// ----- methods for class `StaticStorageFeature' -----
//
const int StaticStorageFeature::MaxFrames = 10000;

StaticStorageFeature::StaticStorageFeature(unsigned dim, const String& nm) :
  VectorFloatFeatureStream(dim, nm), _frames(MaxFrames)
{
  for (int i = 0; i < MaxFrames; i++)
    _frames[i] = gsl_vector_float_alloc(size());
}


StaticStorageFeature::~StaticStorageFeature()
{
  for (int i = 0; i < MaxFrames; i++)
    gsl_vector_float_free(_frames[i]);
}


const gsl_vector_float* StaticStorageFeature::next(int frameX)
{
  if (frameX >= 0 && frameX <= _frameX)
    return _frames[frameX];

  _increment();

  if (_frameX >= _nFrames) /* if (frameX >= _nFrames) */
    throw jiterator_error("end of samples!");

  return _frames[_frameX];
}


void StaticStorageFeature::read(const String& fileName)
{
  FILE* fp = fileOpen(fileName, "r");
  _nFrames = read_int(fp);
  int sz  = read_int(fp);

  if (sz != int(_frames[0]->size))
    throw jdimension_error("Feature dimensions (%d vs. %d) do not match.\n",
         sz, _frames[0]->size);

  for (int i = 0; i < _nFrames; i++)
    gsl_vector_float_fread(fp, _frames[i]);
  fileClose(fileName, fp);

  //printf("StaticStorageFeature: read %i features\n", _nFrames);
  
}


int StaticStorageFeature::evaluate()
{
  return _nFrames;
  //return _frameX;
}


// ----- methods for class `CircularStorageFeature' -----
//
CircularStorageFeature::CircularStorageFeature(const VectorFloatFeatureStreamPtr& src, unsigned framesN,
					       const String& nm) :
  VectorFloatFeatureStream(src->size(), nm), _src(src), _framesN(framesN), _frames(_framesN), _pointerX(_framesN-1)
{
  for (unsigned i = 0; i < _framesN; i++) {
    _frames[i] = gsl_vector_float_calloc(size());
    gsl_vector_float_set_zero(_frames[i]);
  }
}

CircularStorageFeature::~CircularStorageFeature()
{
  for (int i = 0; i < _framesN; i++)
    gsl_vector_float_free(_frames[i]);
}

unsigned CircularStorageFeature::_getIndex(int frameX) const
{
  int diff = _frameX - frameX;
  if (diff >= _framesN)
    throw jconsistency_error("Difference (%d) is not in range [%d, %d)\n", frameX, 0, _framesN);

  return (_pointerX + _framesN - diff) % _framesN;
}

const gsl_vector_float* CircularStorageFeature::next(int frameX)
{
  /*
  printf("FrameX = %d\n", frameX);
  */

  if (frameX >= 0 && frameX <= _frameX) return _frames[_getIndex(frameX)];

  if (frameX >= 0 && frameX != _frameX + 1)
    throw jconsistency_error("Requested frame %d\n", frameX);

  const gsl_vector_float* block = _src->next(_frameX + 1);
  _increment();
  _pointerX = (_pointerX + 1) % _framesN;
  gsl_vector_float_memcpy(_frames[_pointerX], block);

  /*
  printf("Frame %d:\n", _frameX);
  for (unsigned i = 0; i < size(); i++)
    printf("    %f\n", gsl_vector_float_get(_frames[_pointerX], i));
  */

  // printf("Circularly storing frame %d in %s\n", frameX, name().c_str());  fflush(stdout);

  return _frames[_pointerX];
}

void CircularStorageFeature::reset()
{
  _pointerX = _framesN - 1;  _src->reset();  VectorFloatFeatureStream::reset();
}


// ----- methods for class `FilterFeature' -----
//
FilterFeature::
FilterFeature(const VectorFloatFeatureStreamPtr& src, gsl_vector* coeffA,
	      const String& nm)
  : VectorFloatFeatureStream(src->size(), nm),
    _src(src),
    _lenA(coeffA->size),
    _coeffA(gsl_vector_calloc(_lenA)),
    _offset(int((_lenA - 1) / 2)),
    _buffer(size(), _lenA),
    _framesPadded(0)
{
  if (_lenA % 2 != 1)
    throw jdimension_error("Length of filter (%d) is not odd.", _lenA);

  gsl_vector_memcpy(_coeffA, coeffA);
}

FilterFeature::~FilterFeature()
{
  gsl_vector_free(_coeffA);
}

void FilterFeature::_bufferNextFrame(int frameX)
{
  if (frameX == 0) {				// initialize the buffer

    for (int i = 0; i < _offset; i++) {
      const gsl_vector_float* srcVec = _src->next(i);
      _buffer.nextSample(srcVec);
    }

  }

  if (_framesPadded == 0) {			// normal processing

    try {
      const gsl_vector_float* srcVec = _src->next(frameX + _offset);
      _buffer.nextSample(srcVec);
    } catch (jiterator_error& e) {
      _buffer.nextSample();
      _framesPadded++;
    } catch (j_error& e) {
      if (e.getCode() != JITERATOR) {
	cout << e.what() << endl;
	throw;
      }
      _buffer.nextSample();
      _framesPadded++;
    }


  } else if (_framesPadded < _offset) {		// pad with zeros

    _buffer.nextSample();
    _framesPadded++;

  } else {					// end of utterance

    throw jiterator_error("end of samples (FilterFeature)!");

  }
}

const gsl_vector_float* FilterFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  _bufferNextFrame(_frameX + 1);
  _increment();

  /*
  printf("Feature %s:\n", name().c_str());
  _buffer.print();
  */

  for (unsigned coeffX = 0; coeffX < size(); coeffX++) {
    double sum = 0.0;

    for (int i = -_offset; i <= _offset; i++) {

      /*
      printf("  %2d : coeffA = %10.4f : sample = %10.4f\n",
	     coeffX, gsl_vector_get(_coeffA, i + _offset), _buffer.sample(-i, coeffX));
      */

      sum += gsl_vector_get(_coeffA, i + _offset) * _buffer.sample(-i, coeffX);
    }

    gsl_vector_float_set(_vector, coeffX, sum);
  }

  return _vector;
}

void FilterFeature::reset()
{
  _src->reset();

  _framesPadded = 0;
  _buffer.zero();

  VectorFloatFeatureStream::reset();
}


// ----- methods for class `MergeFeature' -----
//
MergeFeature::
MergeFeature(VectorFloatFeatureStreamPtr& stat, VectorFloatFeatureStreamPtr& delta,
	     VectorFloatFeatureStreamPtr& deltaDelta, const String& nm)
  : VectorFloatFeatureStream(stat->size() + delta->size() + deltaDelta->size(), nm)
{
  _flist.push_back(stat);  _flist.push_back(delta);  _flist.push_back(deltaDelta);
}

const gsl_vector_float* MergeFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  unsigned dimX = 0;
  for (_FeatureListIterator itr = _flist.begin(); itr != _flist.end(); itr++) {
    const gsl_vector_float* singVec = (*itr)->next(_frameX + 1);
    for (unsigned i = 0; i < singVec->size; i++)
      gsl_vector_float_set(_vector, dimX++, gsl_vector_float_get(singVec, i));
  }
  _increment();

  return _vector;
}

void MergeFeature::reset()
{
  for (_FeatureListIterator itr = _flist.begin(); itr != _flist.end(); itr++)
    (*itr)->reset();

  VectorFloatFeatureStream::reset();
}

// ----- methods for class `MultiModalFeature' -----
 //
MultiModalFeature::MultiModalFeature( unsigned nModality, unsigned totalVecSize, const String& nm )
  : VectorFloatFeatureStream( totalVecSize, nm ),
    _nModality(nModality),_currVecSize(0)
{
  _samplePeriods = new unsigned[nModality];
  _minSamplePeriod = -1;
}

MultiModalFeature::~MultiModalFeature()
{
  delete [] _samplePeriods;
}

const gsl_vector_float* MultiModalFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;
  
  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);
  
  unsigned modalX = 0;
  unsigned dimX = 0;
  unsigned currFrameX = _frameX + 1;
  for (_FeatureListIterator itr = _flist.begin(); itr != _flist.end(); itr++,modalX++) {
    if( ( currFrameX % (_samplePeriods[modalX]/_minSamplePeriod) ) == 0 || currFrameX==0 ){// update the feature vector
      const gsl_vector_float* singVec = (*itr)->next(currFrameX);
      for (unsigned i = 0; i < (*itr)->size(); i++)
	gsl_vector_float_set(_vector, dimX++, gsl_vector_float_get(singVec, i));
    }
    else{
      dimX += (*itr)->size();
    }
  }
  _increment();
  
  return _vector;
}

void MultiModalFeature::reset()
{
  for (_FeatureListIterator itr = _flist.begin(); itr != _flist.end(); itr++)
    (*itr)->reset();
  
  VectorFloatFeatureStream::reset();
}

void MultiModalFeature::addModalFeature( VectorFloatFeatureStreamPtr &feature, unsigned samplePeriodinNanoSec )
{
  if( _flist.size() == _nModality ){
    fprintf(stderr,"The number of the features exceeds %d\n",_nModality);
    throw jdimension_error("The number of the modal features exceeds %d\n",_nModality);
  }
  _flist.push_back(feature);
  _samplePeriods[_flist.size()-1] = samplePeriodinNanoSec;
  if( samplePeriodinNanoSec < _minSamplePeriod ){
    _minSamplePeriod = samplePeriodinNanoSec;
  }
  _currVecSize += feature->size();
  if( size() < _currVecSize ){
    fprintf(stderr,"The total vector size exceeds %d\n",size());
    throw jdimension_error("The total vector size exceeds %d\n",size());
  }
}

void writeGSLMatrix(const String& fileName, const gsl_matrix* mat)
{
  FILE* fp = fileOpen(fileName, "w");

  fprintf(fp, "%d ", mat->size1);
  fprintf(fp, "%d ", mat->size2);

  int ret = gsl_matrix_fwrite (fp, mat);
  if (ret != 0)
    throw jio_error("Could not write data to file %s.\n", fileName.c_str());
  fileClose(fileName, fp);
}

#ifdef JACK
// ----- methods for class `Jack' -----
//
Jack::Jack(const String& nm)
{
  can_capture = false;
  can_process = false;
  if ((client = jack_client_new (nm.c_str())) == 0)
    throw jio_error("Jack server not running?");
  jack_set_process_callback (client, _process_callback, this);
  jack_on_shutdown (client, _shutdown_callback, this);
  if (jack_activate (client)) {
    throw jio_error("cannot activate client");
  }
}

Jack::~Jack()
{
  jack_client_close(client);
  for (unsigned i = 0; i < channel.size(); i++) {
    jack_ringbuffer_free(channel[i]->buffer);
    delete channel[i];
  }
}

jack_channel_t* Jack::addPort(unsigned buffersize, const String& connection, const String& nm)
{
  jack_channel_t* ch = new (jack_channel_t);
  ch->buffersize = buffersize;
  if ((ch->port = jack_port_register (client, nm.c_str(), JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0)) == 0) {
    delete ch;
    ch = NULL;
    throw jio_error ("cannot register input port \"%s\"!", nm.c_str());
  }
  if (jack_connect (client, connection.c_str(), jack_port_name (ch->port))) {
    delete ch;
    ch = NULL;
    throw jio_error ("cannot connect input port %s to %s\n", nm.c_str(), connection.c_str());
  } 
  
  if (ch) {
    ch->buffer = jack_ringbuffer_create (sizeof(jack_default_audio_sample_t) * buffersize);
    ch->can_process = true;
    channel.push_back(ch);
  }
  can_process = true;		/* process() can start, now */

  return ch;
}

void Jack::shutdown_callback(void)
{
  throw j_error("JACK shutdown");
}

int Jack::process_callback(jack_nframes_t nframes)
{
	size_t bytes;
	jack_default_audio_sample_t *in;

	/* Do nothing until we're ready to begin. */
	if ((!can_process) || (!can_capture))
		return 0;

	for (unsigned i = 0; i < channel.size(); i++)
	  if (channel[i]->can_process) {
	    in = (jack_default_audio_sample_t *) jack_port_get_buffer (channel[i]->port, nframes);
	    bytes = jack_ringbuffer_write (channel[i]->buffer, (char *) in, sizeof(jack_default_audio_sample_t) * nframes);
	  }

	return 0;
}


// ----- methods for class `JackFeature' -----
//
JackFeature::JackFeature(JackPtr& jack, unsigned blockLen, unsigned buffersize,
			 const String& connection, const String& nm) :
  VectorFloatFeatureStream(blockLen, nm), _jack(jack)

{
  channel = jack->addPort(buffersize, connection, nm);
}

const gsl_vector_float* JackFeature::next(int frameX)
{
  unsigned i = 0;
  jack_default_audio_sample_t frame = 0;
  unsigned s = sizeof(jack_default_audio_sample_t)*size();

  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  while ((jack_ringbuffer_read_space(channel->buffer) < s) && (i < 10000)) { i++; usleep(0); }
  if (i >= 10000)
    throw jio_error("safe-loop overrun!");

  for (unsigned i = 0; i < size(); i++) {
    jack_ringbuffer_read (channel->buffer, (char*) &frame, sizeof(frame));
    gsl_vector_float_set(_vector, i, frame);
  }

  _increment();
  return _vector;
}

#endif


// ----- methods for class `ZeroCrossingHammingFeature' -----
// ----- calculates the Zero Crossing Rate with a Hamming window as weighting function
//
ZeroCrossingRateHammingFeature::ZeroCrossingRateHammingFeature(const VectorFloatFeatureStreamPtr& samp, const String& nm)
  : VectorFloatFeatureStream(1, nm), _samp(samp), _windowLen(samp->size()),
    _window(new double[_windowLen])
{
  double temp = 2. * M_PI / (double)(_windowLen - 1);
  for ( unsigned i = 0 ; i < _windowLen; i++ )
    _window[i] = 0.54 - 0.46*cos(temp*i);

  printf("Zero Crossing Rate Hamming Feature Input Size  = %d\n", _samp->size());
  printf("Zero Crossing Rate Hamming Feature Output Size = %d\n", size());
}

const gsl_vector_float* ZeroCrossingRateHammingFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector_float* block = _samp->next(_frameX + 1);
  _increment();

  float sum = 0;
  for (unsigned i = 0; i < _windowLen - 1; i++) {
    int s_n = gsl_vector_float_get(block, i + 1) >= 0 ? 1 : -1;
    int s = gsl_vector_float_get(block, i) >= 0 ? 1 : -1;
    sum += abs(s_n - s) / 2 * _window[i];
  }
  sum /= _windowLen;
  gsl_vector_float_set(_vector, 0, sum);

  return _vector;
}


// ----- methods for class `YINPitchFeature' -----
// ----- source code adapted from aubio library ----
// ----- according to de Cheveigne and Kawahara "YIN, a fundamental frequency estimator for speech and music"
//
YINPitchFeature::YINPitchFeature(const VectorFloatFeatureStreamPtr& samp, unsigned samplerate, float threshold, const String& nm)
  : VectorFloatFeatureStream(1, nm), _samp(samp), _sr(samplerate), _tr(threshold) { }

float YINPitchFeature::_getPitch(const gsl_vector_float *input, gsl_vector_float *yin, float tol)
{ 
  unsigned int c = 0, j,  tau = 0;
  float tmp = 0., tmp2 = 0.;

  gsl_vector_float_set(yin, 0, 1.);
  for (tau = 1; tau < yin->size; tau++) {
    gsl_vector_float_set(yin, tau, 0.);	
    for (j = 0; j < yin->size; j++) {
      tmp = gsl_vector_float_get(input, j) - gsl_vector_float_get(input, j + tau);			
      gsl_vector_float_set(yin, tau, gsl_vector_float_get(yin, tau) + tmp * tmp);			
    }
    tmp2 += gsl_vector_float_get(yin, tau);				
    gsl_vector_float_set(yin, tau, gsl_vector_float_get(yin, tau) * tau/tmp2);

    if((gsl_vector_float_get(yin, tau) < tol) && 
       (gsl_vector_float_get(yin, tau-1) < gsl_vector_float_get(yin,tau))) {
      return tau-1;
    }
  }
  return 0;	
}

const gsl_vector_float* YINPitchFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector_float* input = _samp->next(_frameX + 1);
  _increment();
  gsl_vector_float* yin = gsl_vector_float_calloc(input->size / 2);

  float pitch =  _getPitch(input, yin, _tr);

  if (pitch>0) {
    pitch = _sr/(pitch+0.);
  } else {
     pitch = 0.;
  }

  gsl_vector_float_set(_vector, 0, pitch);

  gsl_vector_float_free(yin);

  return _vector;
}


// ----- methods for class `SpikeFilter' -----
//
SpikeFilter::SpikeFilter(VectorFloatFeatureStreamPtr& src, unsigned tapN, const String& nm)
  : VectorFloatFeatureStream(src->size(), nm), _src(src),
    _adcN(src->size()), _queueN((tapN-1)>>1), _queue(new float[_queueN]), _windowN(tapN),  _window(new float[_windowN])
{
  if (tapN < 3)
    throw jdimension_error("tapN should be at least 3.");

  if (_adcN < tapN)
    throw jdimension_error("Cannot filter with adcN = %d and tapN = %d.", _adcN, tapN);
}

SpikeFilter::~SpikeFilter()
{
  delete[] _queue;
  delete[] _window;
}

const gsl_vector_float* SpikeFilter::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector_float* adc = _src->next(_frameX + 1);
  _increment();

  // fill queue with initial values
  for (unsigned queueX = 0; queueX < _queueN; queueX++)
    _queue[queueX] = gsl_vector_float_get(adc, queueX);
  unsigned queuePnt = 0;

  // move filter window over the waveform array
  for (unsigned adcX = _queueN; adcX < _adcN - _queueN; adcX++) {

    // copy samples into filter window and sort them
    for (unsigned windowX = 0; windowX < _windowN; windowX++) {
      _window[windowX] = gsl_vector_float_get(adc, adcX + windowX - _queueN);
      int i = windowX;
      int j = windowX-1;
      while ((j >= 0) && (_window[j] > _window[i])) {
        float swappy = _window[i];
        _window[i]   = _window[j];
        _window[j]   = swappy;
        i = j--;
      }
    }

    // take oldest sample out of the queue
    gsl_vector_float_set(_vector, adcX - _queueN, _queue[queuePnt]);

    // pick median and copy it into the queue
    _queue[queuePnt++] = _window[_queueN];
    queuePnt %= _queueN;
  }

  return _vector;
}


// ----- methods for class `SpikeFilter2' -----
//
SpikeFilter2::SpikeFilter2(VectorFloatFeatureStreamPtr& src, 
			   unsigned width, float maxslope, float startslope, float thresh, float alpha, unsigned verbose, const String& nm)
  : VectorFloatFeatureStream(src->size(), nm), _src(src),
    _adcN(src->size()), _width(width), _maxslope(maxslope), _startslope(startslope), _thresh(thresh), _alpha(alpha), _beta(1.0 - alpha), _verbose(verbose) { }

void SpikeFilter2::reset()
{
  _src->reset();  VectorFloatFeatureStream::reset();  _meanslope = _startslope;  _count = 0;
}

const gsl_vector_float* SpikeFilter2::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector_float* adc = _src->next(_frameX + 1);
  _increment();

  for (unsigned i = 0; i < _adcN; i++)
    gsl_vector_float_set(_vector, i, gsl_vector_float_get(adc, i));

  unsigned adcP      = 0;
  unsigned adcQ      = 1;
  while (adcQ < _adcN) {

    // check for very high slopes in the signal
    float slope = gsl_vector_float_get(_vector, adcQ) - gsl_vector_float_get(_vector, adcP);
    int signB, signE, spikeE;
    if (slope < 0.0) {
      slope *= -1;
      signB = -1;
    }
    else signB = 1;
    adcP = adcQ++;
    float max  = _thresh * _meanslope;

    // check for spike
    if (slope > max && slope > _maxslope) {
      float oslope = slope;

      // determine width of actual spike
      unsigned spikeB = adcP-1;
      unsigned spikeN = 0;
      while ((adcQ < _adcN) && (spikeN < _width)) {
        slope = gsl_vector_float_get(_vector, adcQ) - gsl_vector_float_get(_vector, adcP);
        if (slope < 0) {
	  slope = -1*slope;
	  signE = -1;
	}
	else signE = 1;
        adcP = adcQ++;
        spikeN++;
        if (signB != signE && slope > max && slope > _maxslope) break;
      }
      spikeE = adcP;

      // filter out spike by linear interpolation
      for (int spikeX = spikeB+1; spikeX < spikeE; spikeX++) {
        float lambda = ((float) (spikeX - spikeB)) / (spikeE - spikeB);
        gsl_vector_float_set(_vector, spikeX, (1.0 - lambda) *  gsl_vector_float_get(_vector, spikeB) + lambda * gsl_vector_float_get(_vector, spikeE));
      }
      _count++;
      if (_verbose > 1) printf("spike %d at %d..%d, slope = %d, max = %d\n",
			       _count, spikeB+1, spikeE-1, oslope, max);

    }
    else {
      _meanslope = _beta * _meanslope + _alpha * slope;
    }
  }
  if (_verbose > 0 && _count > 0) printf("%d spikes removed\n", _count);
  
  return _vector;
}


namespace sndfile {
// ----- methods for class `SoundFile' -----
//
SoundFile::SoundFile(const String& fn,
		     int mode,
		     int format,
		     int samplerate,
		     int channels,
		     bool normalize)
{
   memset(&_sfinfo, 0, sizeof(_sfinfo));
  _sfinfo.channels   = channels;
  _sfinfo.samplerate = samplerate;
  _sfinfo.format     = format;
  _sndfile = sf_open(fn.c_str(), mode, &_sfinfo);
  cout << "Reading sound file " << fn.c_str() << endl;
  if (_sndfile == NULL)
    throw jio_error("Could not open file %s.", fn.c_str());
  if (sf_error(_sndfile)) {
    sf_close(_sndfile);
    throw jio_error("sndfile error: %s.", sf_strerror(_sndfile));
  }
#ifdef DEBUG
  cout << "channels: "   << _sfinfo.channels   << endl;
  cout << "frames: "     << _sfinfo.frames     << endl;
  cout << "samplerate: " << _sfinfo.samplerate << endl;
#endif
  if (normalize)
    sf_command(_sndfile, SFC_SET_NORM_FLOAT, NULL, SF_TRUE);
  else
    sf_command(_sndfile, SFC_SET_NORM_FLOAT, NULL, SF_FALSE);
}
}

// ----- methods for class `DirectSampleFeature' -----
//
DirectSampleFeature::DirectSampleFeature(const SoundFilePtr &sndfile, unsigned blockLen,
					 unsigned start, unsigned end, const String& nm)
  : VectorFloatFeatureStream(blockLen*sndfile->channels(), nm), _sndfile(sndfile),
    _blockLen(blockLen), _start(start), _end(end), _cur(0)
{
  if (_end == (unsigned)-1) _end = _sndfile->frames();
  _sndfile->seek(_start, SEEK_SET);
}

const gsl_vector_float* DirectSampleFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  _increment();

  if (_cur >= _end)
    throw jiterator_error("end of samples!");

  unsigned readN;
  if (_cur + _blockLen >= _end) {
    gsl_vector_float_set_zero(_vector);
    readN = _sndfile->readf(gsl_vector_float_ptr(_vector, 0), _end-_cur);
    if (readN != _end-_cur)
      throw jio_error("Problem while reading from file. (%d != %d)", readN, _end-_cur);
  } else {
    readN = _sndfile->readf(gsl_vector_float_ptr(_vector, 0), _blockLen);
    if (readN != _blockLen)
      throw jio_error("Problem while reading from file. (%d != %d)", readN, _blockLen);
  }
  _cur += readN;
  if (readN == 0)
    throw jiterator_error("end of samples!");
  
  return _vector;
}

// ----- methods for class `DirectSampleOutputFeature' -----
//
DirectSampleOutputFeature::DirectSampleOutputFeature(const VectorFloatFeatureStreamPtr& src,
						     const SoundFilePtr &sndfile,
						     const String& nm)
  : VectorFloatFeatureStream(src->size(), nm), _src(src), _sndfile(sndfile)
{
  _blockLen = size() / _sndfile->channels();
  if ((size() % _sndfile->channels()) != 0)
    throw jconsistency_error("Block length (%d) is not a multiple of the number of channels (%d)\n",
			     size(), _sndfile->channels());
  _sndfile->seek(0, SEEK_SET);
}

const gsl_vector_float* DirectSampleOutputFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  _increment();

  gsl_vector_float_memcpy(_vector, _src->next(_frameX));
  unsigned n = _sndfile->writef(gsl_vector_float_ptr(_vector, 0), _blockLen);
  if (n != _blockLen)
    throw jio_error("Problem while writing to file. (%d != %d)", n, _blockLen);

  return _vector;
}

const gsl_vector_float* ChannelExtractionFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  _increment();

  const gsl_vector_float* allChannels = _src->next(_frameX);
  gsl_vector_float_set_zero(_vector);
  for (unsigned i=0; i<size(); i++) {
    gsl_vector_float_set(_vector, i, gsl_vector_float_get(allChannels, i*_chN+_chX));
  }
  return _vector;
}


// ----- Methods for class 'SignalInterferenceFeature' -----
//
SignalInterferenceFeature::
SignalInterferenceFeature(VectorFloatFeatureStreamPtr& signal,  VectorFloatFeatureStreamPtr& interference,
			  double dBInterference, unsigned blockLen, const String& nm):
  VectorFloatFeatureStream(blockLen, nm),
  _signal(signal), _interference(interference), _level(pow(10.0, dBInterference / 20.0)) { }

const gsl_vector_float* SignalInterferenceFeature::next(int frameX) {
  
  if (frameX == _frameX) return _vector;

  gsl_vector_float_memcpy(_vector, _interference->next(frameX));
  
  gsl_vector_float_scale(_vector, _level);
  gsl_vector_float_add(_vector, _signal->next(frameX));

  _increment();
  return _vector;
}


// ----- Methods for class 'AmplificationFeature' -----
//
const gsl_vector_float* AmplificationFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  _increment();

  const gsl_vector_float* vector = _src->next(_frameX);
  for (unsigned i=0; i<size(); i++) {
    gsl_vector_float_set(_vector, i, gsl_vector_float_get(vector, i)*_amplify);
  }
  return _vector;
}

 // ----- definition for class `WriteSoundFile' -----
//

WriteSoundFile::WriteSoundFile(const String& fn, int sampleRate, int nChan, int format )
{
  _sfinfo.samplerate = sampleRate;
  _sfinfo.channels = nChan;
  _sfinfo.format = format;
  _sfinfo.frames = 0;
  _sfinfo.sections = 0;
  _sfinfo.seekable = 0;
  
  _sndfile = sndfile::sf_open(fn.c_str(), sndfile::SFM_WRITE, &_sfinfo);
  if (!_sndfile)
    throw jio_error("Error opening file %s.", fn.c_str());
}

WriteSoundFile::~WriteSoundFile()
{
  sf_close(_sndfile);
}

int WriteSoundFile::write( gsl_vector *vector )
{
  using namespace sndfile;
  int ret;

  if( _sfinfo.format & SF_FORMAT_FLOAT )
    ret = writeFloat( vector );
  else if(_sfinfo.format & SF_FORMAT_PCM_32 )
    ret = writeInt( vector );
  else
    ret = writeShort( vector );

  return ret;
}

int WriteSoundFile::writeInt( gsl_vector *vector )
{
  sndfile::sf_count_t frames =  _sfinfo.channels * vector->size;
  int *buf = new int[frames];
  
  for(sndfile::sf_count_t i=0;i<frames;i++)
    buf[i] = gsl_vector_get( vector, i );
  int ret = (int)sf_writef_int( _sndfile, buf, frames);
  
  delete [] buf;
  return ret;
}

int WriteSoundFile::writeFloat( gsl_vector *vector )
{
  sndfile::sf_count_t frames =  _sfinfo.channels * vector->size;
  float *buf = new float[frames];
  
  for(sndfile::sf_count_t i=0;i<frames;i++)
    buf[i] = gsl_vector_get( vector, i );
  int ret = (int)sf_writef_float( _sndfile, buf, frames);
  
  delete [] buf;
  return ret;
}

int WriteSoundFile::writeShort( gsl_vector *vector )
{
  sndfile::sf_count_t frames =  _sfinfo.channels * vector->size;
  short *buf = new short[frames];
  
  for(sndfile::sf_count_t i=0;i<frames;i++)
    buf[i] = gsl_vector_get( vector, i );
  int ret = (int)sf_writef_short( _sndfile, buf, frames);
  
  delete [] buf;
  return ret;
}
 
// ----- Methods for class 'HTK' -----
//
#define HASENERGY  0100       /* _E log energy included */
#define HASNULLE   0200       /* _N absolute energy suppressed */
#define HASDELTA   0400       /* _D delta coef appended */
#define HASACCS   01000       /* _A acceleration coefs appended */
#define HASCOMPX  02000       /* _C is compressed */
#define HASZEROM  04000       /* _Z zero meaned */
#define HASCRCC  010000       /* _K has CRC check */
#define HASZEROC 020000       /* _0 0'th Cepstra included */
#define HASVQ    040000       /* _V has VQ index attached */
#define HASTHIRD 0100000       /* _T has Delta-Delta-Delta index attached */

typedef int int32;
/* SwapInt32: swap byte order of int32 data value *p */
void SwapInt32(int32 *p)
{
   char temp,*q;
   
   q = (char*) p;
   temp = *q; *q = *(q+3); *(q+3) = temp;
   temp = *(q+1); *(q+1) = *(q+2); *(q+2) = temp;
}

/* SwapShort: swap byte order of short data value *p */
void SwapShort(short *p)
{
   char temp,*q;
   
   q = (char*) p;
   temp = *q; *q = *(q+1); *(q+1) = temp;
}

/**
   @brief
*/
bool ReadHTKHeader( FILE *fp, bool isBigEndian, int32 *pNSamples, 
		    int32 *pSampPeriod, short *pSampSize, short *pParmKind )
{
  size_t ret;
  int32 nSamples;
  int32 sampPeriod;
  short sampSize;
  short parmKind;

  ret = fread( &nSamples, sizeof(int32),   1, fp);
  ret = fread( &sampPeriod, sizeof(int32), 1, fp);
  ret = fread( &sampSize, sizeof(short), 1, fp);
  ret = fread( &parmKind, sizeof(short), 1, fp);

  if ( isBigEndian == false ){
    SwapInt32(&nSamples); 
    SwapInt32(&sampPeriod);
    SwapShort(&sampSize);
    SwapShort(&parmKind);
  }
  
  *pNSamples = nSamples;
  *pSampPeriod = sampPeriod;
  *pSampSize = sampSize;
  *pParmKind = parmKind;
  
  return true;
}

/**
   @brief write the HTK header taht is s 12 bytes long 
   and contains the following data:
   nSamples   Number of samples in file (4-byte integer)
   sampPeriod Sample period in 100ns units (4-byte integer)
   sampSize   Number of bytes per sample (2-byte integer)
   parmKind   Digit code indicating the sample kind (2-byte integer)
*/
bool WriteHTKHeader(FILE *fp, int32 nSamples, int32 sampPeriod, short sampSize, short parmKind, 
		    bool isBigEndian=false )
{
  size_t ret;

  if ( isBigEndian == false ){
    SwapInt32(&nSamples); 
    SwapInt32(&sampPeriod);
    SwapShort(&sampSize);
    SwapShort(&parmKind);
  }
   
  ret = fwrite( &nSamples, 1, sizeof(int), fp);
  ret = fwrite( &sampPeriod, 1, sizeof(int), fp);
  ret = fwrite( &sampSize, 1, sizeof(short), fp);
  ret = fwrite( &parmKind, 1, sizeof(short), fp);

  return true;
}

bool ReadFloatBinary (FILE *fp, bool isBigEndian, float *x, int n )
{
  int j;
  float *p;
  size_t ret;
   
  if( fread( x, sizeof(float), n ,fp ) != n )
    return false;

  if( isBigEndian == false ){
    for(p=x,j=0;j<n;p++,j++){
      //fprintf(stderr,"%f->", *p);
      SwapInt32((int32*)p);  /* Write in SUNSO unless natWriteOrder=T */
      //fprintf(stderr,"%f ", *p);
    }
  }
  
  return true;
}

bool WriteFloatBinary (FILE *f, float *x, int n, bool isBigEndian=false )
{
   int j;
   float *p;
   size_t ret;
   
   if( isBigEndian == false ){
     for(p=x,j=0;j<n;p++,j++)
       SwapInt32((int32*)p);  /* Write in SUNSO unless natWriteOrder=T */
   }
   ret = fwrite(x,sizeof(float),n,f);
   if( ret != (size_t)n ){
     return false;
   }
   if( isBigEndian == false ){
     for(p=x,j=0;j<n;p++,j++)
       SwapInt32((int32*)p);  /* Swap Back */
   }

   return true;
}

/**
   @brief write values of feature vectors in the HTK format
   @param const VectorFloatFeatureStreamPtr& src
   @param const String& outputfile
   @param int nSamples
   @param int sampPeriod
   @param short sampSize
   @param short parmKind
   @param bool isBigEndian
   0  WAVEFORM
   1  LPC
   2  LPREFC
   3  LPCEPSTRA
   4  LPDELCEP
   5  IREFC
   6  MFCC
   7  FBANK
   8  MELSPEC
   9  USER
   10 DISCRETE
   11 PLP
*/
WriteHTKFeatureFile::WriteHTKFeatureFile(const String& outputfile,
					 int nSamples, int sampPeriod, short sampSize, short parmKind, bool isBigEndian, 
					 const String& nm ):
  _isBigEndian(isBigEndian)
{
  _fp = fopen( outputfile.c_str(), "wb" );
  if( NULL == _fp ){
    fprintf(stderr,"WriteHTKFeatureFile : could not open %s\n",outputfile.c_str());
    throw jio_error("WriteHTKFeatureFile : could not open %s\n",outputfile.c_str());
  }
  
  if( false == WriteHTKHeader( _fp, (int32)nSamples, (int32)sampPeriod, (short)sampSize, (short)parmKind, _isBigEndian ) ){
    fprintf(stderr,"WriteHTKFeatureFile : WriteHTKHeader() failed\n");
    throw jio_error("WriteHTKFeatureFile : WriteHTKHeader() failed\n");
  }

  if( HASCOMPX & parmKind ){
    fprintf(stderr,"The file compression is not supported\n");
    throw jio_error("The file compression is not supported\n");
  }
  if( HASCRCC & parmKind ){
    fprintf(stderr,"The CRC is not supported\n");
    throw jio_error("The CRC is not supported\n");
  }

  _bufSize = sampSize/sizeof(float);
  _buf = new float[_bufSize];
 }

WriteHTKFeatureFile::~WriteHTKFeatureFile()
{
  fclose(_fp);
  delete [] _buf;
}

void WriteHTKFeatureFile::write( gsl_vector *vector )
{
  for (unsigned i=0; i<_bufSize; i++) {
    _buf[i] = gsl_vector_get( vector, i );
    //gsl_vector_float_set(_vector, i, gsl_vector_float_get(vector, i) );    
  }

  if( false==WriteFloatBinary( _fp, _buf, _bufSize, _isBigEndian ) ){
    fprintf(stderr,"WriteHTKFeatureFile : WriteFloatBinary() failed\n");
    throw jio_error("WriteHTKFeatureFile : WriteFloatBinary() failed\n");
  }
  
  return;
}

// ----- definition for class `HTKFeature' -----
// 
HTKFeature::HTKFeature(const String& inputfile, int vecSize, bool isBigEndian, const String& nm ):
  VectorFeatureStream(vecSize, nm), 
  _isBigEndian(isBigEndian),
  _fp(NULL),
  _buf(NULL),
  _sampSize(vecSize*sizeof(float))
{
  _buf = new float[vecSize];

  if ( inputfile != "") read(inputfile);
}

HTKFeature::~HTKFeature()
{
  if( NULL != _fp )
    fclose(_fp);
  if( NULL != _buf )
    delete [] _buf;
}

bool HTKFeature::read( const String& inputfile, bool isBigEndian )
{
  int32 nSamples;
  int32 sampPeriod;
  short sampSize;
  short parmKind;

  _fp = fopen( inputfile.c_str(), "rb" );
  if( NULL == _fp ){
    fprintf(stderr,"cannot open file %s\n", inputfile.c_str() );
    return false;
  }

  if( false== ReadHTKHeader( _fp, _isBigEndian, &nSamples, &sampPeriod, &sampSize, &parmKind ) )
    return false;
  
  if( HASCOMPX & parmKind ){
    fprintf(stderr,"The file is compressed\n"
	    "SAVECOMPRESSED=F\n");
    _nSamples   = nSamples - 4;
    return false;
  }
  else if( HASCRCC & parmKind ){
    fprintf(stderr,"The CRC is not supported\n"
	    "set SAVEWITHCRC=F\n");
    return false;
  }
  else{
    _nSamples   = nSamples;
    _sampPeriod = sampPeriod;
    _sampSize = sampSize;
    _parmKind = parmKind;
  }

#if 0
  fprintf(stderr,"nSample %d\n", _nSamples);
  fprintf(stderr,"Period %d\n",  _sampPeriod);
  fprintf(stderr,"Size per sample %d\n", _sampSize);
  fprintf(stderr,"Parameter Kind %0x\n", _parmKind);
#endif

  if( NULL != _buf )
    delete [] _buf;
  _buf = new float[_sampSize/sizeof(float)];

  return true;
}

const gsl_vector* HTKFeature::next(int frameX)
{
  
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX){
    fprintf(stderr,"Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);
  }
  
  _increment();

  if( false==ReadFloatBinary( _fp, _isBigEndian, _buf, _sampSize/sizeof(float) ) ){
    fprintf(stderr,"ReadFloatBinary() failed\n");
    throw j_error("ReadFloatBinary() failed\n");
  }

  for (unsigned i=0; i<size(); i++) {
    gsl_vector_set( _vector, i, _buf[i] );    
  }
 
  return _vector;
}
