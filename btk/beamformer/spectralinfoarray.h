#ifndef _spectralinfoarray_
#define _spectralinfoarray_

// ----- definition for class `SnapShotArray' -----
// 
class SnapShotArray {
 public:
  SnapShotArray(unsigned fftLn, unsigned nChn);
  virtual ~SnapShotArray();

  const gsl_vector_complex* getSnapShot(unsigned fbinX) const {
    assert (fbinX < _fftLen);
    return _specSnapShots[fbinX];
  }

  void newSample(const gsl_vector_complex* samp, unsigned chanX) const;
  void newSnapShot(const gsl_vector_complex* snapshots, unsigned fbinX);

  unsigned fftLen() const { return _fftLen; }
  unsigned nChan()  const { return _nChan;  }

  virtual void update();
  virtual void zero();

 protected:
  const unsigned	_fftLen;
  const unsigned	_nChan;

  mutable gsl_vector_complex**	_specSamples;
  mutable gsl_vector_complex**	_specSnapShots;
};

typedef refcount_ptr<SnapShotArray> 	SnapShotArrayPtr;


// ----- definition for class `SpectralMatrixArray' -----
// 
class SpectralMatrixArray : public SnapShotArray {
 public:
  SpectralMatrixArray(unsigned fftLn, unsigned nChn, double forgetFact = 0.95);
  virtual ~SpectralMatrixArray();

  gsl_matrix_complex* getSpecMatrix(unsigned idx) const {
    assert (idx < _fftLen);
    return _specMatrices[idx];
  }

  virtual void update();
  virtual void zero();

 protected:
  const gsl_complex	_mu;
  gsl_matrix_complex**	_specMatrices;
};

typedef refcount_ptr<SpectralMatrixArray> 	SpectralMatrixArrayPtr;

// ----- definition for class `FBSpectralMatrixArray' -----
// 
class FBSpectralMatrixArray : public SpectralMatrixArray {
 public:
  FBSpectralMatrixArray(unsigned fftLn, unsigned nChn,
			double forgetFact = 0.95);
  virtual ~FBSpectralMatrixArray();

  virtual void update();
};

#endif
