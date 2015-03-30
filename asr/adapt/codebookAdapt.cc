//
//                               Millennium
//                   Automatic Speech Recognition System
//                                  (asr)
//
//  Module:  asr.adapt
//  Purpose: Maximum likelihood model space adaptation. 
//  Author:  John McDonough
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
#include "adapt/codebookAdapt.h"


// --- methods for class `CodebookAdapt' ---
//
CodebookAdapt::CodebookAdapt(const String& nm, UnShrt rfN, UnShrt dmN, CovType cvTp,
			     VectorFloatFeatureStreamPtr feat)
  : CodebookBasic(nm, rfN, dmN, cvTp, feat), _origRV(NULL) { }

CodebookAdapt::CodebookAdapt(const String& nm, UnShrt rfN, UnShrt dmN, CovType cvTp,
			     const String& featureName)
  : CodebookBasic(nm, rfN, dmN, cvTp, featureName), _origRV(NULL) { }

CodebookAdapt::~CodebookAdapt()
{
  if (_origRV) gsl_matrix_float_free(_origRV);
}

void CodebookAdapt::_allocRV()
{
  CodebookBasic::_allocRV();
  
  if (_origRV == NULL) return;

  gsl_matrix_float_free(_origRV);  _origRV = NULL;
}

void CodebookAdapt::resetMean()
{
  if (_origRV == NULL) return;

  if (_rv != NULL) gsl_matrix_float_free(_rv);

  _rv = _origRV;  _origRV = NULL;
}


// ----- methods for class `CodebookAdapt::GaussDensity' -----
//
void CodebookAdapt::GaussDensity::allocMean()
{
  // allocate space for new means
  gsl_matrix_float* origRV = cbk()->_origRV;
  if (origRV != NULL) return;

  cbk()->_origRV = cbk()->_rv;
  cbk()->_rv     = gsl_matrix_float_alloc(cbk()->_refN, cbk()->featLen());

  // copy over original values
  for (UnShrt refX = 0; refX < cbk()->_refN; refX++) {
    NaturalVector newMean(cbk()->_rv->data + refX * cbk()->_rv->size2, cbk()->_rv->size2);
    NaturalVector oldMean(cbk()->_origRV->data + refX * cbk()->_origRV->size2, cbk()->_origRV->size2);
    newMean = oldMean;
  }
}


// ----- methods for class `CodebookSetAdapt' -----
//
CodebookSetAdapt::CodebookSetAdapt(const String& descFile, FeatureSetPtr& fs, const String& cbkFile)
  : CodebookSetBasic(/* descFile= */ "", fs)
{
  if (descFile == "") return;  

  freadAdd(descFile, ';', this);

  if (cbkFile == "") return;

  load(cbkFile);
}

void CodebookSetAdapt::_addCbk(const String& name, UnShrt rfN, UnShrt dmN, CovType cvTp,
			       const String& featureName)
{
  CodebookAdaptPtr cbk(new CodebookAdapt(name, rfN, dmN, cvTp, featureName));

  _cblist.add(name, cbk);
}

void CodebookSetAdapt::_addCbk(const String& name, UnShrt rfN, UnShrt dmN, CovType cvTp,
			       VectorFloatFeatureStreamPtr feat)
{
  CodebookAdaptPtr cbk(new CodebookAdapt(name, rfN, dmN, cvTp, feat));

  _cblist.add(name, cbk);
}

void CodebookSetAdapt::resetMean()
{
  for (_Iterator itr(_cblist); itr.more(); itr++) {
    CodebookAdaptPtr& cbk(Cast<CodebookAdaptPtr>(*itr));
    cbk->resetMean();
  }
}
