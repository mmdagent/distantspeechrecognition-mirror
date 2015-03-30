//
//                               Millennium
//                   Distant Speech Recognition System
//                                  (dsr)
//
//  Module:  asr.gaussian
//  Purpose: Basic acoustic likelihood computation.
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
#include "gaussian/distribBasic.h"
#include "common/mach_ind_io.h"


// ----- methods for class `DistribBasic' -----
//
DistribBasic::DistribBasic(const String& nm, CodebookBasicPtr& cb)
  : Distrib(nm),
    _val(new float[cb->refN()]), _valN(cb->refN()), _count(0.0),
    _codebook(cb)
{
  _scoreFunction = &_score;

  for (unsigned valX = 0; valX < _valN; valX++)
    _val[valX] = log(double(_valN));
}

DistribBasic::DistribBasic(const DistribBasic& ds)
  : Distrib(ds.name()),
    _val(new float[ds._valN]), _valN(ds._valN), _count(ds._count),
    _codebook(ds._codebook)
{
  memcpy(_val, ds._val, _valN * sizeof(float));
}

String DistribBasic::puts() const
{
  static char buffer[500];

  sprintf(buffer, "Distribution '%s' : %d Gaussians",
	  _name.c_str(), _valN);

  return String(buffer);
}

DistribBasic& DistribBasic::operator=(const DistribBasic& ds)
{
  _name     = ds._name;

  if (_valN != ds._valN) {
    delete[] _val;
    _val  = new float[ds._valN];
    _valN = ds._valN;
  }
  memcpy(_val, ds._val, _valN * sizeof(float));

  _count    = ds._count;
  _codebook = ds._codebook;

  return *this;
}

DistribBasic::~DistribBasic()
{
  delete[] _val;
}

const unsigned DistribBasic::MaxNameLength = 200;

void DistribBasic::save(FILE *fp) const
{
  // write the names of the distribution and the codebook used
  write_string(fp, name());  write_string(fp, cbk()->name());

  // write the size of the distribution (and the codebook)
  // The negative value indicates that there is also a
  // count field present.
  write_int(fp, (-1) * _valN);

  // write the counts the distribution got during training
  write_float(fp,_count);

  // write the values of the distribution
  for (UnShrt refX = 0; refX < _valN; refX++)
    write_float(fp, _val[refX]);
}

void DistribBasic::load(FILE *fp, bool readName)
{
  static char cbname[MaxNameLength];
  if (readName)
    read_string(fp, cbname);
  read_string(fp, cbname);

  if (strcmp(cbk()->name(),cbname) != 0)
    throw j_error("Codebook names (%s vs. %s) do not match.",
		  cbk()->name().chars(), cbname);

  // Read the size of the distribution.
  // If the size is negative, another file format has been used. In this
  // case, immediately after refN one float is in the file which gives
  // the number of counts this distribution got during training, and refN
  // is just (-1.0 * refN). This is to ensure harmony between the old
  // file format without the count field and the newer one with count.
  int refN = read_int(fp);

  // is a count field present? Indicator is a negative refN
  if (refN < 0) { _valN = (-refN); _count = read_float(fp); }
  else { _valN = refN; }

  // check the size of the used codebook
  if (_valN != cbk()->refN()) {
      for (UnShrt refX = 0; refX < _valN; refX++) read_float(fp);

      throw j_error( "Distribution/codebook size mismatch: %s/%s %d/%d (not loaded!)\n",
		     _name.chars(), cbk()->name().chars(), _valN, cbk()->refN());
  } else {

    // now, everything is checked, so read the weights
    if (_val==NULL) {
      for (UnShrt refX = 0; refX < _valN; refX++) (void)read_float(fp);
      throw j_error("Value array of %s is NULL", _name.chars());
    }
    for (UnShrt refX = 0; refX < _valN; refX++) _val[refX] = read_float(fp);
  }
  if (feof(fp)) throw j_error("premature end of file\n");
}

void DistribBasic::write(FILE *fp) const
{
  fprintf(fp, "%-25s%-25s\n", _name.c_str(), cbk()->name().c_str());
}

const float DistribBasic::SmallCount = 1.0E-04;

void DistribBasic::initializeFromCodebook()
{
  const float* cbkCounts = cbk()->count();

  double ttlCounts = 0.0;
  for (unsigned refX = 0; refX < cbk()->refN(); refX++)
    ttlCounts += (cbkCounts[refX] + SmallCount);

  if ((ttlCounts - cbk()->refN() * SmallCount) < SmallCount)
    printf("WARNING: Codebook %s has no counts.\n", cbk()->name().c_str());

  for (unsigned refX = 0; refX < cbk()->refN(); refX++)
    _val[refX] = -log((cbkCounts[refX] + SmallCount) / ttlCounts);

  _count = ttlCounts;
}

void DistribBasic::copyWeights(const DistribBasicPtr& ds)
{
  if (ds->_valN != _valN) {
    delete[] _val;
    _val = new float[ds->_valN];
    _valN = ds->_valN;
  }

  memcpy(_val, ds->_val, _valN * sizeof(float));
  _count = 0.0;
}


// ----- methods for class `DistribMultiStream' -----
//
DistribMultiStream::DistribMultiStream(const String& nm, DistribPtr& audioDist, DistribPtr& videoDist, double audioWeight, double videoWeight)
  : Distrib(nm), _audioDist(audioDist), _videoDist(videoDist), _audioWeight(audioWeight), _videoWeight(videoWeight)
{
  _scoreFunction = &_score;
}


// ----- methods for class `DistribSetBasic' -----
//
DistribSetBasic::DistribSetBasic(CodebookSetBasicPtr& cbs, const String& descFile, const String& distFile)
  : _codebookSet(cbs)
{
  if (descFile == "") return;

  freadAdd(descFile, ';', this);

  if (distFile == "")
    _initializeFromCodebooks();
  else
    load(distFile);
}

DistribSetBasic::~DistribSetBasic() { }

static void _splitList(const String& s, list<String>& line)
{
  String::size_type pos = 0;
  for (unsigned i = 0; i < 2; i++) {
    String::size_type spacePos = s.find_first_of(' ', pos);
    line.push_back(s.substr(pos, spacePos - pos));
    pos = spacePos + 1;
    while (s[pos] == ' ') pos++;
  }
}

void DistribSetBasic::__add(const String& s)
{
  list<String> line;
  // splitList(s, line);
  _splitList(s, line);

  list<String>::const_iterator itr = line.begin();

  String  name(*itr);   itr++;   assert(itr != line.end());
  String  dsname(*itr);

  // printf("name='%s' dsname='%s'\n", name.c_str(), dsname.c_str());

  _addDist(name, dsname);
}

void DistribSetBasic::_addDist(const String& name, const String& cbname)
{
  CodebookBasicPtr& cb(cbs()->find(cbname));
  DistribBasic* ptr = new DistribBasic(name, cb);

  _list.add(name, DistribBasicPtr(ptr));
}

const unsigned DistribSetBasic::MaxNameLength = 200;
const int      DistribSetBasic::Magic         = 64207531;

void DistribSetBasic::_initializeFromCodebooks()
{
  printf("Initializing mixture weights from codebook counts.\n");  fflush(stdout);
  for (_Iterator itr(_list); itr.more(); itr++) {
    DistribBasicPtr& ds(Cast<DistribBasicPtr>(*itr));
    ds->initializeFromCodebook();
  }
}

void DistribSetBasic::load(const String& filename)
{
  if (filename == "")
    throw j_error("Distribution set file name is null.");

  FILE* fp = fileOpen(filename, "r");

  if (fp == NULL)
    throw jio_error("Could not open distribution set file %s.", filename.chars());

  printf("Loading distrib set from file %s.\n", filename.chars());
  fflush(stdout);

  int l_ds0 = 0;
  int l_dsN = read_int(fp);

  static char name[MaxNameLength];

  for (int idst = l_ds0; idst < l_dsN; idst++) {
    read_string(fp, name);

    /*
    printf("Searching for distribution set %s.\n", name);  fflush(stdout);
    */

    String nm(name);
    DistribBasicPtr ds = find(nm);
    ds->load(fp);
  }

  fileClose(filename, fp);
}

void DistribSetBasic::save(const String& filename) const
{
  if (filename == "")
    throw j_error("Distribution set file name is null.");

  FILE* fp = fileOpen(filename, "w");

  if (fp == NULL)
    throw jio_error("Could not open distribution set file %s.", filename.chars());

  write_int(fp, ndists());
  for (_ConstIterator itr(_list); itr.more(); itr++) {
    DistribBasicPtr& ds(Cast<DistribBasicPtr>(*itr));
    ds->save(fp);
  }

  fileClose(filename, fp);
}

void DistribSetBasic::write(const String& fileName, const String& time) const
{
  FILE* fp = fileOpen(fileName, "w");

  fprintf(fp, "; -------------------------------------------------------\n");
  fprintf(fp, ";  Name            : %s\n", "dss");
  fprintf(fp, ";  Type            : %s\n", "DistribSet");
  fprintf(fp, ";  Number of Items : %d\n", _list.size());
  fprintf(fp, ";  Date            : %s\n", time.c_str());
  fprintf(fp, "; -------------------------------------------------------\n");

  for (_ConstIterator itr(_list); itr.more(); itr++)
    Cast<DistribBasicPtr>((*itr))->write(fp);

  fileClose(fileName, fp);
}


// ----- definition for container class `DistribMap' -----
//
void DistribMap::read(const String& fileName)
{
  if (fileName == "")
    throw j_error("Distribution map file name is null.");

  FILE* fp = fileOpen(fileName, "r");

  if (fp == NULL)
    throw jio_error("Could not open distribution map file %s.", fileName.chars());

  unsigned distribN = read_int(fp);
  _mapping.erase(_mapping.begin(), _mapping.end());
  static char fromName[20];
  static char toName[20];
  for (unsigned i = 0; i < distribN; i++) {
    read_string(fp, fromName);
    read_string(fp, toName);
    _mapping[fromName] = toName;
  }

  fileClose(fileName, fp);
}

const String& DistribMap::mapped(const String& name) const
{
  _MappingConstIterator itr = _mapping.find(name);

  if (itr == _mapping.end())
    throw jindex_error("No mapping for %s", name.c_str());

  return (*itr).second;
}

void DistribMap::write(const String& fileName) const
{
  if (fileName == "")
    throw j_error("Distribution map file name is null.");

  FILE* fp = fileOpen(fileName, "w");

  if (fp == NULL)
    throw jio_error("Could not open distribution map file %s.", fileName.chars());

  write_int(fp, _mapping.size());
  for (_MappingConstIterator itr = _mapping.begin(); itr != _mapping.end(); itr++) {
    write_string(fp, (*itr).first);
    write_string(fp, (*itr).second);
  }

  fileClose(fileName, fp);
}


// ----- methods for class `DistibSetMultiStream' -----
//
DistribSetMultiStream::
DistribSetMultiStream(DistribSetBasicPtr& audioDistSet, DistribSetBasicPtr& videoDistSet, double audioWeight, double videoWeight)
  : _audioDistribSet(audioDistSet), _videoDistribSet(videoDistSet)
{
  for (DistribSetBasic::Iterator itr(_audioDistribSet); itr.more(); itr++) {
    DistribBasicPtr& audio(itr.dst());
    const String& name(audio->name());
    DistribBasicPtr& video(_videoDistribSet->find(name));

    _list.add(name, DistribMultiStreamPtr(new DistribMultiStream(name, audio, video, audioWeight, videoWeight)));
  }
}

DistribSetMultiStream::
DistribSetMultiStream(DistribSetBasicPtr& audioDistSet, DistribSetBasicPtr& videoDistSet, DistribMapPtr& distribMap, double audioWeight, double videoWeight)
  : _audioDistribSet(audioDistSet), _videoDistribSet(videoDistSet)
{
  cout << "Creating DistribSetMultiStream object" << endl;

  for (DistribSetBasic::Iterator itr(_audioDistribSet); itr.more(); itr++) {
    DistribBasicPtr& audio(itr.dst());
    const String& name(audio->name());

    // cout << "Adding " << name << " --> " << distribMap->mapped(name) <<  endl;

    DistribBasicPtr& video(_videoDistribSet->find(distribMap->mapped(name)));

    _list.add(name, DistribMultiStreamPtr(new DistribMultiStream(name, audio, video, audioWeight, videoWeight)));
  }
}

void DistribSetMultiStream::setWeights( double audioWeight, double videoWeight )
{
  for (_ConstIterator itr(_list); itr.more(); itr++){
    Cast<DistribMultiStreamPtr>((*itr))->setWeights( audioWeight, videoWeight );    
  }
}
