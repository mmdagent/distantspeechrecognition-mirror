#include "fileStream.h"
#include <string>
using namespace std;

FileHandler::FileHandler( const String &filename, const String &mode ):
  _fp(NULL),_filename(filename)
{
  _fp = fileOpen( filename.chars(), mode.chars() );
  if( NULL == _fp ){
    fprintf(stderr,"could not open %s\n", filename.chars());
  }
}

FileHandler::~FileHandler()
{
  fileClose( _filename.chars(), _fp );
}

int FileHandler::readInt()
{
  return read_int(_fp);
}

String FileHandler::readString()
{
  read_string(_fp, _buf);

  return String(_buf);
}

void FileHandler::writeInt( int val )
{
  write_int( _fp, val );
}
 
void FileHandler::writeString( String val )
{
  write_string(_fp, val.chars());
}

