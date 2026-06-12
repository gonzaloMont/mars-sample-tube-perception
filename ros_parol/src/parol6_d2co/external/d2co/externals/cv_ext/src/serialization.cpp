#include "cv_ext/serialization.h"
#include "cv_ext/conversions.h"

#include <stdexcept>

using namespace std;
using namespace cv;
using namespace cv_ext;

string cv_ext::generateYAMLFilename ( const string& filename )
{
  int pos1 = filename.length() - 4, pos2 = filename.length() - 5;
  string ext_str1 = filename.substr ((pos1 > 0)?pos1:0);
  string ext_str2 = filename.substr ((pos2 > 0)?pos2:0);

  if( ext_str1.compare(".yml") && ext_str2.compare(".yaml") &&
      ext_str1.compare(".YML") && ext_str2.compare(".YAML") )
    return filename + ".yml";
  else 
    return filename;
}

void cv_ext::read3DTransf( string filename, Mat &rotation, 
                           Mat &translation, RotationFormat rf )
{
  string yml_filename = generateYAMLFilename(filename);
  FileStorage fs ( yml_filename, FileStorage::READ );

  Mat r, t;
  fs["rotation"] >> r;
  fs["translation"] >> t;
  
  
  switch( rf )
  {
    case ROT_FORMAT_MATRIX:
      
      if( r.rows == 3 && r.cols == 3 )
        rotation = r;
      else if( r.rows == 3 && r.cols == 1 )
        exp2RotMat<double>(r, rotation);
      else if( r.rows == 1 && r.cols == 3 )
      {
        cv::transpose(r,r);
        exp2RotMat<double>(r, rotation);
      }
      else
        throw runtime_error("read3DTransf() : invalid file format");
      
      break;
      
    case ROT_FORMAT_AXIS_ANGLE:
      
      if( r.rows == 3 && r.cols == 1 )
        rotation = r;
      else if( r.rows == 1 && r.cols == 3 )
        cv::transpose(r,rotation);
      else if( r.rows == 3 && r.cols == 3 )
        rotMat2Exp<double>(r, rotation );
      else
        throw runtime_error("read3DTransf() : invalid file format");
      
      break;
  }
  
  if( t.rows == 3 && t.cols == 1 )
    translation = t;
  else if( t.rows == 1 && t.cols == 3 )
    cv::transpose(t,translation);
  else
    throw runtime_error("read3DTransf() : invalid file format");

  fs.release();
}

void cv_ext::write3DTransf( string filename, const Mat &rotation,
                                    const Mat &translation )
{
  if( !( rotation.rows == 3 && rotation.cols == 3 ) &&
      !( rotation.rows == 3 && rotation.cols == 1 ) && 
      !( rotation.rows == 1 && rotation.cols == 3 ) )
    throw invalid_argument("write3DTransf() : invalid file format");

  if( !( translation.rows == 3 && translation.cols == 1 ) &&
      !( translation.rows == 1 && translation.cols == 3 ) )
    throw invalid_argument("write3DTransf() : invalid file format");
  
  string yml_filename = generateYAMLFilename(filename);
  FileStorage fs ( yml_filename, FileStorage::WRITE );

  fs << "rotation" << rotation;
  fs << "translation" << translation;

  fs.release();
}
