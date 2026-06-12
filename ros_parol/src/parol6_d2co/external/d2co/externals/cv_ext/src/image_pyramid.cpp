#include <stdexcept>

#include "cv_ext/image_pyramid.h"

using namespace cv;
using namespace cv_ext;
using namespace std;

ImagePyramid::ImagePyramid() :
  num_levels_ (0),
  scale_factor_ (0.0),
  gaussian_pyr_(true),
  interp_type_(cv_ext::INTERP_NEAREST_NEIGHBOR){}

void ImagePyramid::buildGaussianPyrFromImage( const Mat &img, int pyr_num_levels, 
                                              int pyr_img_type, bool deep_copy )
{
  if( pyr_num_levels < 1 )
    throw invalid_argument ( "Number of level must be greater or equal than 1" );
  
  scale_factor_ = 2.0; 
  gaussian_pyr_ = true;
  num_levels_ = pyr_num_levels;
  interp_type_ = cv_ext::INTERP_NEAREST_NEIGHBOR;
  
  initPyr( img, pyr_img_type, deep_copy );
}

void ImagePyramid::buildFromImage ( const Mat& img, int pyr_num_levels, double pyr_scale_factor,
                                    int pyr_img_type, bool deep_copy, InterpolationType interp_type )
{
  
  if ( pyr_scale_factor <= 1.0 || pyr_scale_factor > 2.0 )
    throw invalid_argument ( "Pyramid scale factor must be in the range (1,2]" );
  
  if( pyr_num_levels < 1 )
    throw invalid_argument ( "Number of level must be greater or equal than 1" );

  scale_factor_ = pyr_scale_factor; 
  gaussian_pyr_ = false;
  num_levels_ = pyr_num_levels;
  interp_type_ = interp_type;
  
  initPyr( img, pyr_img_type, deep_copy );
}

void cv_ext::ImagePyramid::precomputeAllLevels()
{
  computeScaleLevel ( num_levels_ - 1 );
}


double ImagePyramid::getScale ( int level ) const
{
  if( level < 0 || level >= num_levels_ )
    throw invalid_argument("Scale index out of range");
  
  return scales_[level];
}

Mat& ImagePyramid::operator[]( int pyr_level )
{
  if( pyr_level < 0 || pyr_level >= num_levels_ )
    throw invalid_argument("Scale index out of range");
  
  if( !init_[pyr_level] )
    computeScaleLevel( pyr_level );
  return pyr_imgs_[pyr_level]; 
}

const Mat& ImagePyramid::at( int pyr_level )
{
  return operator[]( pyr_level );
}

void cv_ext::ImagePyramid::initPyr( const Mat& img, int pyr_img_type, bool deep_copy )
{    
  pyr_imgs_.resize( num_levels_ );
  scales_.resize( num_levels_ );
  init_.resize( num_levels_ );
  init_.assign(init_.size(), false);
  init_[0] = true;
  scales_[0] = 1.0;

  for ( int i = 1; i < num_levels_; i++ )
    scales_[i] = scales_[i-1]*scale_factor_;
  
  if( pyr_img_type < 0 || img.type() == pyr_img_type )
  {
    if( deep_copy )
      pyr_imgs_[0] = img.clone();
    else
      pyr_imgs_[0] = img;
  }
  else
    img.convertTo( pyr_imgs_[0], pyr_img_type );
}

void ImagePyramid::computeScaleLevel ( int scale_index )
{
  // Scale down to obtain the required pyramid level
  if( gaussian_pyr_ )
  {
    // Use pyrDown() for scale factor 2
    for ( int i = 1; i <= scale_index; i++ )
    {
      if( !init_[i] )
      {
        pyrDown(pyr_imgs_[i-1], pyr_imgs_[i]);
        init_[i] = true;
      }
    }
  }
  else
  {
    // ... otherwise, use resize()
    double scale = 1.0/scale_factor_;
    
    int interp_type;
    switch( interp_type_ )
    {
      case cv_ext::INTERP_NEAREST_NEIGHBOR:
        interp_type = INTER_NEAREST;
        break;
      case cv_ext::INTERP_BILINEAR:
        interp_type = INTER_LINEAR;
        break;
      case cv_ext::INTERP_BICUBIC:
        interp_type = INTER_CUBIC;
        break;
      default:
        interp_type = INTER_NEAREST;
        break;
    }
    
    for ( int i = 1; i <= scale_index; i++ )
    {
      if( !init_[i] )
      {
        resize(pyr_imgs_[i-1], pyr_imgs_[i], Size(0,0), scale, scale, interp_type );
        init_[i] = true;
      }
    }
  }
}
