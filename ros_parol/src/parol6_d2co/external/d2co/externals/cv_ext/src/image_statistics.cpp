#include "cv_ext/image_statistics.h"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <boost/concept_check.hpp>
#include <boost/thread/locks.hpp>

#include "cv_ext/interpolations.h"
#include "cv_ext/debug_tools.h"

using namespace cv_ext;

ImageStatistics :: ImageStatistics( const cv::Mat &img, bool deep_copy, unsigned int pyr_levels, 
                                    double pyr_scale_factor, double gaussian_stddev, bool bgr_color_order )
: OUT_OF_IMAGE_VALUE( std::numeric_limits<float>::max() )
{
  if( !img.rows || !img.cols || ( img.channels() != 3 && img.channels() != 1 ) )
    throw std::invalid_argument("Unsopported image: supported only images with 1 (i.e., grey levels) \
                                 or 3 channels (i.e., BGR or RGB)");
  
  bgr_color_order_ = bgr_color_order;
  pyr_num_levels_ = pyr_levels; 
  if(pyr_num_levels_ < 1 ) pyr_num_levels_ = 1;
  pyr_scale_factor_ = pyr_scale_factor;
  if( pyr_scale_factor_ < 1.0) pyr_scale_factor_ = 1.0;
  gaussian_stddev_ = gaussian_stddev;
  if( gaussian_stddev_ < 0) gaussian_stddev_ = 1.0;
  
  blured_imgs_.resize( pyr_num_levels_ );
  dx_imgs_.resize( pyr_num_levels_ );
  dy_imgs_.resize( pyr_num_levels_ );
  eigen_imgs_.resize( pyr_num_levels_ );
  eigen_mag_imgs_.resize( pyr_num_levels_ );
  gradient_dir_imgs_.resize( pyr_num_levels_ );
  gradient_mag_imgs_.resize( pyr_num_levels_ );
  
  setImage( img, deep_copy );
}

void ImageStatistics :: setImage( const cv::Mat& img, bool deep_copy )
{
  if( img.channels() == 3 )
  {
    // RGB input -> initialize both the pyramids
    rgb_pyr_.buildFromImage( img, pyr_num_levels_, pyr_scale_factor_,
                             cv::DataType<cv::Vec3f>::type, deep_copy, INTERP_BILINEAR );
    cv::Mat gl_img( img.size(), cv::DataType<float>::type );
    cv::cvtColor( rgb_pyr_.at(0), gl_img, 
                  bgr_color_order_?cv::COLOR_BGR2GRAY:cv::COLOR_RGB2GRAY );
    gl_pyr_.buildFromImage( gl_img, pyr_num_levels_, pyr_scale_factor_, 
                            cv::DataType<float>::type, false, INTERP_BILINEAR );
  }
  else
  {
    // GL input -> initialize only the GL pyramids
    gl_pyr_.buildFromImage( img, pyr_num_levels_, pyr_scale_factor_, 
                            cv::DataType<float>::type, deep_copy, INTERP_BILINEAR );
  }
}

void ImageStatistics :: computeBlurredImage( int scale_index )
{
  //cv::boxFilter(gl_img_, blured_img_, -1, cv::Size(3,3));
  cv::GaussianBlur( getIntensitiesImage( scale_index ), blured_imgs_[scale_index], cv::Size(0,0), gaussian_stddev_ );  
}

void ImageStatistics :: computeGradientImages( int scale_index )
{
  //cv::Sobel(_gl_img, dx_img, cv::DataType<float>::type, 1, 0, 3);
  //cv::Sobel(_gl_img, dy_img, cv::DataType<float>::type, 0, 1, 3); 
  cv::Scharr( getIntensitiesImage( scale_index ), dx_imgs_[scale_index], cv::DataType<float>::type, 1, 0 );
  cv::Scharr( getIntensitiesImage( scale_index ), dy_imgs_[scale_index], cv::DataType<float>::type, 0, 1 );  
}

void ImageStatistics :: computeEigenMagnitude( int scale_index )
{
  if( blured_imgs_[scale_index].empty() )
    computeBlurredImage( scale_index );
  
  // Compute Eigenvalues and Eigenvectors
  // TODO Optimize here
  cv::cornerEigenValsAndVecs(blured_imgs_[scale_index], eigen_imgs_[scale_index], 3, 3);
  
  cv::Mat lamda1_mat( gl_pyr_.at(scale_index).rows, gl_pyr_.at(scale_index).cols, cv::DataType<float>::type ),
          lamda2_mat( gl_pyr_.at(scale_index).rows, gl_pyr_.at(scale_index).cols, cv::DataType<float>::type );
  
  cv::Mat out[] = { lamda1_mat, lamda2_mat };
  int from_to[] = { 0,0, 1,1 };
  cv::mixChannels( &eigen_imgs_[scale_index], 1, out, 2, from_to, 2 );
  
  // Compute Eigenvalues magnitude
  cv::magnitude(lamda1_mat, lamda2_mat, eigen_mag_imgs_[scale_index]);
  cv::normalize(eigen_mag_imgs_[scale_index], eigen_mag_imgs_[scale_index], 
                1e-8, 1.0, cv::NORM_MINMAX);  
}

void ImageStatistics :: computeGradientDirections( int scale_index )
{
  if( dx_imgs_[scale_index].empty() || dy_imgs_[scale_index].empty() )
    computeGradientImages( scale_index );

  
  cv::Mat gradient_dir_img;

  int width = dx_imgs_[scale_index].cols, height = dx_imgs_[scale_index].rows;
  gradient_dir_imgs_[scale_index] = cv::Mat( dx_imgs_[scale_index].size(), cv::DataType<float>::type );
  
  for( int y = 0; y < height; y++)
  {
    const float *dx_p = dx_imgs_[scale_index].ptr<float>(y), 
                *dy_p = dy_imgs_[scale_index].ptr<float>(y);
    float *dir_p = gradient_dir_imgs_[scale_index].ptr<float>(y);
    for( int x = 0; x < width; x++, dx_p++, dy_p++, dir_p++)
    {
      if( *dx_p )
        *dir_p = atan(*dy_p / *dx_p);
      else
        *dir_p = M_PI/2;
    }
  }
}

void ImageStatistics :: computeGradientMagnitude( int scale_index )
{
  if( dx_imgs_[scale_index].empty() || dy_imgs_[scale_index].empty() )
    computeGradientImages( scale_index );
  
  cv::Mat gradient_mag_img, abs_dx_img, abs_dy_img;
  
  // Compute the gradient magnitude image
  abs_dx_img = cv::abs(dx_imgs_[scale_index]);
  abs_dy_img = cv::abs(dy_imgs_[scale_index]);
  gradient_mag_img = abs_dx_img + abs_dy_img;
  
  // WARNING
  //cv::pow(gradient_mag_img,1.5,gradient_mag_img);
  cv::normalize(gradient_mag_img, gradient_mag_imgs_[scale_index], 0, 1.0, cv::NORM_MINMAX, cv::DataType<float>::type);
}

const cv::Mat & ImageStatistics :: getIntensitiesImage( int scale_index )
{
  boost::lock_guard<boost::recursive_mutex> lock(mutex_);
  return gl_pyr_.at( scale_index );
}

const cv::Mat & ImageStatistics :: getColorsImage( int scale_index )
{
  boost::lock_guard<boost::recursive_mutex> lock(mutex_);
  // Only GL image is available: create a "dummy" RGB image converting from GL, and initialize the RGB pyramid
  if( !rgb_pyr_.numLevels() )
  {
    cv::Mat rgb_img( gl_pyr_.at(0).size(), cv::DataType<cv::Vec3f>::type );
    cv::cvtColor( gl_pyr_.at(0), rgb_img, 
                  bgr_color_order_?cv::COLOR_GRAY2BGR:cv::COLOR_GRAY2RGB );
    rgb_pyr_.buildFromImage( rgb_img, pyr_num_levels_, pyr_scale_factor_,
                             cv::DataType<cv::Vec3f>::type, false, INTERP_BILINEAR );
  }
 
  return rgb_pyr_.at( scale_index );
}

const cv::Mat & ImageStatistics :: getBlurredImage( int scale_index )
{
  boost::lock_guard<boost::recursive_mutex> lock(mutex_);
  checkScaleIndex( scale_index );
  if( blured_imgs_[scale_index].empty() )
    computeBlurredImage( scale_index );
  return blured_imgs_[scale_index];
}

const cv::Mat & ImageStatistics :: getGradientDirectionsImage( int scale_index )
{
  boost::lock_guard<boost::recursive_mutex> lock(mutex_);
  checkScaleIndex( scale_index );
  if( gradient_dir_imgs_[scale_index].empty() )
    computeGradientDirections( scale_index );
  return gradient_dir_imgs_[scale_index];
}

const cv::Mat & ImageStatistics :: getGradientMagnitudesImage( int scale_index )
{
  boost::lock_guard<boost::recursive_mutex> lock(mutex_);
  checkScaleIndex( scale_index );
  if( gradient_mag_imgs_[scale_index].empty() )
    computeGradientMagnitude( scale_index );
  return gradient_mag_imgs_[scale_index];
}

const cv::Mat & ImageStatistics :: getEigenMagnitudesImage( int scale_index )
{
  boost::lock_guard<boost::recursive_mutex> lock(mutex_);
  checkScaleIndex( scale_index );
  if( eigen_mag_imgs_[scale_index].empty() )
    computeEigenMagnitude( scale_index );
  return eigen_mag_imgs_[scale_index];
}

boost::shared_ptr< std::vector<float> > 
  ImageStatistics :: getIntensities( int scale_index, const cv::Mat &img_mask )
{
  return getStatisticsFromMask<float>( getIntensitiesImage( scale_index ), img_mask );
}

boost::shared_ptr< std::vector<float> > 
  ImageStatistics :: getBlurredIntensities( int scale_index, const cv::Mat &img_mask )
{
  return getStatisticsFromMask<float>( getBlurredImage( scale_index ), img_mask );
}

boost::shared_ptr< std::vector<float> > 
  ImageStatistics :: getGradientDirections( int scale_index, const cv::Mat &img_mask )
{
  return getStatisticsFromMask<float>( getGradientDirectionsImage( scale_index ), img_mask );
}

boost::shared_ptr< std::vector<float> > 
  ImageStatistics :: getGradientMagnitudes( int scale_index, const cv::Mat &img_mask )
{
  return getStatisticsFromMask<float>( getGradientMagnitudesImage( scale_index ), img_mask );
}
  
boost::shared_ptr< std::vector<float> > 
  ImageStatistics :: getEigenMagnitudes( int scale_index, const cv::Mat &img_mask )
{
  return getStatisticsFromMask<float>( getEigenMagnitudesImage( scale_index ), img_mask );
}

boost::shared_ptr< std::vector<float> > 
  ImageStatistics :: getIntensities( const std::vector<cv::Point2f> &coords, 
                                     int scale_index, 
                                     InterpolationType type )
{
  if( type == INTERP_NEAREST_NEIGHBOR )
    return getStatisticsFromCordsNN<float>( getIntensitiesImage( scale_index ), coords ); 
  else if( type == INTERP_BILINEAR )
    return getStatisticsFromCordsBL<float>( getIntensitiesImage( scale_index ), coords ); 
  else if( type == INTERP_BICUBIC )
    return getStatisticsFromCordsBC<float>( getIntensitiesImage( scale_index ), coords ); 
  else
    throw std::invalid_argument("Unknown interpolation type");
}

boost::shared_ptr< std::vector<float> > 
  ImageStatistics :: getBlurredIntensities( const std::vector<cv::Point2f> &coords,
                                            int scale_index, 
                                            InterpolationType type )
{  
  
  if( type == INTERP_NEAREST_NEIGHBOR )
    return getStatisticsFromCordsNN<float>( getBlurredImage( scale_index ), coords ); 
  else if( type == INTERP_BILINEAR )
    return getStatisticsFromCordsBL<float>( getBlurredImage( scale_index ), coords ); 
  else if( type == INTERP_BICUBIC )
    return getStatisticsFromCordsBC<float>( getBlurredImage( scale_index ), coords ); 
  else
    throw std::invalid_argument("Unknown interpolation type");
}

boost::shared_ptr< std::vector<float> > 
  ImageStatistics :: getGradientDirections( const std::vector<cv::Point2f> &coords, 
                                            int scale_index, 
                                            InterpolationType type )
{
  if( type == INTERP_NEAREST_NEIGHBOR )
    return getStatisticsFromCordsNN<float>( getGradientDirectionsImage( scale_index ), coords ); 
  else if( type == INTERP_BILINEAR )
    return getStatisticsFromCordsBL<float>( getGradientDirectionsImage( scale_index ), coords ); 
  else if( type == INTERP_BICUBIC )
    return getStatisticsFromCordsBC<float>( getGradientDirectionsImage( scale_index ), coords ); 
  else
    throw std::invalid_argument("Unknown interpolation type");
}

boost::shared_ptr< std::vector<float> > 
  ImageStatistics :: getGradientMagnitudes( const std::vector<cv::Point2f> &coords, 
                                            int scale_index, 
                                            InterpolationType type )
{
  if( type == INTERP_NEAREST_NEIGHBOR )
    return getStatisticsFromCordsNN<float>( getGradientMagnitudesImage( scale_index ), coords ); 
  else if( type == INTERP_BILINEAR )
    return getStatisticsFromCordsBL<float>( getGradientMagnitudesImage( scale_index ), coords ); 
  else if( type == INTERP_BICUBIC )
    return getStatisticsFromCordsBC<float>( getGradientMagnitudesImage( scale_index ), coords ); 
  else
    throw std::invalid_argument("Unknown interpolation type");
}

boost::shared_ptr< std::vector<float> > 
  ImageStatistics :: getEigenMagnitudes( const std::vector<cv::Point2f> &coords, 
                                         int scale_index, 
                                         InterpolationType type )
{
  if( type == INTERP_NEAREST_NEIGHBOR )
    return getStatisticsFromCordsNN<float>( getEigenMagnitudesImage( scale_index ), coords ); 
  else if( type == INTERP_BILINEAR )
    return getStatisticsFromCordsBL<float>( getEigenMagnitudesImage( scale_index ), coords ); 
  else if( type == INTERP_BICUBIC )
    return getStatisticsFromCordsBC<float>( getEigenMagnitudesImage( scale_index ), coords ); 
  else
    throw std::invalid_argument("Unknown interpolation type");
}
    
template <typename T> boost::shared_ptr< std::vector<float> > 
  ImageStatistics :: getStatisticsFromMask( const cv::Mat &stats_img, 
                                            const cv::Mat &img_mask ) const
{
  if( !img_mask.empty() &&
      (img_mask.depth() != cv::DataType<uchar>::type || img_mask.channels() != 1 ||
       img_mask.rows != stats_img.rows || 
       img_mask.cols != stats_img.cols) )
    throw std::invalid_argument("Invalid image mask");
  
  boost::shared_ptr< std::vector<float> > stats_vec_ptr( new std::vector<float> );
  std::vector<float> &stats_vec = *(stats_vec_ptr.get());
  stats_vec.resize( stats_img.total() );
  
  int i_pt = 0;
  if( img_mask.empty() )
  {
    for (int y = 0; y < stats_img.rows; y++)
    {
      const T *stats = stats_img.ptr<T>(y);
      for (int x = 0; x < stats_img.cols; x++, stats++)
        stats_vec[i_pt++] = float(*stats);
    }
  }
  else
  {
    for (int y = 0; y < stats_img.rows; y++)
    {
      const T *stats = stats_img.ptr<T>(y);
      const uchar *masks = img_mask.ptr<uchar>(y);
      for (int x = 0; x < stats_img.cols; x++, stats++, masks++)
        if(*masks)
          stats_vec[i_pt++] = float(*stats);
    }
  }
  stats_vec.resize(i_pt);
  
  return stats_vec_ptr;
}

template <typename T> boost::shared_ptr< std::vector<float> > 
  ImageStatistics :: getStatisticsFromCordsNN( const cv::Mat &stats_img, 
                                               const std::vector<cv::Point2f> &coords ) const
{
  boost::shared_ptr< std::vector<float> > stats_vec_ptr( new std::vector<float>);
  std::vector<float> &stats_vec = *(stats_vec_ptr.get());
  int v_size = coords.size();
  stats_vec.resize( v_size );
  int w = stats_img.cols, h = stats_img.rows;
  
  for( int i = 0; i < v_size; i++ )
  {
    const cv::Point2f &coord = coords.at(i);
    const int x = round(coord.x), y = round(coord.y);
    if( x >= 0 && y >= 0 && x < w && y < h )
      stats_vec[i] = float( stats_img.at<T> ( y, x ) );
    else
      stats_vec[i] = OUT_OF_IMAGE_VALUE;
  }
  
  return stats_vec_ptr;
}

template <typename T> boost::shared_ptr< std::vector<float> > 
  ImageStatistics :: getStatisticsFromCordsBL( const cv::Mat &stats_img, 
                                               const std::vector<cv::Point2f> &coords ) const
{
  boost::shared_ptr< std::vector<float> > stats_vec_ptr( new std::vector<float>);
  std::vector<float> &stats_vec = *(stats_vec_ptr.get());
  int v_size = coords.size();
  stats_vec.resize( v_size );
  float w = float(stats_img.cols), h = float(stats_img.rows);
  
  for( int i = 0; i < v_size; i++ )
  {
    const cv::Point2f &coord = coords.at(i);
    const float &x = coord.x, &y = coord.y;
    if( x >= 1.0f && y >= 1.0f && x < w - 1.0f && y < h - 1.0f )
      stats_vec[i] =  bilinearInterp<T>( stats_img, x, y );
    else
      stats_vec[i] = OUT_OF_IMAGE_VALUE;
  }
  
  return stats_vec_ptr;
}

template <typename T> boost::shared_ptr< std::vector<float> > 
  ImageStatistics :: getStatisticsFromCordsBC( const cv::Mat &stats_img, 
                                               const std::vector<cv::Point2f> &coords ) const
{
  boost::shared_ptr< std::vector<float> > stats_vec_ptr( new std::vector<float>);
  std::vector<float> &stats_vec = *(stats_vec_ptr.get());
  int v_size = coords.size();
  stats_vec.resize( v_size );
  float w = float(stats_img.cols), h = float(stats_img.rows);
  
  for( int i = 0; i < v_size; i++ )
  {
    const cv::Point2f &coord = coords.at(i);
    const float &x = coord.x, &y = coord.y;
    if( x >= 2.0f && y >= 2.0f && x < w - 2.0f && y < h - 2.0f )
      stats_vec[i] =  bicubicInterp<T>( stats_img, x, y );
    else
      stats_vec[i] = OUT_OF_IMAGE_VALUE;
  }
  
  return stats_vec_ptr;
}

void ImageStatistics :: checkScaleIndex( int scale_index )
{
  if( scale_index >= pyr_num_levels_ )
    throw std::invalid_argument("The scale_index is too high!");
}
