#pragma once

#include <boost/shared_ptr.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include "cv_ext/image_pyramid.h"

/* TODO
 * 
 * Optimize Eigenvalues and Eigenvectors computation
 */
namespace cv_ext
{
/** @brief Shared pointer typedef */
typedef boost::shared_ptr< class ImageStatistics > ImageStatisticsPtr;

class ImageStatistics
{
public:
  
  static ImageStatisticsPtr createImageStatistics( const cv::Mat &img, bool deep_copy = true,
                                                   unsigned int pyr_levels = 1, double pyr_scale_factor = 2.0,
                                                   double gaussian_stddev = 1.0, bool bgr_color_order = true )
  {
    return ImageStatisticsPtr ( new ImageStatistics ( img, deep_copy, pyr_levels, pyr_scale_factor,
                                                      gaussian_stddev, bgr_color_order ) );
  };
  
  virtual ~ImageStatistics(){};
  
  int numPyrLevels() const { return pyr_num_levels_; };
  double pyrScaleFactor() const { return pyr_scale_factor_; };
  double gaussianBlurStdDev()const { return gaussian_stddev_; };
  bool useBGRColorOrder() const { return bgr_color_order_; };
  double getPyrScale( int scale_index ) const { return gl_pyr_.getScale( scale_index ); };
    
  inline bool outOfImage( float &val ){ return (val == OUT_OF_IMAGE_VALUE); };
    
  const cv::Mat &getIntensitiesImage( int scale_index = 0 );
  const cv::Mat &getColorsImage( int scale_index = 0  );
  const cv::Mat &getBlurredImage( int scale_index = 0  );
  const cv::Mat &getGradientDirectionsImage( int scale_index = 0 );
  const cv::Mat &getGradientMagnitudesImage( int scale_index = 0 );
  const cv::Mat &getEigenMagnitudesImage( int scale_index = 0 );
  
  boost::shared_ptr< std::vector<float> > 
    getIntensities( int scale_index = 0, const cv::Mat &img_mask = cv::Mat() );
  boost::shared_ptr< std::vector<float> > 
    getBlurredIntensities( int scale_index = 0, const cv::Mat &img_mask = cv::Mat() );
  boost::shared_ptr< std::vector<float> > 
    getGradientDirections( int scale_index = 0, const cv::Mat &img_mask = cv::Mat() );
  boost::shared_ptr< std::vector<float> > 
    getGradientMagnitudes( int scale_index = 0, const cv::Mat &img_mask = cv::Mat() );
  boost::shared_ptr< std::vector<float> > 
    getEigenMagnitudes( int scale_index = 0, const cv::Mat &img_mask = cv::Mat() );
  
  boost::shared_ptr< std::vector<float> > 
    getIntensities( const std::vector<cv::Point2f> &coords, int scale_index = 0, 
                    InterpolationType type = INTERP_BILINEAR );
  boost::shared_ptr< std::vector<float> > 
    getBlurredIntensities( const std::vector<cv::Point2f> &coords, int scale_index = 0, 
                           InterpolationType type = INTERP_BILINEAR );
  boost::shared_ptr< std::vector<float> > 
    getGradientDirections( const std::vector<cv::Point2f> &coords, int scale_index = 0, 
                           InterpolationType type = INTERP_BILINEAR  );
  boost::shared_ptr< std::vector<float> > 
    getGradientMagnitudes( const std::vector<cv::Point2f> &coords, int scale_index = 0, 
                           InterpolationType type = INTERP_BILINEAR  );
  boost::shared_ptr< std::vector<float> > 
    getEigenMagnitudes( const std::vector<cv::Point2f> &coords, int scale_index = 0, 
                        InterpolationType type = INTERP_BILINEAR  );
    
private:

  const float OUT_OF_IMAGE_VALUE;
    
  ImageStatistics( const cv::Mat &img, bool deep_copy, 
                   unsigned int pyr_levels, double pyr_scale_factor,
                   double gaussian_stddev, bool bgr_color_order );
  
  void setImage( const cv::Mat &img, bool deep_copy );
  void computeBlurredImage( int scale_index );
  void computeGradientImages( int scale_index );
  void computeGradientDirections( int scale_index );
  void computeGradientMagnitude( int scale_index );
  void computeEigenMagnitude( int scale_index );
    
  template <typename T> boost::shared_ptr< std::vector<float> > 
    getStatisticsFromMask( const cv::Mat &stats_img, const cv::Mat &img_mask ) const;
  template <typename T> boost::shared_ptr< std::vector<float> > 
   getStatisticsFromCordsNN( const cv::Mat &stats_img, 
                             const std::vector<cv::Point2f> &coords ) const;
  template <typename T> boost::shared_ptr< std::vector<float> > 
   getStatisticsFromCordsBL( const cv::Mat &stats_img, 
                             const std::vector<cv::Point2f> &coords ) const;
  template <typename T> boost::shared_ptr< std::vector<float> > 
   getStatisticsFromCordsBC( const cv::Mat &stats_img, 
                             const std::vector<cv::Point2f> &coords ) const;

                             
  void checkScaleIndex( int scale_index );
  
  boost::recursive_mutex mutex_;
  
  int pyr_num_levels_;
  double pyr_scale_factor_;
  double gaussian_stddev_;
  bool bgr_color_order_;
  
  ImagePyramid gl_pyr_;
  ImagePyramid rgb_pyr_;
  
  std::vector<cv::Mat> blured_imgs_;
  std::vector<cv::Mat> dx_imgs_, dy_imgs_;
  std::vector<cv::Mat> eigen_imgs_;
  std::vector<cv::Mat> eigen_mag_imgs_;
  std::vector<cv::Mat> gradient_dir_imgs_;
  std::vector<cv::Mat> gradient_mag_imgs_;
  
};

}
