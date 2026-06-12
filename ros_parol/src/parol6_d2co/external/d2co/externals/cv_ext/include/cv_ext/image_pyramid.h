#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

#include "cv_ext/types.h"

namespace cv_ext
{
/**
 * @brief ImagePyramid is an utility class used to produce a gaussian or a general image pyramid 
 *        from an input image
 */
class ImagePyramid
{
public:

  /**
   * @brief Default constructor, it does nothing
   */  
  ImagePyramid();

  /**
   * @brief Builds a gaussian image pyramid from an input image
   *
   * @param[in] img Input image used to build the pyramid
   * @param[in] pyr_num_levels Numbers of pyramid levels, it should be greater or equal than 1.
   * @param[in] pyr_img_type  Desired pyramid's images type (i.e., the depth since the number of channels 
   *                          are the same as the input). if pyr_img_type is negative, the pyramid's images
   *                          will have the same type as the input image
   * @param[in] deep_copy If true, make a deep copy of the input image as level 0 of the pyramid
   * 
   * A gaussian pyramid is built smoothing with a gaussian filter the input image and then subsampling
   * by a scale factor of 2 the smoothed image
   * 
   * \note The pyramid is not precomputed, each level is computed on demand. If you want to pre-compute the whole image,
   * use precomputeAllLevels()
   */  
  void buildGaussianPyrFromImage( const cv::Mat &img, int pyr_num_levels, 
                                  int pyr_img_type = -1, bool deep_copy = true );

  /**
   * @brief Builds a general image pyramid with user defined scale factor from an input image
   *
   * @param[in] img Input image used to build the pyramid
   * @param[in] pyr_num_levels Numbers of pyramid levels, it should be greater or equal than 1.
   * @param[in] pyr_scale_factor Desired scale factor between two consecutive levels
   * @param[in] pyr_img_type  Desired pyramid's images type (i.e., the depth since the number of channels 
   *                          are the same as the input). if pyr_img_type is negative, the pyramid's images
   *                          will have the same type as the input image
   * @param[in] deep_copy If true, make a deep copy of the input image as level 0 of the pyramid
   * @param[in] interp_type Interpolation type used to resize the image between two consecutive levels
   * 
   * A general image pyramid is built resizing the input image with the given scale factor and the given interpolation type
   * 
   * \note The pyramid is not precomputed, each level is computed on demand. If you want to pre-compute the whole image,
   * use precomputeAllLevels()
   */   
  void buildFromImage( const cv::Mat &img, int pyr_num_levels, double pyr_scale_factor, int pyr_img_type = -1, 
                       bool deep_copy = true,  InterpolationType interp_type = cv_ext::INTERP_NEAREST_NEIGHBOR );
  
  /**
   * @brief Utility function used to precompute the whole pyramid
   */
  void precomputeAllLevels();
  
  /**
   * @brief Provides the numbers of pyramid levels
   */  
  int numLevels() { return num_levels_; };
  
  /**
   * @brief Provides the scale factor between two consecutive pyramid levels
   */  
  double scaleFactor() { return scale_factor_; };

  /**
   * @brief Provides a reference of the image associated to a desired pyramid's level
   * 
   * @param[in] pyr_level Desired pyramid's level
   * 
   * \note If not already precomputed, the image is computed on demand. See also precomputeAllLevels()
   */  
  cv::Mat& operator[]( int pyr_level );  
  
  /**
   * @brief Provides a reference of the image associated to a desired pyramid's level
   * 
   * @param[in] pyr_level Desired pyramid's level
   * 
   * \note If not already precomputed, the image is computed on demand. See also precomputeAllLevels()
   */  
  const cv::Mat& at( int pyr_level );
  
  /**
   * @brief Provides the scale associated to a desired pyramid's level
   * 
   * @param[in] pyr_level Desired pyramid's level
   * 
   * \note If not already precomputed, the image is computed on demand. See also precomputeAllLevels()
   */    
  double getScale( int pyr_level ) const;
  
private:
  
  ImagePyramid ( const ImagePyramid& other );
  ImagePyramid& operator= ( const ImagePyramid& other );
  
  void initPyr( const cv::Mat& img, int pyr_img_type, bool deep_copy );
  void computeScaleLevel( int scale_index );
  
  int num_levels_;
  double scale_factor_;

  std::vector<cv::Mat> pyr_imgs_;
  std::vector<bool> init_;
  std::vector<double> scales_; 
  bool gaussian_pyr_;
  InterpolationType interp_type_;
};

}
