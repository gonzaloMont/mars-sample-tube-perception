#pragma once

#include <memory>
#include <vector>

#include "cv_ext/cv_ext.h"
#include "edge_detectors.h"
#include <opencv2/imgproc.hpp>

typedef std::vector< cv_ext::DirectionalIntegralImage<float,double> > DirectionalIntegralImageVector;
typedef std::shared_ptr< DirectionalIntegralImageVector > DirectionalIntegralImageVectorPtr;

class DistanceTransform
{
public:
   //! @brief Default (empty) constructor: all the private members are set to their default values.
  DistanceTransform(){};
  ~DistanceTransform(){};

  //! @brief Provide the distance type for distance transform, see the OpenCV enum cv::DistanceTypes
  int distType(){ return dist_type_; };
  //! @brief Provide the size of the distance transform mask, see the OpenCV enum cv::DistanceTransformMasks
  int maskSize(){ return mask_size_; };

  //! @brief Provide the distance threshold used in the distance transform, see setDistThreshold()
  float distThreshold(){ return dist_thresh_; };
  //! @brief Return true is the algorithms are parallized
  bool isParallelismEnabled() const { return parallelism_enabled_; };

  /**
   * @brief Set the distance type for distance transform 
   * 
   * @param[in] distanceType Type of distance, see the OpenCV enum cv::DistanceTypes
   * 
   * The default distance Transform type is CV_DIST_L2 (euclidean distance)
   */
  void setDistType( int type ){ dist_type_ = type; };
  
  
  /**
   * @brief Set the size of the distance transform mask, see the OpenCV enum cv::DistanceTransformMasks
   * 
   * @param[in] size Size of the mask, see cv::DistanceTransformMasks
   * 
   * The default mask size is cv::DIST_MASK_5
   * 
   * \note In case of the DIST_L1 or DIST_C distance type, the parameter should 
   *       be set to  3 because a 3×3 mask gives the same result as 5×5 
   *       or any larger aperture.
   */
  void setMaskSize( int size ){ mask_size_ = size; };

  /**
   * @brief Set the distance threshold used in the distance transform
   * 
   * @param[in] thresh Threshold in pixels
   * 
   * Pixel with distance to the closes edgel greater than thresh , are set to thresh.
   */  
  void setDistThreshold( float thresh ){ dist_thresh_ = thresh; };
  
  /**
   * @brief Enable/disable parallelism 
   * 
   * @param[in] enable If true, some algorithms are run in a parallel region,
   *                   otherwise they run as a single thread
   */  
  void enableParallelism( bool enable ){ parallelism_enabled_ = enable; };

  void setEdgeDetector( EdgeDetectorUniquePtr ed );

  void computeDistanceMap( const cv::Mat &src_img, cv::Mat &dist_map );
  void computeDistanceMap( const cv::Mat &src_img, cv::Mat &dist_map,
                           cv::Mat &closest_edgels_map );  
  void computeDistDirMap( const cv::Mat &src_img, cv::Mat &dist_map,
                           cv::Mat &closest_dir_map, int num_directions );
  void computeDistanceMapTensor( const cv::Mat &src_img,
                                 cv_ext::ImageTensorPtr &dist_map_tensor_ptr,
                                 int num_directions, double lambda,
                                 bool smooth_tensor = true );
  void computeDistanceMapTensor( const cv::Mat &src_img,
                                 cv_ext::ImageTensorPtr &dist_map_tensor_ptr,
                                 cv_ext::ImageTensorPtr &edgels_map_tensor_ptr,
                                 int num_directions, double lambda,
                                 bool smooth_tensor = true );
  void computeDistanceMapTensor( const cv::Mat &src_img,
                                 cv_ext::ImageTensorPtr &dist_map_tensor_ptr,
                                 cv_ext::ImageTensorPtr &x_dist_map_tensor_ptr,
                                 cv_ext::ImageTensorPtr &y_dist_map_tensor_ptr,
                                 cv_ext::ImageTensorPtr &edgels_map_tensor_ptr,
                                 int num_directions, double lambda,
                                 bool smooth_tensor = true );
private:

  void distanceMapClosestEdgels( const cv::Mat &edge_map, cv::Mat &dist_map,
                                 cv::Mat &closest_edgels_map );
  void distanceMapClosestEdgelsDir( const cv::Mat &edge_map, const cv::Mat &edge_dir_map,
                                    cv::Mat &dist_map, cv::Mat &closest_edgels_dir_map );

  void smoothTensor( cv_ext::ImageTensorPtr &dist_map_tensor_ptr );

  int dist_type_ = cv::DIST_L2, mask_size_ = 5;
  float dist_thresh_ = -1;
  EdgeDetectorUniquePtr edge_detector_ = EdgeDetectorUniquePtr(new LSDEdgeDetector());
  bool parallelism_enabled_ = false;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};