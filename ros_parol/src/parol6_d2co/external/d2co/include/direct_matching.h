#pragma once

#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>

#include "cv_ext/cv_ext.h"
#include "raster_object_model.h"
#include "template_matching.h"

struct ScaledImage { cv::Mat img; double scale; };
typedef std::shared_ptr< std::vector< ScaledImage > > ScaledImagesListPtr;

void computeGradientMagnitudePyr( const cv::Mat& src_img, ScaledImagesListPtr &g_mag_pyr_ptr,
                                  unsigned int pyr_levels, double smooth_std = 1.0 );

class DirectMatching : public TemplateMatching
{
public:
  
  DirectMatching();
  virtual ~DirectMatching(){};  

  void setInput( const ScaledImagesListPtr &img_pyr_ptr );
  
  virtual void match( int num_best_matches, std::vector< TemplateMatch > &matches, int image_step = -1 );

protected:
  
  virtual void setupExhaustiveMatching();
  virtual void updateOptimizer( int idx );
  virtual double optimize();
  virtual double avgDistance( int idx );
  
private:
  
  ScaledImagesListPtr imgs_list_ptr_;
  std::vector<cv_ext::PinholeCameraModel> scaled_cam_models_;
  
  /* Pimpl idiom */
  class Optimizer;
  std::vector< std::shared_ptr< Optimizer > > optimizer_pyr_ptr_;
};
