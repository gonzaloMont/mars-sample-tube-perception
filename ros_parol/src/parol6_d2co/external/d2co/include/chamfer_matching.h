#pragma once

#include <memory>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

#include "cv_ext/cv_ext.h"
#include "template_matching.h"
#include "distance_transforms.h"

class ChamferMatching : public TemplateMatching
{
public:
  
  ChamferMatching();
  virtual ~ChamferMatching(){};  
  
  void setupExhaustiveMatching( int max_template_pts = -1 );
    
  void  setInput( const cv::Mat &dist_map );
  
  void match( int num_best_matches, std::vector< TemplateMatch > &matches, int image_step = -1 );

protected:
  
  virtual void updateOptimizer( int idx );
  virtual double optimize();
  virtual double avgDistance( int idx );
  
private:
  
  inline float templateDist( const std::vector<cv::Point> &proj_pts );
  
  cv::Mat dist_map_, fast_dist_map_;
  
  /* Pimpl idiom */
  class Optimizer;
  std::shared_ptr< Optimizer > optimizer_ptr_;
  
  std::vector< std::vector<cv::Point> > templates_pts_;
  std::vector<cv::Rect> templates_bb_;
  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class OrientedChamferMatching : public TemplateMatching
{
public:
  
  OrientedChamferMatching();
  virtual ~OrientedChamferMatching(){};  
  
  void setNumDirections( int n );
  
  void setupExhaustiveMatching( int max_template_pts = -1 );
    
  void  setInput( const cv::Mat& dist_map, const cv::Mat& closest_dir_map );
  
  void match( int num_best_matches, std::vector< TemplateMatch > &matches, int image_step = -1 );

protected:
  
  virtual void updateOptimizer( int idx );
  virtual double optimize();
  virtual double avgDistance( int idx );
  
private:
  
  inline float templateDist( const std::vector<cv::Point> &proj_pts, const std::vector<int> &dirs );
  
  cv::Mat dist_map_, closest_dir_map_, fast_dist_map_;
//   /* Pimpl idiom */
//   class Optimizer;
//   std::shared_ptr< Optimizer > optimizer_ptr_;
  
  int num_directions_;
  float eta_direction_;

  std::vector< std::vector<cv::Point> > templates_pts_;
  std::vector< std::vector<int> > templates_i_dir_;
  std::vector<cv::Rect> templates_bb_;
  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
};

class ICPChamferMatching : public TemplateMatching
{
public:
  
  ICPChamferMatching();
  virtual ~ICPChamferMatching(){};
  
  void setupExhaustiveMatching( int max_template_pts = -1 );  
  void setInput( const cv::Mat &closest_edgels_map );
  void match( int num_best_matches, std::vector< TemplateMatch > &matches, int image_step = -1 );

protected:
  
  virtual void updateOptimizer( int idx );
  virtual double optimize();
  virtual double avgDistance( int idx );
  
private:
  
  cv::Mat closest_edgels_map_;
  std::vector<cv::Point3f> model_pts_;
  int num_icp_iterations_;
  
  /* Pimpl idiom */
  class Optimizer;
  std::shared_ptr< Optimizer > optimizer_ptr_;
  int selected_idx_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class DirectionalChamferMatching : public TemplateMatching
{
public:
  
  DirectionalChamferMatching();
  virtual ~DirectionalChamferMatching(){};  

  void setNumDirections( int n );
  int numDirections(){ return num_directions_; };
  
  void setInput( const cv_ext::ImageTensorPtr &dist_map_tensor_ptr );
  void setupExhaustiveMatching( int max_template_pts = -1 );  
  void match( int num_best_matches, std::vector< TemplateMatch > &matches, int image_step = -1 );

  // TODO integrate&remove this function
  double getDistance( const std::vector< cv::Point2f >& proj_pts,
                      const std::vector< float >& normal_directions ) const;

protected:

  virtual void updateOptimizer( int idx );
  virtual double optimize();
  virtual double avgDistance( int idx );
  
private:

  inline float templateDist( const std::vector<cv::Point> &proj_pts, const std::vector<int> &dirs );
  
  cv_ext::ImageTensorPtr dist_map_tensor_ptr_;
  cv_ext::ImageTensor fast_dist_map_tensor_;
//   DirectionalIntegralImageVectorPtr int_dist_map_tensor_ptr_;
  
  /* Pimpl idiom */
  class Optimizer;
  std::shared_ptr< Optimizer > optimizer_ptr_;
  
  int num_directions_;
  float eta_direction_;

  std::vector< std::vector<cv::Point> > templates_pts_;
  std::vector< std::vector<int> > templates_i_dir_;
  std::vector<cv::Rect> templates_bb_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class HybridDirectionalChamferMatching : public TemplateMatching
{
public:

  HybridDirectionalChamferMatching();
  virtual ~HybridDirectionalChamferMatching(){};

  void setNumDirections( int n );
  int numDirections(){ return num_directions_; };

  void setupExhaustiveMatching( int max_template_pts = -1 );
  void setInput( const cv_ext::ImageTensorPtr &dist_map_tensor_ptr,
                 const cv_ext::ImageTensorPtr &edgels_map_tensor_ptr );
  
  void match( int num_best_matches, std::vector< TemplateMatch > &matches, int image_step = -1 );

protected:

  virtual void updateOptimizer( int idx );
  virtual double optimize();
  virtual double avgDistance( int idx );

private:

  cv_ext::ImageTensorPtr dist_map_tensor_ptr_;
  cv_ext::ImageTensorPtr edgels_map_tensor_ptr_;
//   DirectionalIntegralImageVectorPtr int_dist_map_tensor_ptr_;
  std::vector<cv::Point3f> model_pts_, model_dpts_;

  /* Pimpl idiom */
  class Optimizer;
  std::shared_ptr< Optimizer > d2co_optimizer_ptr_, icp_optimizer_ptr_;

  int max_icp_iterations_;
  int num_directions_;
  float eta_direction_;
  int selected_idx_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class BidirectionalChamferMatching  : public TemplateMatching
{
public:
  
  BidirectionalChamferMatching ();
  
  virtual ~BidirectionalChamferMatching (){};  

  void setNumDirections( int n );
  int numDirections(){ return num_directions_; };
  void setupExhaustiveMatching( int max_template_pts = -1 );
  void setInput( const cv_ext::ImageTensorPtr &x_dist_map_tensor_ptr,
                 const cv_ext::ImageTensorPtr &y_dist_map_tensor_ptr );
  
  void match( int num_best_matches, std::vector< TemplateMatch > &matches, int image_step = -1 );

protected:
  
  virtual void updateOptimizer( int idx );
  virtual double optimize();
  virtual double avgDistance( int idx );
  
private:
  
  cv_ext::ImageTensorPtr x_dist_map_tensor_ptr_, y_dist_map_tensor_ptr_;
//   DirectionalIntegralImageVectorPtr int_dist_map_tensor_ptr_;
  
  /* Pimpl idiom */
  class Optimizer;
  std::shared_ptr< Optimizer > optimizer_ptr_;
  
  int num_directions_;
  float eta_direction_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d> > PoseVec;
typedef std::vector<cv_ext::ImageTensorPtr, Eigen::aligned_allocator<cv_ext::ImageTensorPtr> > ImageTensorPtrVec;
struct MultiViewsInput
{
  RasterObjectModel3DPtr model_ptr;
  PoseVec views;
  ImageTensorPtrVec dist_map_tensor_ptr_vec;
  cv_ext::PinholeCameraModel camera_model;
};

typedef std::vector<MultiViewsInput, Eigen::aligned_allocator<MultiViewsInput> > MultiViewsInputVec;

class MultiViewsDirectionalChamferMatching  : public TemplateMatching
{
public:
  MultiViewsDirectionalChamferMatching ();
  virtual ~MultiViewsDirectionalChamferMatching (){};

  void setNumDirections( int n );
  int numDirections(){ return num_directions_; };
  void setupExhaustiveMatching( int max_template_pts = -1 ); 
  void setInput( const MultiViewsInputVec& input );

protected:

  virtual void updateOptimizer( int idx );
  virtual double optimize();
  virtual double avgDistance( int idx );

private:
  
  MultiViewsInputVec input_;

  /* Pimpl idiom */
  class Optimizer;
  std::shared_ptr< Optimizer > optimizer_ptr_;

  int num_directions_;
  float eta_direction_;
  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class ICPDirectionalChamferMatching : public TemplateMatching
{
public:
  
  ICPDirectionalChamferMatching();
    
  virtual ~ICPDirectionalChamferMatching(){};  

  void setNumDirections( int n );
  int numDirections(){ return num_directions_; };
  void setupExhaustiveMatching( int max_template_pts = -1 ); 
  void setInput( const cv_ext::ImageTensorPtr &edgels_map_tensor_ptr );
  
  void match( int num_best_matches, std::vector< TemplateMatch > &matches, int image_step = -1 );

protected:
 
  virtual void updateOptimizer( int idx );
  virtual double optimize();
  virtual double avgDistance( int idx );
  
private:
  
  cv_ext::ImageTensorPtr edgels_map_tensor_ptr_;
  std::vector<cv::Point3f> model_pts_;
  int num_icp_iterations_;
  
  /* Pimpl idiom */
  class Optimizer;
  std::shared_ptr< Optimizer > optimizer_ptr_;
  
  int num_directions_;
  float eta_direction_;
  int selected_idx_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
