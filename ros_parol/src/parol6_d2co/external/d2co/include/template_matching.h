#pragma once

#include <utility>
#include "cv_ext/pinhole_camera_model.h"
#include "raster_object_model.h"
#include "raster_object_model3D.h"

struct TemplateMatch
{
  TemplateMatch(){};
  TemplateMatch( int id,  double dist, const Eigen::Quaterniond &r, 
                 const Eigen::Vector3d &t, const cv::Point &offset ) :
    id(id),
    distance(dist),
    img_offset(offset),
    r_quat(r),
    t_vec(t)
    {};

  int id;
  double distance;
  cv::Point img_offset;
  Eigen::Quaterniond r_quat;
  Eigen::Vector3d t_vec;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


class TemplateMatching
{
public:
  
  TemplateMatching();
  virtual ~TemplateMatching(){};

  void setTemplateModel( const RasterObjectModelPtr &model_ptr );
    
  void enableVerbouseMode( bool enabled ) { verbouse_mode_ = enabled; };
  void setOptimizerNumIterations (int n ) { optimizer_num_iterations_ = n; };

  double getAvgDistance( const double r_quat[4], const double t_vec[3] );
  double getAvgDistance( const Eigen::Quaterniond &r_quat, const Eigen::Vector3d &t_vec );
  double getAvgDistance( const cv::Mat_<double> &r_vec, const cv::Mat_<double> &t_vec );
  double getAvgDistance( int idx );

  double refinePosition( double r_quat[4], double t_vec[3] );
  double refinePosition( Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec );
  double refinePosition( cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec );

  double refinePosition( int idx, double r_quat[4], double t_vec[3] );
  double refinePosition( int idx, Eigen::Quaterniond& r_quat, Eigen::Vector3d& t_vec );
  double refinePosition( int idx, cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec );
  
  /**
   * @brief Enable/disable parallelism for some/all algorithm
   *
   * @param enable If true, some algorithms are run in a parallel region,
   *               otherwise it is run as a single thread
   * 
   * \note If parallelism is enabled, some results may slightly change due to the data processing order
   */
  void enableParallelism ( bool enable )
  {
    parallelism_enabled_ = enable;
  };

  /**
   * @brief Return true is the algorithms are run in a parallel region
   */
  bool isParallelismEnabled() const
  {
    return parallelism_enabled_;
  };
  
  
  /**
   * @brief Normalize an input match by setting its actual 6D position and no image offset
   *
   * @param m An intpu TemplateMatch
   * 
   * Given an input TemplateMatch instance, this function estimates the actual location of the object 
   * as if it had no image offset. The r_quat and t_vec are set accordling, while img_offset is set to (0,0). 
   * If the match has no image offset (i.e., img_offset is (0,0)) r_quat and t_vec are not changed.
   */  
  void normalizeMatch( TemplateMatch &m );
    
protected:

  virtual void updateOptimizer( int idx ) = 0;
  virtual double optimize() = 0;
  virtual double avgDistance( int idx ) = 0;
  
  void setPos( const double r_quat[4], const double t_vec[3] );
  void setPos( const Eigen::Quaterniond &r_quat, const Eigen::Vector3d &t_vec );
  void setPos( const cv::Mat_<double> &r_vec, const cv::Mat_<double> &t_vec );

  void getPos( double r_quat[4], double t_vec[3] ) const;
  void getPos( Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec ) const;
  void getPos( cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec ) const;
  
  cv_ext::PinholeCameraModel cam_model_;
  Eigen::Matrix< double, 8, 1> transf_;
  RasterObjectModelPtr model_ptr_;
  
  int optimizer_num_iterations_;
  double verbouse_mode_;
  bool parallelism_enabled_;
  
private:

  TemplateMatching( const TemplateMatching &other );
  TemplateMatching& operator=( const TemplateMatching &other );
  
  bool update_optimizer_;
  
  //! Camera matrix and distortion coefficients: redundant information used for efficiency issuess
  cv::Mat camera_matrix_, dist_coeff_;
  
  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
