#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "cv_ext/cv_ext.h"

class IterativePnP
{
public:

  void setCamModel ( const cv_ext::PinholeCameraModel& cam_model );
  void setNumIterations( int n ){ num_iterations_ = n; };
  void fixTranslationComponent( int index, double value );

  void compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<cv::Point2f> &proj_pts, 
                double r_quat[4], double t_vec[3] );
  void compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<cv::Point2f> &proj_pts, 
                Eigen::Quaterniond &r_quat, Eigen::Vector3d &t_vec );
  void compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<cv::Point2f> &proj_pts, 
                cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec );
  
private:
  
  void compute( const std::vector<cv::Point3f> &obj_pts, const std::vector<cv::Point2f> &proj_pts );
  
  Eigen::Matrix< double, 8, 1> transf_;

  cv_ext::PinholeCameraModel cam_model_;

  int num_iterations_ = 100;
  int fixed_index_ = -1;
  double fixed_value_;
  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  
};
