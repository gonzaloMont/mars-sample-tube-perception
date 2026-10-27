#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <boost/concept_check.hpp>

// TODO Deal with quaternion normalization
namespace cv_ext
{
  template <typename _T, int _ROWS, int _COLS> inline void openCv2Eigen( const cv::Mat_<_T> &cv_mat, 
                                                                         Eigen::Matrix<_T, _ROWS, _COLS> &eigen_mat )
  {
    if( cv_mat.rows != _ROWS || cv_mat.cols != _COLS )
      throw std::invalid_argument("Input matrix has wrong size");
    
    for(int r = 0; r < _ROWS; r++)
      for(int c = 0; c < _COLS; c++)
        eigen_mat(r,c) = cv_mat(r,c);
  }
  
  template <typename _T, int _ROWS, int _COLS> inline void eigen2openCv( const Eigen::Matrix<_T, _ROWS, _COLS> &eigen_mat,
                                                                         cv::Mat_<_T> &cv_mat )
  {
    cv_mat = cv::Mat_<_T>(_ROWS,_COLS);
    
    for(int r = 0; r < _ROWS; r++)
      for(int c = 0; c < _COLS; c++)
        cv_mat(r,c) = eigen_mat(r,c);
  }
  
  template <typename _T> inline void exp2RotMat( const cv::Mat &r_vec, cv::Mat &r_mat )
  {
    if( r_vec.rows < 3 )
      throw std::invalid_argument("Input matrix has wrong size");
    
    cv::Mat_<_T>tmp_r_vec(r_vec);
    cv::Rodrigues(tmp_r_vec, r_mat );
  }

  template <typename _T> inline void exp2RotMat( const cv::Vec<_T, 3> &r_vec, cv::Mat &r_mat )
  {
    cv::Mat_<_T>tmp_r_vec(r_vec);
    cv::Rodrigues(tmp_r_vec, r_mat );
  }
  
  template <typename _T> inline void exp2RotMat( const cv::Mat &r_vec,
                                                Eigen::Matrix<_T, 3, 3> &r_mat )
  {
    if( r_vec.rows < 3 )
      throw std::invalid_argument("Input matrix has wrong size");
    
    cv::Mat_<_T>tmp_r_vec(r_vec), tmp_r_mat;
    cv::Rodrigues(tmp_r_vec, tmp_r_mat );
    openCv2Eigen<_T, 3, 3>( tmp_r_mat, r_mat );
  }

  template <typename _T> inline void exp2RotMat( const cv::Vec<_T, 3> &r_vec,
                                                Eigen::Matrix<_T, 3, 3> &r_mat)
  {
    cv::Mat_<_T>tmp_r_vec(r_vec), tmp_r_mat;
    cv::Rodrigues(tmp_r_vec, tmp_r_mat );
    openCv2Eigen<_T, 3, 3>( tmp_r_mat, r_mat );
  }
  
  template <typename _T> inline void rotMat2Exp( const cv::Mat &r_mat, cv::Mat &r_vec )
  {
    if( r_mat.rows < 3 || r_mat.cols < 3)
      throw std::invalid_argument("Input matrix has wrong size");
    
    cv::Mat_<_T>tmp_r_mat(r_mat);
    cv::Rodrigues(tmp_r_mat, r_vec );
  }

  template <typename _T> inline void rotMat2Exp( const cv::Mat &r_mat, cv::Vec<_T, 3> &r_vec )
  {
    if( r_mat.rows < 3 || r_mat.cols < 3)
      throw std::invalid_argument("Input matrix has wrong size");
    
    cv::Mat_<_T>tmp_r_mat(r_mat), tmp_r_vec;
    cv::Rodrigues(tmp_r_mat, tmp_r_vec );
    r_vec[0] = tmp_r_vec(0); r_vec[1] = tmp_r_vec(1); r_vec[2] = tmp_r_vec(2);
  }
  
  template <typename _T> inline void rotMat2Exp( const Eigen::Matrix<_T, 3, 3> &r_mat,
                                                cv::Mat &r_vec )
  {
    cv::Mat_<_T> tmp_r_mat;
    eigen2openCv<_T, 3, 3>(r_mat, tmp_r_mat);
    cv::Rodrigues(tmp_r_mat, r_vec );
  }
  
  template <typename _T> inline void rotMat2Exp( const Eigen::Matrix<_T, 3, 3> &r_mat,
                                                cv::Vec<_T, 3> &r_vec )
  {
    cv::Mat_<_T> tmp_r_vec;
    rotMat2Exp<_T>( r_mat, tmp_r_vec );
    r_vec[0] = tmp_r_vec(0); r_vec[1] = tmp_r_vec(1); r_vec[2] = tmp_r_vec(2);
  }
  
  template <typename _T> inline void exp2TransfMat( const cv::Mat &r_vec, 
                                                   const cv::Mat &t_vec,
                                                   cv::Mat &g_mat )
  {
    if( r_vec.rows < 3 || t_vec.rows < 3)
      throw std::invalid_argument("Input matrix has wrong size");
    
    cv::Mat_<_T> tmp_t_vec(t_vec), r_mat;
    exp2RotMat<_T>( r_vec, r_mat );
    
    g_mat = (cv::Mat_< _T >(4, 4) 
              << r_mat(0,0), r_mat(0,1), r_mat(0,2), tmp_t_vec(0,0),
                 r_mat(1,0), r_mat(1,1), r_mat(1,2), tmp_t_vec(1,0),
                 r_mat(2,0), r_mat(2,1), r_mat(2,2), tmp_t_vec(2,0),
                 0,          0,          0,          1);
  }

  template <typename _T> inline void exp2TransfMat( const cv::Vec<_T, 3> &r_vec, 
                                                   const cv::Vec<_T, 3> &t_vec,
                                                   cv::Mat &g_mat )
  {
    cv::Mat_<_T> tmp_r_vec(r_vec), tmp_t_vec(t_vec);
    exp2TransfMat<_T>( tmp_r_vec, tmp_t_vec, g_mat );
  }
  
  template <typename _T> inline void exp2TransfMat( const cv::Mat &r_vec,
                                                   const cv::Mat &t_vec,
                                                   Eigen::Matrix<_T, 4, 4> &g_mat )
  {
    if(r_vec.rows < 3 || t_vec.rows < 3 )
      throw std::invalid_argument("Input matrix has wrong size");
    
    cv::Mat_<_T> tmp_t_vec(t_vec), r_mat;
    exp2RotMat<_T>( r_vec, r_mat );
    
    g_mat << r_mat(0,0), r_mat(0,1), r_mat(0,2), tmp_t_vec(0,0),
             r_mat(1,0), r_mat(1,1), r_mat(1,2), tmp_t_vec(1,0),
             r_mat(2,0), r_mat(2,1), r_mat(2,2), tmp_t_vec(2,0),
             0,          0,          0,          1;
  }

  template <typename _T> inline void exp2TransfMat( const cv::Vec<_T, 3> &r_vec,
                                                   const cv::Vec<_T, 3> &t_vec,
                                                   Eigen::Matrix<_T, 4, 4> &g_mat )
  {
    cv::Mat_<_T>tmp_r_vec(r_vec), tmp_t_vec(t_vec);
    exp2TransfMat<_T>( tmp_r_vec, tmp_t_vec, g_mat );
  }
  
  template <typename _T> inline void transfMat2Exp( const cv::Mat &g_mat, 
                                                   cv::Mat &r_vec, cv::Mat &t_vec )
  {
    if( g_mat.rows < 4 || g_mat.cols < 4)
      throw std::invalid_argument("Input matrix has wrong size");
    
    cv::Mat_<_T> tmp_g_mat(g_mat);
    cv::Mat_< _T > r_mat = (cv::Mat_< _T >(3, 3) 
                        << tmp_g_mat(0,0), tmp_g_mat(0,1), tmp_g_mat(0,2),
                           tmp_g_mat(1,0), tmp_g_mat(1,1), tmp_g_mat(1,2),
                           tmp_g_mat(2,0), tmp_g_mat(2,1), tmp_g_mat(2,2));
              
    rotMat2Exp<_T>(r_mat, r_vec);
    t_vec = (cv::Mat_< _T >(3, 1)<<tmp_g_mat(0,3), tmp_g_mat(1,3), tmp_g_mat(2,3));
  }
  
  template <typename _T> inline void transfMat2Exp( const cv::Mat &g_mat, 
                                                   cv::Vec<_T, 3> &r_vec,
                                                   cv::Vec<_T, 3> &t_vec )
  {
    if( g_mat.rows < 4 || g_mat.cols < 4)
      throw std::invalid_argument("Input matrix has wrong size");
    
    cv::Mat_<_T> tmp_r_vec, tmp_t_vec;
    transfMat2Exp<_T>( g_mat, tmp_r_vec, tmp_t_vec );
    r_vec[0] = tmp_r_vec(0); r_vec[1] = tmp_r_vec(1); r_vec[2] = tmp_r_vec(2);
    t_vec[0] = tmp_t_vec(0); t_vec[1] = tmp_t_vec(1); t_vec[2] = tmp_t_vec(2);
  }  

  template <typename _T> inline void transfMat2Exp( const Eigen::Matrix<_T, 4, 4>  &g_mat,
                                                   cv::Mat &r_vec, cv::Mat &t_vec )
  {
    cv::Mat_< _T > r_mat = (cv::Mat_< _T >(3, 3) 
                        << g_mat(0,0), g_mat(0,1), g_mat(0,2),
                           g_mat(1,0), g_mat(1,1), g_mat(1,2),
                           g_mat(2,0), g_mat(2,1), g_mat(2,2));
              
    rotMat2Exp<_T>(r_mat, r_vec);
    t_vec = (cv::Mat_< _T >(3, 1)<<g_mat(0,3), g_mat(1,3), g_mat(2,3));
  }
  
  template <typename _T> inline void transfMat2Exp( const Eigen::Matrix<_T, 4, 4>  &g_mat,
                                                   cv::Vec<_T, 3> &r_vec,
                                                   cv::Vec<_T, 3> &t_vec )
  {
    cv::Mat_<_T> tmp_r_vec, tmp_t_vec;
    transfMat2Exp<_T>( g_mat, tmp_r_vec, tmp_t_vec );
    r_vec[0] = tmp_r_vec(0); r_vec[1] = tmp_r_vec(1); r_vec[2] = tmp_r_vec(2);
    t_vec[0] = tmp_t_vec(0); t_vec[1] = tmp_t_vec(1); t_vec[2] = tmp_t_vec(2);
  }
  
  template <typename _T> inline void eigenQuat2Quat( const Eigen::Quaternion< _T > &eigen_r_quat, _T r_quat[4] )
  {
    r_quat[0] = eigen_r_quat.w(); 
    r_quat[1] = eigen_r_quat.x();
    r_quat[2] = eigen_r_quat.y();
    r_quat[3] = eigen_r_quat.z();
  }
  
  template <typename _T> inline void quat2EigenQuat( const _T r_quat[4], Eigen::Quaternion< _T > &eigen_r_quat )
  {
    eigen_r_quat = Eigen::Quaternion< _T >( r_quat[0], r_quat[1], r_quat[2], r_quat[3] );
  }
}
