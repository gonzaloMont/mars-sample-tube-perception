#include "cv_ext/pinhole_camera_model.h"
#include "cv_ext/serialization.h"

using namespace std;
using namespace cv;
using namespace cv_ext;

PinholeCameraModel::PinholeCameraModel( const Mat &camera_matrix, int img_width, int img_height,
                                        const Mat &dist_coeff ) :
  orig_img_width_(img_width), 
  orig_img_height_(img_height)
{
  if( img_width <= 0 || img_height <= 0 )
    throw invalid_argument("PinholeCameraModel::PinholeCameraModel() img_width and img_height must be > 0");
    
  if( camera_matrix.channels() != 1 || 
      camera_matrix.rows != 3 || camera_matrix.cols != 3 || 
      ( !dist_coeff.empty() && (dist_coeff.channels() != 1 ||
      ( dist_coeff.rows != 1 && dist_coeff.cols != 1 ) || 
      ( dist_coeff.rows < 4 && dist_coeff.cols < 4 ) ) ) )
    throw invalid_argument("PinholeCameraModel::PinholeCameraModel() camera_matrix or dist_coeff wrong size");

  Mat_<double>tmp_cam_mat(camera_matrix);
  
  orig_fx_ = tmp_cam_mat(0,0); 
  orig_fy_ = tmp_cam_mat(1,1);
  orig_cx_ = tmp_cam_mat(0,2); 
  orig_cy_ = tmp_cam_mat(1,2);
  
  has_dist_coeff_ = !dist_coeff.empty();
  dist_px_ = dist_py_ = dist_k0_ = dist_k1_ = dist_k2_ = dist_k3_ = dist_k4_ = dist_k5_ = 0;
  
  if( has_dist_coeff_ )
  {
    Mat_<double>tmp_dist_coeff(dist_coeff);
    
    if( tmp_dist_coeff.rows < tmp_dist_coeff.cols )
      cv::transpose(tmp_dist_coeff, tmp_dist_coeff);
    
    dist_k0_ = tmp_dist_coeff(0,0);
    dist_k1_ = tmp_dist_coeff(1,0); 
    dist_px_ = tmp_dist_coeff(2,0);
    dist_py_ = tmp_dist_coeff(3,0);
    
    if( tmp_dist_coeff.rows > 4 )
      dist_k2_ = tmp_dist_coeff(4,0);
    if( tmp_dist_coeff.rows > 5 )
      dist_k3_ = tmp_dist_coeff(5,0);
    if( tmp_dist_coeff.rows > 6 )
      dist_k4_ = tmp_dist_coeff(6,0);
    if( tmp_dist_coeff.rows > 7 )
      dist_k5_ = tmp_dist_coeff(7,0);      
  }
  
  reset();
}

void PinholeCameraModel::readFromFile ( string filename )
{
  string yml_filename = generateYAMLFilename(filename);
  FileStorage fs(yml_filename, FileStorage::READ);

  fs["img_width"]>>orig_img_width_;
  fs["img_height"]>>orig_img_height_;

  fs["fx"]>>orig_fx_;
  fs["fy"]>>orig_fy_;
  fs["cx"]>>orig_cx_;
  fs["cy"]>>orig_cy_;
  
  int has_dist_coeff;
  fs["has_dist_coeff"]>>has_dist_coeff;
  has_dist_coeff_ = (has_dist_coeff != 0);
  dist_px_ = dist_py_ = dist_k0_ = dist_k1_ = dist_k2_ = dist_k3_ = dist_k4_ = dist_k5_ = 0;
  
  if(has_dist_coeff_)
  {
    fs["dist_px"]>>dist_px_;
    fs["dist_py"]>>dist_py_;
    fs["dist_k0"]>>dist_k0_;
    fs["dist_k1"]>>dist_k1_;
    fs["dist_k2"]>>dist_k2_;
    fs["dist_k3"]>>dist_k3_;
    fs["dist_k4"]>>dist_k4_;
    fs["dist_k5"]>>dist_k5_;
  }

  fs.release();
  
  reset();
}

void PinholeCameraModel::writeToFile ( string filename )
{
  string yml_filename = generateYAMLFilename(filename);
  FileStorage fs(yml_filename, FileStorage::WRITE);

  fs<<"img_width"<<orig_img_width_;
  fs<<"img_height"<<orig_img_height_;

  fs<<"fx"<<orig_fx_;
  fs<<"fy"<<orig_fy_;
  fs<<"cx"<<orig_cx_;
  fs<<"cy"<<orig_cy_;

  fs<<"has_dist_coeff"<<((has_dist_coeff_)?1:0);

  if(has_dist_coeff_)
  {
    fs<<"dist_px"<<dist_px_;
    fs<<"dist_py"<<dist_py_;
    fs<<"dist_k0"<<dist_k0_;
    fs<<"dist_k1"<<dist_k1_;
    fs<<"dist_k2"<<dist_k2_;
    fs<<"dist_k3"<<dist_k3_;
    fs<<"dist_k4"<<dist_k4_;
    fs<<"dist_k5"<<dist_k5_;
  }

  fs.release();
}

void PinholeCameraModel::setSizeScaleFactor ( double scale_factor ) 
{ 
  size_scale_factor_ = scale_factor;
  if( size_scale_factor_ <= 0.0)
    throw invalid_argument("PinholeCameraModel::setSizeScaleFactor() The scale factor should be > 0");
  
  updateCurrentParamters();
}

void PinholeCameraModel::setRegionOfInterest ( Rect& roi )
{
  if( roi.x < 0 || roi.x >= orig_img_width_ || roi.y < 0 || roi.y >= orig_img_height_ ||
      roi.width > orig_img_width_ - roi.x || roi.height > orig_img_height_ - roi.y )
    throw invalid_argument("PinholeCameraModel::setRegionOfInterest() The region of interest should be inside the image ");  
  
  orig_roi_ = roi;
  updateCurrentParamters();
}

void PinholeCameraModel::enableRegionOfInterest ( bool enable )
{
  roi_enabled_ = enable;
  updateCurrentParamters();
}

void PinholeCameraModel::updateCurrentParamters() 
{  
  fx_ = orig_fx_/size_scale_factor_; 
  fy_ = orig_fy_/size_scale_factor_;
  cx_ = orig_cx_/size_scale_factor_; 
  cy_ = orig_cy_/size_scale_factor_;
  
  inv_fx_ = 1.0/fx_;
  inv_fy_ = 1.0/fy_;
  
  roi_.x = orig_roi_.x/size_scale_factor_;
  roi_.y = orig_roi_.y/size_scale_factor_;
  roi_.width = orig_roi_.width/size_scale_factor_;
  roi_.height = orig_roi_.height/size_scale_factor_;
  
  if( roi_enabled_ )
  {
    cx_ -= roi_.x;
    cy_ -= roi_.y;
    img_width_ = roi_.width;
    img_height_ = roi_.height;
  }
  else
  {
    img_width_ = orig_img_width_/size_scale_factor_;
    img_height_ = orig_img_height_/size_scale_factor_;    
  }
}

void PinholeCameraModel::reset()
{
  size_scale_factor_ = 1.0;
  orig_roi_ = Rect(0,0,orig_img_width_,orig_img_height_);
  roi_enabled_ = false;
  term_epsilon_ = 1e-7;
  updateCurrentParamters();
}

