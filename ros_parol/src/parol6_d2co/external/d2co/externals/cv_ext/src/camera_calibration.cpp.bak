#include "cv_ext/camera_calibration.h"
#include "cv_ext/image_pyramid.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <uuid/uuid.h>

using namespace cv;
using namespace std;
using namespace cv_ext;
using namespace boost::filesystem;

static const std::string cache_folder_basename ( ".camera_calibration_cache_" );
static const std::string cache_file_basename ( "camera_calibration_file_" );

static void calcBoardCornerPositions ( vector<Point3f>& object_points, Size board_size, float square_size )
{
  for ( int i = 0; i < board_size.height; ++i )
    for ( int j = 0; j < board_size.width; ++j )
    {
      object_points.push_back ( Point3f ( float ( j*square_size ), float ( i*square_size ), 0 ) );
    }
}

static bool findChessboardCornersPyr ( Mat img, Size board_size, vector<Point2f> &img_pts,
                                       int pyr_num_levels, bool has_white_circle )
{
  bool corners_found = false;
  ImagePyramid img_pyr;
  img_pyr.buildGaussianPyrFromImage ( img, pyr_num_levels, -1, false );
  if ( findChessboardCorners ( img_pyr[pyr_num_levels - 1], board_size, img_pts,
                               CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE ) )
  {
    corners_found = true;

    if ( has_white_circle )
    {
      // Look for the white circle in the pattern corresponding to the origin of the reference frame
      vector<Point2f> square_pts ( 4 ), square_img_pts ( 4 ), test_pts ( 5 ), test_img_pts ( 5 );

      square_pts[0] = Point2f ( 0.0f, 0.0f );
      square_pts[1] = Point2f ( 1.0f, 0.0f );
      square_pts[2] = Point2f ( 0.0f, 1.0f );
      square_pts[3] = Point2f ( 1.0f, 1.0f );

      square_img_pts[0] = img_pts[0];
      square_img_pts[1] = img_pts[1];
      square_img_pts[2] = img_pts[board_size.width];
      square_img_pts[3] = img_pts[board_size.width + 1];

      Mat h1 = getPerspectiveTransform ( square_pts, square_img_pts );

      int n_pts = img_pts.size();
      square_img_pts[0] = img_pts[n_pts - board_size.width - 2];
      square_img_pts[1] = img_pts[n_pts - board_size.width - 1];
      square_img_pts[2] = img_pts[n_pts - 2];
      square_img_pts[3] = img_pts[n_pts - 1];

      Mat h2 = getPerspectiveTransform ( square_pts, square_img_pts );

      test_pts[0] = Point2f ( 0.5f, 0.5f );
      test_pts[1] = Point2f ( 0.6f, 0.5f );
      test_pts[2] = Point2f ( 0.4f, 0.5f );
      test_pts[3] = Point2f ( 0.5f, 0.6f );
      test_pts[4] = Point2f ( 0.5f, 0.4f );

      Mat debug_img;
      cvtColor ( img_pyr[pyr_num_levels - 1], debug_img, cv::COLOR_GRAY2BGR );

      perspectiveTransform ( test_pts, test_img_pts, h1 );

      float score1 = 0.0f, score2 = 0.0f;

      for ( auto &p : test_img_pts )
      {
        score1 += img_pyr[pyr_num_levels - 1].at<uchar> ( p.y, p.x );
      }

      perspectiveTransform ( test_pts, test_img_pts, h2 );

      for ( auto &p : test_img_pts )
      {
        score2 += img_pyr[pyr_num_levels - 1].at<uchar> ( p.y, p.x );
      }

      if ( score2 > score1 )
      {
        std::reverse ( img_pts.begin(),img_pts.end() );
      }

    }
    cornerSubPix ( img_pyr[pyr_num_levels - 1], img_pts, Size ( 11,11 ),
                   Size ( -1,-1 ), TermCriteria ( TermCriteria::EPS+TermCriteria::COUNT, 30, 0.1 ) );

    for ( int l = pyr_num_levels - 2; l >= 0; l-- )
    {
      for ( auto &p : img_pts )
      {
        p *= 2.0f;
      }
      cornerSubPix ( img_pyr[l], img_pts, Size ( 11,11 ),
                     Size ( -1,-1 ), TermCriteria ( TermCriteria::EPS+TermCriteria::COUNT, 30, 0.1 ) );
    }
  }
  return corners_found;
}

CameraCalibrationBase::CameraCalibrationBase ( int cache_max_size, std::string cache_folder ) :
  cache_max_size_ ( cache_max_size ),
  cache_folder_ ( cache_folder )
{

  if ( cache_max_size_ < 1 )
  {
    cache_max_size_ = 1;
  }

  cache_folder_ += "/";
  cache_folder_ += cache_folder_basename;
  uuid_t uuid;
  char uuid_str[37];
  uuid_generate ( uuid );
  uuid_unparse_lower ( uuid, uuid_str );
  cache_folder_ += uuid_str;

  path cf_path ( cache_folder_ );
  if ( !create_directory ( cf_path ) )
  {
    throw std::runtime_error ( "CameraCalibrationBase::CameraCalibrationBase() : Failed to create the cache folder" );
  }
}

CameraCalibrationBase::~CameraCalibrationBase()
{
  clearDiskCache();
  path cf_path ( cache_folder_ );
  remove ( cf_path );
}

void CameraCalibrationBase::clearDiskCache()
{
  path cf_path ( cache_folder_ );
  if ( exists ( cf_path ) && is_directory ( cf_path ) )
  {
    // cf_path actually exist: delete all the previously cached files inside
    for ( directory_iterator d_iter ( cf_path ); d_iter!=directory_iterator(); d_iter++ )
    {
      if ( boost::starts_with ( d_iter->path().filename().string(), cache_file_basename ) )
      {
        remove ( d_iter->path() );
      }
    }
  }
}

void CameraCalibrationBase::cachePutImage ( int i, cv::Mat &img )
{
  images_cache_.push_front ( pair<int,Mat> ( i,img ) );
  if ( int ( images_cache_.size() ) > cache_max_size_ )
  {
    // Save to disk the oldest image in the cache
    imwrite ( getCacheFilename ( images_cache_.back().first ), images_cache_.back().second );
    // .. and remove it from memory
    images_cache_.pop_back();
  }
}

Mat CameraCalibrationBase::cacheGetImage ( int i )
{
  // Look for the image
  auto it = images_cache_.begin();
  for ( ; it != images_cache_.end(); it++ )
    if ( it->first == i )
      break;

  Mat img;

  if ( it == images_cache_.begin() )
  {
    // Cache contains the i-th image and it is the newest element: already in the right position!
    img = it->second;
  }
  else if ( it != images_cache_.end() )
  {
    // Cache contains the i-th image: move it to front
    img = it->second;
    images_cache_.splice ( images_cache_.begin(), images_cache_, it, std::next ( it ) );
  }
  else
  {
    // Cache does not contain the i-th image: load it from file and push_to front

    img = imread ( getCacheFilename ( i ), cv::IMREAD_UNCHANGED );

    images_cache_.push_front ( pair<int,Mat> ( i,img ) );
    // In case, remove from memory the oldest image
    if ( int ( images_cache_.size() ) > cache_max_size_ )
    {
      string cache_fn = getCacheFilename ( images_cache_.back().first );
      path cf_path ( cache_fn );

      if ( !exists ( cf_path ) )
        // Not yet cached into disk
      {
        imwrite ( cache_fn, images_cache_.back().second );
      }

      images_cache_.pop_back();
    }
  }
  
  return img;
}

string CameraCalibrationBase::getCacheFilename ( int i )
{
  stringstream sstr;
  sstr<<cache_folder_;
  sstr<<"/";
  sstr<<cache_file_basename;
  sstr<<setfill ( '0' ) <<setw ( 8 ) <<i;
  sstr<<".png";

  return sstr.str();
}

CameraCalibration::CameraCalibration ( int cache_max_size, std::string cache_folder ) :
  CameraCalibrationBase ( cache_max_size, cache_folder )
{}

bool CameraCalibration::addImage ( cv::Mat &img )
{
  // Check image size and type
  if ( num_images_ )
  {
    if ( img.size() != image_size_ || img.type() != image_type_  )
      return false;
  }
  else if ( image_size_ != cv::Size ( -1,-1 ) )
    if ( img.size() != image_size_ ) 
      return false;
    
  Mat img_gray;
  if ( img.channels() == 3 )
  {
    cvtColor ( img, img_gray, cv::COLOR_BGR2GRAY );
  }
  else if ( img.channels() == 1 )
  {
    img_gray = img;
  }
  else
    // Unknown format
  {
    return false;
  }

  // Extract corners, otherwise return false
  vector<Point2f> img_pts;
  if ( !findChessboardCornersPyr ( img_gray, board_size_, img_pts, pyr_num_levels_, pattern_has_white_circle_ ) )
  {
    return false;
  }

  // Store extracted corners
  images_points_.push_back ( img_pts );

  images_masks_.push_back ( true );
  per_view_errors_.push_back ( std::numeric_limits<double>::infinity() );

  // First image? Set image size and type
  if ( !num_images_ )
  {
    image_size_ = img.size();
    image_type_ = img.type();
  }

  // Add the image to cache
  cachePutImage ( num_images_, img );

  num_images_++;
  num_active_images_++;

  return true;
}

bool CameraCalibration::addImageFile ( string &filename )
{
  Mat img = imread ( filename, cv::IMREAD_UNCHANGED );
  if ( img.empty() )
  {
    return false;
  }

  return addImage ( img );
}

void CameraCalibration::setImageActive ( int i, bool active )
{
  if ( images_masks_[i] != active )
  {
    images_masks_[i] = active;
    num_active_images_ += active?1:-1;
  }
}

double CameraCalibration::calibrate()
{
  // Use only the selected images
  std::vector< std::vector<cv::Point2f> > calib_images_points;
  for ( int i = 0; i < num_images_; i++ )
  {
    calib_images_points.reserve(num_active_images_);
    if ( images_masks_[i] )
      calib_images_points.push_back ( images_points_[i] );
  }

  if ( !calib_images_points.size() )
  {
    return std::numeric_limits<double>::infinity();
  }

  vector<vector<Point3f> > object_points ( 1 );
  calcBoardCornerPositions ( object_points[0], board_size_, square_size_ );
  object_points.resize ( calib_images_points.size(), object_points[0] );

  int flags = 0;
  if ( fix_principal_point_ )
    flags |= cv::CALIB_FIX_PRINCIPAL_POINT;
  if ( zero_tan_dist_ )
    flags |= cv::CALIB_ZERO_TANGENT_DIST;
  if ( fix_aspect_ratio_ )
    flags |= cv::CALIB_FIX_ASPECT_RATIO;
  if ( use_intrinsic_guess_ && !camera_matrix_.empty() )
    flags |= cv::CALIB_USE_INTRINSIC_GUESS;

  vector<Mat> r_vecs, t_vecs;
  cv::calibrateCamera ( object_points, calib_images_points, image_size_, camera_matrix_,
                        dist_coeffs_, r_vecs, t_vecs,
                        flags | cv::CALIB_FIX_K4|cv::CALIB_FIX_K5|cv::CALIB_FIX_K6 );

  // Compute reprojection error
  vector<Point2f> rep_image_points;
  int num_points = 0, i_calib = 0;
  double total_err = 0, err;
  int obj_pts_size = ( int ) object_points[0].size();

  for ( int i = 0; i < num_images_; i++ )
  {
    if ( images_masks_[i] )
    {
      projectPoints ( Mat ( object_points[0] ), r_vecs[i_calib], t_vecs[i_calib], camera_matrix_,
                      dist_coeffs_, rep_image_points );
      err = norm ( Mat ( images_points_[i] ), Mat ( rep_image_points ), NORM_L2 );

      per_view_errors_[i] = std::sqrt ( err*err/obj_pts_size );

      total_err += err*err;
      num_points += obj_pts_size;
      i_calib++;
    }
    else
    {
      per_view_errors_[i] = std::numeric_limits<double>::infinity();
    }
  }

  return std::sqrt ( total_err/num_points );
}

void CameraCalibration::getCamModel( PinholeCameraModel &cam_model )
{
  if ( camera_matrix_.empty() )
    cam_model = PinholeCameraModel();
  else
    cam_model =  PinholeCameraModel ( camera_matrix_, image_size_.width,
                                      image_size_.height, dist_coeffs_ );
}

void CameraCalibration::setCamModel ( const cv_ext::PinholeCameraModel &model )
{
  camera_matrix_ = model.cameraMatrix();
  dist_coeffs_ = model.distorsionCoeff();
  image_size_ = model.imgSize();
}

void CameraCalibration::computeAverageExtrinsicParameters ( cv::Mat &r_vec, cv::Mat &t_vec )
{
  vector<Point3f> object_points, corner_pos;
  vector<Point2f> image_points;

  calcBoardCornerPositions ( corner_pos, board_size_, square_size_ );
  object_points.reserve ( corner_pos.size() *num_images_ );
  image_points.reserve ( corner_pos.size() *num_images_ );
  for ( int i = 0; i < num_images_; i++ )
  {
    if ( images_masks_[i] )
    {
      object_points.insert ( object_points.end(), corner_pos.begin(), corner_pos.end() );
      image_points.insert ( image_points.end(), images_points_[i].begin(), images_points_[i].end() );
    }
  }

  solvePnP ( object_points, image_points, camera_matrix_, dist_coeffs_, r_vec, t_vec );
}

void CameraCalibration::getCornersImage ( int i, cv::Mat &corners_img, float scale_factor )
{
  Mat img = cacheGetImage ( i );
  
  if ( scale_factor > 1 )
  {
    float scale = 1.0f/scale_factor;
    std::vector<cv::Point2f> img_pts;
    for ( auto &p : images_points_[i] )
      img_pts.push_back ( scale*p );
    
    if ( img.channels() < 3 )
      cvtColor ( img, img, COLOR_GRAY2BGR );
    
    cv::resize ( img, corners_img, cv::Size(), scale, scale );
    drawChessboardCorners ( corners_img, board_size_, Mat ( img_pts ), true );
  }
  else
  {
    if ( img.channels() < 3 )
      cvtColor ( img, corners_img, COLOR_GRAY2BGR );
    else
      img.copyTo(corners_img);

    drawChessboardCorners ( corners_img, board_size_, Mat ( images_points_[i] ), true );
  }
}

void CameraCalibration::getUndistortedImage ( int i, cv::Mat &und_img, float scale_factor )
{
  if ( camera_matrix_.empty() || dist_coeffs_.empty() )
    return;

  Mat img = cacheGetImage ( i );

  if ( scale_factor > 1 )
  {
    Mat tmp_und_img;
    cv::undistort ( img, tmp_und_img, camera_matrix_, dist_coeffs_ );
    float scale = 1.0f/scale_factor;
    cv::resize ( tmp_und_img, und_img, cv::Size(), scale, scale );
  }
  else
    cv::undistort ( img, und_img, camera_matrix_, dist_coeffs_ );
}

void CameraCalibration::getCornersDistribution ( float kernel_stdev, cv::Mat &corner_dist, float scale_factor )
{
  if ( scale_factor < 1 )
    scale_factor = 1.0f;

  float scale = 1.0f/scale_factor;
  Mat accumulator = Mat::zeros ( Size ( scale*image_size_.width, scale*image_size_.height ),
                                 cv::DataType<float>::type );

  if( scale_factor > 1 )
  {
    for ( int i = 0; i < int ( images_points_.size() ); i++ )
    {
      if ( images_masks_[i] )
      {
        for ( auto &p : images_points_[i] )
          accumulator.at<float> ( scale*p.y, scale*p.x ) += 1.0f;
      }
    }
  }
  else
  {
    for ( int i = 0; i < int ( images_points_.size() ); i++ )
    {
      if ( images_masks_[i] )
      {
        for ( auto &p : images_points_[i] )
          accumulator.at<float> ( p.y, p.x ) += 1.0f;
      }
    }    
  }

  kernel_stdev *= scale;
  int kernel_size = cvRound ( kernel_stdev*4*2 + 1 ) |1;
  Mat kernel1D = getGaussianKernel ( kernel_size, kernel_stdev, CV_32F );
  normalize ( kernel1D,kernel1D, 0.0, 1.0, cv::NORM_MINMAX, cv::DataType<float>::type );
  sepFilter2D ( accumulator, corner_dist, -1, kernel1D, kernel1D );
}

void CameraCalibration::clear() 
{
  clearDiskCache();

  num_images_ = num_active_images_ = 0;  
  
  camera_matrix_ = cv::Mat();
  dist_coeffs_ = cv::Mat();
  
  images_points_.clear();
  images_masks_.clear();
  per_view_errors_.clear();
}

StereoCameraCalibration::StereoCameraCalibration ( int cache_max_size, std::string cache_folder ) :
  CameraCalibrationBase ( cache_max_size, cache_folder )
{}

void StereoCameraCalibration::setCamModels ( const PinholeCameraModel cam_models[2] )
{
  if ( cam_models[0].imgSize() != cam_models[1].imgSize() )
    return;

  image_size_ = cam_models[0].imgSize();
  for ( int k = 0; k < 2; k++ )
  {
    camera_matrices_[k] = cam_models[k].cameraMatrix();
    dist_coeffs_[k] = cam_models[k].distorsionCoeff();
  }
}

void StereoCameraCalibration::getExtrinsicsParameters ( Mat& r_mat, Mat& t_vec )
{ 
  r_mat = r_mat_;
  t_vec = t_vec_;
}

void StereoCameraCalibration::getCamModels( PinholeCameraModel cam_models[2] )
{
  if ( !camera_matrices_[0].empty() && !camera_matrices_[1].empty() )
  {
    for ( int k = 0; k < 2; k++ )
      cam_models[k] = PinholeCameraModel ( camera_matrices_[k], image_size_.width,
                                           image_size_.height, dist_coeffs_[k] );
  }
  else
  {
    for ( int k = 0; k < 2; k++ )
      cam_models[k] = PinholeCameraModel();
  }
}

bool StereoCameraCalibration::addImagePair ( Mat imgs[2] )
{
  // Check image size and type
  if ( num_pairs_ )
  {
    if ( imgs[0].size() != image_size_ || imgs[1].size() != image_size_ || 
         imgs[0].type() != image_type_ || imgs[1].type() != image_type_ )
      return false;
  }
  else if ( image_size_ != cv::Size ( -1,-1 ) )
    if ( imgs[0].size() != image_size_ || imgs[1].size() != image_size_ )
      return false;
    
  vector<Point2f> img_pts[2];
  for ( int k = 0; k < 2; k++ )
  {
    Mat img_gray;
    if ( imgs[k].channels() == 3 )
    {
      cvtColor ( imgs[k], img_gray, cv::COLOR_BGR2GRAY );
    }
    else if ( imgs[k].channels() == 1 )
    {
      img_gray = imgs[k];
    }
    else
      // Unknown format
    {
      return false;
    }

    // Extract corners, otherwise return false
    if ( !findChessboardCornersPyr ( img_gray, board_size_, img_pts[k], pyr_num_levels_, pattern_has_white_circle_ ) )
    {
      return false;
    }
  }

  // Store extracted corners
  for ( int k = 0; k < 2; k++ )
  {
    images_points_[k].push_back ( img_pts[k] );
  }

  pairs_masks_.push_back ( true );
  per_view_errors_.push_back ( std::numeric_limits<double>::infinity() );

  // First images? Set image size and type
  if ( !num_pairs_ )
  {
    image_size_ = imgs[0].size();
    image_type_ = imgs[0].type();
  }
  
  // Add the images to cache
  for ( int k = 0; k < 2; k++ )
  {
    cachePutImage ( 2*num_pairs_ + k, imgs[k] );
  }

  num_pairs_++;
  num_active_pairs_++;

  return true;
}

bool StereoCameraCalibration::addImagePairFiles ( string filenames[2] )
{
  Mat imgs[2];
  for ( int k = 0; k < 2; k++ )
  {
    imgs[k] = imread ( filenames[k], cv::IMREAD_UNCHANGED );
    if ( imgs[k].empty() )
      return false;
  }

  return addImagePair ( imgs );
}

void StereoCameraCalibration::setImagePairActive ( int i, bool active )
{
  if ( pairs_masks_[i] != active )
  {
    pairs_masks_[i] = active;
    num_active_pairs_ += active?1:-1;
  }
}

double StereoCameraCalibration::calibrate()
{
  // Use only the selected pairs
  std::vector< std::vector<cv::Point2f> > calib_images_points[2];
  for ( int k = 0; k < 2; k++ )
    calib_images_points[k].reserve(num_active_pairs_);
  
  for ( int i = 0; i < num_pairs_; i++ )
  {
    if ( pairs_masks_[i] )
    {
      for ( int k = 0; k < 2; k++ )
        calib_images_points[k].push_back ( images_points_[k][i] );
    }
  }

  if ( !calib_images_points[0].size() )
    return std::numeric_limits<double>::infinity();

  vector<vector<Point3f> > object_points ( 1 );
  calcBoardCornerPositions ( object_points[0], board_size_, square_size_ );
  object_points.resize ( calib_images_points[0].size(), object_points[0] );

  int flags = 0;
  if ( camera_matrices_[0].empty() || camera_matrices_[1].empty() ||
       use_intrinsic_guess_ )
  {
    if ( fix_principal_point_ )
      flags |= cv::CALIB_FIX_PRINCIPAL_POINT;
    if ( zero_tan_dist_ )
      flags |= cv::CALIB_ZERO_TANGENT_DIST;
    if ( fix_aspect_ratio_ )
      flags |= cv::CALIB_FIX_ASPECT_RATIO;
    if ( force_same_focal_lenght_ )
      flags |= cv::CALIB_SAME_FOCAL_LENGTH;
    if ( use_intrinsic_guess_ && !camera_matrices_[0].empty() && !camera_matrices_[1].empty() )
      flags |= cv::CALIB_USE_INTRINSIC_GUESS;

    flags |= cv::CALIB_FIX_K4|cv::CALIB_FIX_K5|cv::CALIB_FIX_K6;
  }
  else
  {
    flags |= cv::CALIB_FIX_INTRINSIC;
  }

  Mat ess_mat, fund_mat;
  cv::stereoCalibrate ( object_points, calib_images_points[0], calib_images_points[1], camera_matrices_[0], dist_coeffs_[0],
                        camera_matrices_[1], dist_coeffs_[1], image_size_, r_mat_, t_vec_, ess_mat, fund_mat, flags );

  
  PinholeCameraModel cam_models[2];
  getCamModels( cam_models );
  stereo_rect_.setCameraParameters(cam_models, r_mat_, t_vec_ );
  stereo_rect_.update();

  /* Code taken from the OpenCV examples
   * Because the output fundamental matrix implicitly includes all the output information,
   * we can check the quality of calibration using the epipolar geometry constraint: m2^t*F*m1=0 */
  int num_points = 0;
  double err, total_err = 0;
  
  vector<Vec3f> lines[2];
  for ( int i = 0; i < num_pairs_; i++ )
  {
    if ( pairs_masks_[i] )
    {
      
      int npt = ( int ) images_points_[0][i].size();
      Mat img_pts[2];
      for ( int k = 0; k < 2; k++ )
      {
        cv::undistortPoints ( Mat ( images_points_[k][i] ), img_pts[k], camera_matrices_[k], dist_coeffs_[k], Mat(), camera_matrices_[k] );
        cv::computeCorrespondEpilines ( img_pts[k], k+1, fund_mat, lines[k] );
      }
      
      err = 0;
      for ( int j = 0; j < npt; j++ )
      {
        err += fabs ( images_points_[0][i][j].x*lines[1][j][0] + 
                      images_points_[0][i][j].y*lines[1][j][1] + lines[1][j][2] ) +
               fabs ( images_points_[1][i][j].x*lines[0][j][0] +
                      images_points_[1][i][j].y*lines[0][j][1] + lines[0][j][2] );
      }
      total_err += err;
      num_points += npt;
      per_view_errors_[i] = err/npt;
    }
    else
    {
      per_view_errors_[i] = std::numeric_limits<double>::infinity();
    }
  }
  
  return total_err/num_points;
}

void StereoCameraCalibration::getCornersImagePair ( int i, cv::Mat corners_imgs[2], float scale_factor )
{
  for( int k = 0; k < 2; k++ )
  {
    Mat img = cacheGetImage ( 2*i + k );
    
    if ( scale_factor > 1 )
    {
      float scale = 1.0f/scale_factor;
      std::vector<cv::Point2f> img_pts;
      for ( auto &p : images_points_[k][i] )
        img_pts.push_back ( scale*p );
      
      if ( img.channels() < 3 )
        cvtColor ( img, img, COLOR_GRAY2BGR );
      
      cv::resize ( img, corners_imgs[k], cv::Size(), scale, scale );
      drawChessboardCorners ( corners_imgs[k], board_size_, Mat ( img_pts ), true );
    }
    else
    {
      if ( img.channels() < 3 )
        cvtColor ( img, corners_imgs[k], COLOR_GRAY2BGR );
      else
        img.copyTo(corners_imgs[k]);

      drawChessboardCorners ( corners_imgs[k], board_size_, Mat ( images_points_[k][i] ), true );
    }
  }
}

void StereoCameraCalibration::getCornersDistribution ( float kernel_stdev, cv::Mat corner_dists[2], 
                                                       float scale_factor )
{
  if ( scale_factor < 1 )
    scale_factor = 1.0f;

  float scale = 1.0f/scale_factor;
  
  kernel_stdev *= scale;
  int kernel_size = cvRound ( kernel_stdev*4*2 + 1 ) |1;
  Mat kernel1D = getGaussianKernel ( kernel_size, kernel_stdev, CV_32F );
  normalize ( kernel1D,kernel1D, 0.0, 1.0, cv::NORM_MINMAX, cv::DataType<float>::type );
  
  
  for( int k = 0; k < 2; k++ )
  {
    Mat accumulator = Mat::zeros ( Size ( scale*image_size_.width, scale*image_size_.height ),
                                  cv::DataType<float>::type );

    if( scale_factor > 1 )
    {
      for ( int i = 0; i < int ( images_points_[k].size() ); i++ )
      {
        if ( pairs_masks_[i] )
        {
          for ( auto &p : images_points_[k][i] )
            accumulator.at<float> ( scale*p.y, scale*p.x ) += 1.0f;
        }
      }
    }
    else
    {
      for ( int i = 0; i < int ( images_points_[k].size() ); i++ )
      {
        if ( pairs_masks_[i] )
        {
          for ( auto &p : images_points_[k][i] )
            accumulator.at<float> ( p.y, p.x ) += 1.0f;
        }
      }
    }
    sepFilter2D ( accumulator, corner_dists[k], -1, kernel1D, kernel1D );
  }
}

void StereoCameraCalibration::rectifyImagePair ( int i, Mat rect_imgs[2], float scale_factor )
{
  Mat imgs[2];
  for( int k = 0; k < 2; k++ )
    imgs[k] = cacheGetImage ( 2*i + k );
  
  if ( scale_factor > 1 )
  {
    float scale = 1.0f/scale_factor;
    Mat tmp_rect_imgs[2];
    stereo_rect_.rectifyImagePair( imgs, tmp_rect_imgs );

    for( int k = 0; k < 2; k++ )
      cv::resize ( tmp_rect_imgs[k], rect_imgs[k], cv::Size(), scale, scale );
  }
  else
  {
    stereo_rect_.rectifyImagePair( imgs, rect_imgs );
  }
}

void StereoCameraCalibration::clear()
{
  clearDiskCache();

  num_pairs_ = num_active_pairs_ = 0;
  force_same_focal_lenght_ = false;
  
  for( int k = 0; k < 2; k++ )
  {
    images_points_[k].clear();
    camera_matrices_[k] = cv::Mat();
    dist_coeffs_[k] = cv::Mat();
  }
  
  r_mat_ = cv::Mat();
  t_vec_ = cv::Mat();

  pairs_masks_.clear();
  per_view_errors_.clear();

  stereo_rect_ = cv_ext::StereoRectification();
}

void cv_ext::readExtrinsicsFromFile ( std::string filename, cv::Mat &r_vec, cv::Mat &t_vec )
{
  cout<<"WARNING: readExtrinsicsFromFile () is a deprecated function! Replece it with cv_ext::read3DTransf()";
  int pos1 = filename.length() - 4, pos2 = filename.length() - 5;
  std::string ext_str1 = filename.substr ( ( pos1 > 0 ) ?pos1:0 );
  std::string ext_str2 = filename.substr ( ( pos2 > 0 ) ?pos2:0 );

  if ( ext_str1.compare ( ".yml" ) && ext_str2.compare ( ".yaml" ) )
  {
    filename += ".yml";
  }

  FileStorage fs ( filename, FileStorage::READ );

  fs["r_vec"] >> r_vec;
  fs["t_vec"] >> t_vec;

  fs.release();
}

void cv_ext::writeExtrinsicsToFile ( std::string filename, const cv::Mat &r_vec,
                                     const cv::Mat &t_vec )
{
  cout<<"WARNING: writeExtrinsicsToFile () is a deprecated function! Replece it with cv_ext::write3DTransf()";  
  int pos1 = filename.length() - 4, pos2 = filename.length() - 5;
  std::string ext_str1 = filename.substr ( ( pos1 > 0 ) ?pos1:0 );
  std::string ext_str2 = filename.substr ( ( pos2 > 0 ) ?pos2:0 );

  if ( ext_str1.compare ( ".yml" ) && ext_str2.compare ( ".yaml" ) )
  {
    filename += ".yml";
  }

  FileStorage fs ( filename, FileStorage::WRITE );

  fs << "r_vec" << r_vec;
  fs << "t_vec" << t_vec;

  fs.release();
}