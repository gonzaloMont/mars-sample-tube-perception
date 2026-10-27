#pragma once

#include <string>
#include <vector>
#include <utility>
#include <list>

#include "cv_ext/pinhole_camera_model.h"
#include "cv_ext/stereo_camera.h"

namespace cv_ext
{

/** @brief Abstract base class for calibration objects */       
class CameraCalibrationBase
{
public:

  /** @brief Object constructor
   * 
   * @param[in] cache_max_size Max number of calibration images kept in memory before start to cache them to disk
   * @param[in] cache_folder Path of the directory used to cache the calibration images
   * 
   * The constructor creates the directory cache_folder used to temporarily cache the calibration images. 
   * If this directory exits, the constructor deletes all the images previously saved
   */
  CameraCalibrationBase( int cache_max_size, std::string cache_folder );
  
  /** @brief Object destructor
   * 
   * The destructor deletes all the images temporarily cached (see the constructor)
   */
  virtual ~CameraCalibrationBase();
  
  /** @brief Set the size of the board, i.e. the number of internal corners by width and height */
  void setBoardSize( cv::Size s ){ board_size_ = s; };
  
  /** @brief Provide the size of the board, i.e. the number of internal corners by width and height */
  cv::Size boardSize(){ return board_size_; };
  
  /** @brief Set the size of a chesserboard square in the defined unit (points, millimeters, etc) */
  void setSquareSize( float s ){ square_size_ = s; };
  
  /** @brief Provide the size of a chesserboard square in the defined unit (points, millimeters, etc) */
  float squareSize(){ return square_size_; };
  
  /** @brief If the chesserboard has a white circle in the black square corresponding to the origin 
   *         of the reference frame, call this method with the enable parameter as true
   * 
   * Set to false by default
   * 
   * @note The use of a chesserboard with a white circle in the black square corresponding to the origin 
   *       of the reference frame is generally recommended
   */
  void setUseChessboardWhiteCircle( bool enable ){ pattern_has_white_circle_ = enable; };
  
  /** @brief Return true if the chesserboard has a white circle in the black square corresponding to the origin 
   *         of the reference frame
   */
  bool useChessboardWhiteCircle(){ return pattern_has_white_circle_; };
  
  /** @brief If enabled, the calibration uses a provided camera model as initial guess
   * 
   * Set to false by default
   */
  void setUseIntrinsicGuess( bool enable ){ use_intrinsic_guess_ = enable; };
  
  /** @brief Return true if the calibration uses a provided camera model as initial guess 
   */
  bool useIntrinsicGuess(){ return use_intrinsic_guess_; };
  
  /** @brief If enabled, consider in the calibration only fy as a free parameter, with fx/fy = 1
   * 
   * Set to false by default
   */
  void setFixAspectRatio( bool enable ){ fix_aspect_ratio_ = enable; };
  
  /** @brief Return true if the calibration considers only fy as a free parameter, with fx/fy = 1
   */
  bool fixAspectRatio(){ return fix_aspect_ratio_; };
  
  /** @brief If enabled, the principal point is not changed during the global optimization
   * 
   * Set to false by default
   */
  void setFixPrincipalPoint( bool enable ){ fix_principal_point_ = enable; };
  
  /** @brief Return true if the principal point is not changed during the global optimization
   */
  void fixPrincipalPoint( bool enable ){ fix_principal_point_ = enable; };
  
  /** @brief If enabled, the calibration assumes zero tangential distortion
   * 
   * Set to false by default
   */
  void setZeroTanDist( bool enable ){ zero_tan_dist_ = enable; };
    
  /** @brief Return true if the calibration assumes zero tangential distortion  */
  bool zeroTanDist(){ return zero_tan_dist_; };
  
  /** @brief Set the number of levels of the gaussian image pyramid used to extract corners
   * 
   * @param[in] pyr_num_levels Numbers of pyramid levels, it should be greater or equal than 1.
   * 
   * Build a gaussian pyramid of the image and start to extract the corners from the higher
   * level of a gaussian pyramid. For large images, it may be useful to use a 2 or 3 levels pyramid.
   * Set to 1 by default
   */
  void setPyramidNumLevels( int pyr_num_levels ) { pyr_num_levels_ = pyr_num_levels; };

  /** @brief Provide the number of levels of the gaussian image pyramid used to extract corners, 
   *         see setPyramidNumLevels() */
  int pyramidNumLevels() { return pyr_num_levels_; };

  /** @brief Provide the size of the images used for calibration
   * 
   * A (-1,-1) size is returnend if no images have been added
   */  
  cv::Size imagesSize(){ return image_size_; }
  
  /** @brief Provide the OpenCV image type of the images used for calibration
   * 
   * See cv::Mat::type()
   * A -1 type is returnend if no images have been added
   */  
  int imagesType(){ return image_type_; }

  /** @brief Pure virtual method that should be overridden in derived classes
   * 
   * It should run the calibration given the loadaed/selected images.
   * 
   * @return Some metric about the calibration
   */
  virtual double calibrate() = 0;
  
  /** @brief Pure virtual method that should be overridden in derived classes.
   *
   *  It should clear all the loaded images and the chache 
   **/
  virtual void clear() = 0;
  
protected:
  
  void clearDiskCache();
  void cachePutImage( int i, cv::Mat &img );
  cv::Mat cacheGetImage( int i );
  std::string getCacheFilename( int i );
  
  int cache_max_size_;
  std::string cache_folder_;
  cv::Size board_size_ = cv::Size(-1,-1);
  float square_size_ = 0.0f;
  bool pattern_has_white_circle_ = false,
       use_intrinsic_guess_ = false,
       fix_aspect_ratio_ = false, 
       zero_tan_dist_ = false,
       fix_principal_point_ = false;
  int pyr_num_levels_ = 1;
  
  cv::Size image_size_ = cv::Size(-1,-1);
  int image_type_;

private:
  /** List used to implement a very simple, linear-time access cache */
  std::list< std::pair<int, cv::Mat> > images_cache_;
};


/** @brief CameraCalibration is a class to be used to estimate the intrisic parameters of a single camera */
class CameraCalibration : public CameraCalibrationBase
{
public:

  /** @brief Object constructor
   * 
   * @param[in] cache_max_size Max number of calibration images kept in memory before start to cache them to disk
   * @param[in] cache_folder Path of the directory used to cache the calibration images
   * 
   * The constructor creates the directory cache_folder used to temporarily cache the calibration images. 
   * If this directory exits, the constructor deletes all the images previously saved
   */
  CameraCalibration( int cache_max_size = 100, std::string cache_folder = "/tmp" );
  
  /** @brief Object destructor
   * 
   * The destructor deletes all the images temporarily cached (see the constructor)
   */
  virtual ~CameraCalibration(){};
  
  /** @brief Set a previously computed camera model
   * 
   * @param[in] cam_model Input camera model
   * 
   * If setUseIntrinsicGuess() is set to true, this model will be used as an initial guess in the calibration
   */
  void setCamModel( const PinholeCameraModel &model );
  
  /** @brief Provide the resulting camera model 
   * 
   * @param[out] cam_model Output camera model
   * 
   * @note If no calibration has been performed, or no models have been set with setCamModel(),
   *       this method provides a default PinholeCameraModel object
   */
  void getCamModel( PinholeCameraModel &cam_model );
  
  /** Add an image to be used for calibration
   * 
   * @param[in] img A reference to a one or three channels image
   * 
   * @return True if the image is valid and the chessboard has been
   *         succesfully extracted, false otherwise
   */
  bool addImage( cv::Mat &img );

  /** @brief Load and add an image to be used for calibration
   * 
   * @param[in] filename Path of image file to be loaded.
   * 
   * @return True if the image is valid and the chessboard has been 
   *         succesfully extracted, false otherwise
   */  
  bool addImageFile ( std::string &filename );
  
  /** @brief Provide the number of images succesfully added with the addImage() or addImageFile()
   *         methods so far */
  int numImages(){ return num_images_; };

  /** @brief Provide the number of active images, i.e. the images actually used for 
   *         calibration (see setImageActive() and calibrate()) */
  int numActiveImages(){ return num_active_images_; };
  
  /** @brief Set whether an added image will be used in the calibration process
   * 
   * @param[in] i Index of the image, from 0 to numImages() - 1
   * @param[in] active If false, the image with index i will not be used for calibration
   * 
   * By default, all added images are used for calibration.
   */
  void setImageActive( int i, bool active );
  
  /** Return true if an added image will be used in the calibration process
   * 
   * @param[in] i Index of the image, from 0 to numImages() - 1
   */
  bool isImageActive( int i ){ return images_masks_[i]; };
  
  /** @brief Perform the calibration.
   * 
   * @return The root mean squared reprojection error, computed only on the images marked as active
   * 
   * The calibration is performed using only the images marked as active (see setImageActive())
   * If no images have been added (see addImage() and addImageFile()), or all images have been marked 
   * as not active, this method will return an infinite value
   */
  virtual double calibrate();

  /** @brief Provide a root mean squared reprojection error for the i-th image 
   * 
   * @param[in] i Index of the image, from 0 to numImages() - 1
   * 
   * If the last calibration has not been performed using the required image, 
   * this method will return an infinite value
   */
  double getReprojectionError( int i ){ return per_view_errors_[i]; };
  
  /** @brief Provide a possibly scaled image with drawn the detected chessboard corners
   * 
   * @param[in] i Index of the image, from 0 to numImages() - 1
   * @param[out] corners_img Output image with represented the extracted corners
   * @param[in] scale_factor The output image scale factor, it should be >= 1 
   */  
  void getCornersImage( int i, cv::Mat &corners_img, float scale_factor = 1.0f );

  /** @brief Provide a possibly scaled image that depicts a qualitative representation of the 
   *  non-normalized density of the chesserboard corners
   * 
   * @param[in] kernel_stdev Standard deviation of the Gaussian kernel used in the density 
   *                         estimation
   * @param[out] corner_dist Output corner distribution image  (one channel, depth CV_32F)
   * @param[in] scale_factor The output image scale factor, it should be >= 1
   * 
   * getCornersDistribution() considers the corners extracted from each added image (see addImage() and 
   * addImageFile()) marked as active (see setImageActive()).
   * The density is obtained by means of kernel density estimation, i.e. a simplified version 
   * of the Parzen–Rosenblatt window method, using a non-normalized Gaussian kernel (kernel(0,0) = 1) 
   * with standard deviation kernel_stdev.
   */  
  void getCornersDistribution( float kernel_stdev, cv::Mat &corner_dist, float scale_factor = 1.0f );
  
  /** @brief Provide a possibly scaled image, undistorted using the current calibration parameters
   * 
   * @param[in] i Index of the image, from 0 to numImages() - 1
   * @param[out] und_img Output undistorted image
   * @param[in] scale_factor The output image scale factor, it should be >= 1
   */
  void getUndistortedImage( int i, cv::Mat &und_img, float scale_factor = 1.0f );

  /** @brief Clear all the loaded images and the chache **/
  void clear();

  void computeAverageExtrinsicParameters( cv::Mat &r_vec, cv::Mat &t_vec );
  // TODO Move this method away??

private:
  
  int num_images_ = 0, num_active_images_ = 0;
  cv::Mat camera_matrix_, dist_coeffs_;
  
  std::vector< std::vector<cv::Point2f> > images_points_;
  std::vector<bool> images_masks_;
  std::vector<double> per_view_errors_;
};

class StereoCameraCalibration : public CameraCalibrationBase
{ 
public:
  /** @brief Object constructor
   * 
   * @param[in] cache_max_size Max number of calibration images kept in memory before start to cache them to disk
   * @param[in] cache_folder Path of the directory used to cache the calibration images
   * 
   * The constructor creates the directory cache_folder used to temporarily cache the calibration images. 
   * If this directory exits, the constructor deletes all the images previously saved
   */
  StereoCameraCalibration( int cache_max_size = 100, std::string cache_folder = "/tmp" );
  
  /** @brief Object destructor
   * 
   * The destructor deletes all the images temporarily cached (see the constructor)
   */
  virtual ~StereoCameraCalibration(){};
  
  /** @brief If enabled, enforce the focal lenghts to be the same for both camera
   *
   * This method has effect if no camera model has been provided in input (see setCamModels()) or
   * setUseIntrinsicGuess() is set to true.
   * Set to false by default
   */
  void setForceSameFocalLenght( bool enable ){ force_same_focal_lenght_ = enable; }; 

  /** @brief Return true if the calibration enforces the focal lenghts to be the same for both camera, 
   *         see setForceSameFocalLenght() */
  bool forceSameFocalLenght(){ return force_same_focal_lenght_; }; 
  
  /** @brief Set previously computed camera models
   * 
   * @param[in] cam_models Input camera models
   * 
   * Before calibrate a stereo camera, it is recommended to calibrate the cameras 
   * individually using CameraCalibration an to provide the resulting camera parameters to 
   * StereoCameraCalibration using setCamModels().
   * If setUseIntrinsicGuess() is set to true, this models will be used as an initial guess in 
   * the calibration, otherwise these parameters will be keep fixed and the calibration will 
   * estimate only the extrinsics parameters between the cameras.
   */  
  void setCamModels( const PinholeCameraModel cam_models[2] );   
 
  /** @brief Provide the resulting camera models
   * 
   * @param[out] cam_models Output camera models
   * 
   * @note If no calibration has been performed, or no models have been set with setCamModels(),
   *       this method provides two default PinholeCameraModel objects.
   */
  void getCamModels( PinholeCameraModel cam_models[2] );
  
  /** @brief Provide the resulting extrinsic parameters
   * 
   * @param[out] r_mat Rotation matrix between the first and the second camera
   * @param[out] t_vec Translation vector between the two cameras
   * 
   * @note If no calibration has been performed, this method provides two empty matrices
   */
  void getExtrinsicsParameters( cv::Mat &r_mat, cv::Mat &t_vec );
  
  /** Add a stereo pair of images to be used for calibration
   * 
   * @param[in] img An array of two one or three channels images
   * 
   * @return True if the images are valid and the chessboard has been 
   *         succesfully extracted in both images, false otherwise
   */  
  bool addImagePair( cv::Mat imgs[2] );
  
  /** @brief Load and add a stereo pair of images to be used for calibration
   * 
   * @param[in] filenames An array of two paths of image file to be loaded.
   * 
   * @return True if the images are valid and the chessboard has been 
   *         succesfully extracted in both images, false otherwise
   */   
  bool addImagePairFiles ( std::string filenames[2] );
  
  /** @brief Provide the number of image pairs succesfully added with the addImagePair() or
   *         addImagePairFiles() methods so far */  
  int numImagePairs(){ return num_pairs_; };

  /** @brief Provide the number of active image pairs, i.e. the pairs actually used for 
   *         calibration (see setImagePairActive() and calibrate()) */  
  int numActiveImagePairs(){ return num_active_pairs_; };
  
  /** @brief Set whether an added image pair will be used in the calibration process
   * 
   * @param[in] i Index of the pair, from 0 to numImagePairs() - 1
   * @param[in] active If false, the image pair with index i will not be used for calibration
   * 
   * By default, all added image pairs are used for calibration.
   */  
  void setImagePairActive( int i, bool active );
  
  /** Return true if an added image pair will be used in the calibration process
   * 
   * @param[in] i Index of the image, from 0 to numImagePairs() - 1
   */  
  bool isImagePairActive( int i ){ return pairs_masks_[i]; };
  
  /** @brief Perform the calibration.
   * 
   * @return The average epipolar error, computed only on the image pairs marked as active
   * 
   * The calibration is performed using only the image pairs marked as active (see setImagePairActive())
   * If no image pairs have been added (see addImagePair() and addImagePairFiles()), or all images have been marked 
   * as not active, this method will return an infinite value
   */  
  virtual double calibrate();

  /** @brief Provide the average epipolar error for the i-th image pair
   * 
   * @param[in] i Index of the image pair, from 0 to numImagePairs() - 1
   * 
   * If the last calibration has not been performed using the required image pair, 
   * this method will return an infinite value
   */
  double getEpipolarError( int i ){ return per_view_errors_[i]; };

  /** @brief Provide a pair possibly scaled images with drawn the detected chessboard corners
   * 
   * @param[in] i Index of the image pair, from 0 to numImagePairs() - 1
   * @param[out] corners_img Output image pair with represented the extracted corners
   * @param[in] scale_factor The output image scale factor, it should be >= 1 
   */  
  void getCornersImagePair( int i, cv::Mat corners_imgs[2], float scale_factor = 1.0f );
  
  /** @brief Provide a pair possibly scaled images that depicts a qualitative representation of the 
   *  non-normalized density of the chesserboard corners
   * 
   * @param[in] kernel_stdev Standard deviation of the Gaussian kernel used in the density 
   *                         estimation
   * @param[out] corner_dists Output corner distribution images (one channel, depth CV_32F)
   * @param[in] scale_factor The output image scale factor, it should be >= 1
   * 
   * getCornersDistribution() considers the corners extracted from each added image pair
   * (see addImagePair() and addImagePairFiles()) marked as active (see setImagePairActive()).
   * The density is obtained by means of kernel density estimation, i.e. a simplified version 
   * of the Parzen–Rosenblatt window method, using a non-normalized Gaussian kernel (kernel(0,0) = 1) 
   * with standard deviation kernel_stdev
   */  
  void getCornersDistribution( float kernel_stdev, cv::Mat corner_dists[2], float scale_factor = 1.0f );
  
  /** @brief Provide a pair possibly scaled rectified images
   * 
   * @param[in] i Index of the image pair to be rectified, from 0 to numImagePairs() - 1
   * @param[out] corners_img Output rectified image pair
   * @param[in] scale_factor The output image scale factor
   *
   * This method internally uses the StereoRectification() object, can can be called only after calibrate(),
   * otherwise two empty images will be returned.
   */  
  void rectifyImagePair( int i, cv::Mat rect_imgs[2], float scale_factor = 1.0f );

  /** @brief Clear all the loaded images and the chache **/
  void clear();
  
private:
  
  int num_pairs_ = 0, num_active_pairs_ = 0;
  bool force_same_focal_lenght_ = false;
  cv::Mat camera_matrices_[2], dist_coeffs_[2];
  cv::Mat r_mat_, t_vec_;
  
  std::vector< std::vector<cv::Point2f> > images_points_[2];
  std::vector<bool> pairs_masks_;
  std::vector<double> per_view_errors_;
  
  cv_ext::StereoRectification stereo_rect_;
}; 

void readExtrinsicsFromFile( std::string filename, cv::Mat &r_vec, cv::Mat &t_vec );
void writeExtrinsicsToFile( std::string filename, const cv::Mat &r_vec, const cv::Mat &t_vec );

}
