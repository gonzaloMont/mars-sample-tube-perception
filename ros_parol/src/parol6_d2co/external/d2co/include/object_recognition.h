#pragma once

#include <string>
#include <vector>
#include <map>
#include <fstream>

#include "cv_ext/cv_ext.h"
#include "raster_object_model3D.h"

#include "chamfer_matching.h"
#include "raster_model_image.h"
#include <Eigen/Dense>

/** TODO
 *
 * -Total refactory needed
 *  -- Remove hard-coded paths
 *  -- ...
 *
 */

class ObjectRecognition
{
public:
  ObjectRecognition();
  ~ObjectRecognition(){};
  
  void setCamModel( cv_ext::PinholeCameraModel &cam_model );
  void setModel( std::string model_name, bool precompute_views = false, 
                 double step_model = 0.001, double step_xy = 0.02 );
  void setSearchingPlane( Eigen::Vector3d plane_normal, double plane_dist );
  void clearModels();
  
  bool searchObject( const std::string &model_name, const cv::Mat &img, 
                     Eigen::Quaterniond &q_rot, Eigen::Vector3d &t );
                     
  bool detectObject ( const int N, RasterObjectModel3DPtr &obj_model_ptr, const cv::Mat& img, 
                                       const std::vector<Eigen::Quaterniond>& initial_q, const std::vector<Eigen::Vector3d>& initial_t,
                                       std::vector<std::pair<double,Eigen::Affine3d> >& model_views);
                     
  bool searchObjectWithInitialGuess( RasterObjectModel3DPtr &obj_model_ptr, RasterModelImagePtr &model_image_ptr, const cv::Mat& img, 
                                     std::vector<Eigen::Quaterniond>& initial_q, std::vector<Eigen::Vector3d>& initial_t,
                                     Eigen::Quaterniond& q_rot_out, Eigen::Vector3d& t_out, double &score, 
                                     bool save_results=false, std::string save_dir="" );
                                     
  void precomputeAndSaveModelViews( RasterObjectModel3DPtr &obj_model_ptr,
                                              const std::vector< Eigen::Quaterniond > &init_rotations, 
                                              const std::vector<Eigen::Vector3d> &init_translations);
                                              
  void precomputeAndSaveModelViews( RasterObjectModel3DPtr &obj_model_ptr,
                                    RasterModelImagePtr &model_image_ptr,
                                    const std::vector< Eigen::Quaterniond > &init_rotations, 
                                    const std::vector<Eigen::Vector3d> &init_translations);
                                    
  void precomputeModelViews( RasterObjectModel3DPtr &obj_model_ptr,
                                                    RasterModelImagePtr &model_image_ptr,
                                                    const std::vector< Eigen::Quaterniond > &init_rotations, 
                                                    const std::vector<Eigen::Vector3d> &init_translations);
                                    
  bool initialGuessMethods ( RasterObjectModel3DPtr &obj_model_ptr, RasterModelImagePtr &model_image_ptr, cv::Mat& img, 
                            Eigen::Quaterniond& r_gr, Eigen::Vector3d& t_gr,
                            bool algorithm, int &false_pos, int &time, 
                            int &time_with_images_computation,
                            bool save_results=false, std::string save_dir="" );
  
  
  /// used in perception_strategy ///
  void extractObjectCandidates( const int N, RasterObjectModel3DPtr &obj_model_ptr, const cv::Mat& img, 
                                std::vector<std::pair<double,Eigen::Affine3d> >& model_views );
  void extractMultiObjectCandidates( const int N, std::vector<RasterObjectModel3DPtr> &multi_obj_model_ptr, const cv::Mat& img, 
                                std::vector<std::pair<int, std::pair<double,Eigen::Affine3d> > >& model_views );
  void getCandidatesAvgDist( RasterObjectModel3DPtr &obj_model_ptr, const cv::Mat& img, 
                             std::vector<std::pair<double,Eigen::Affine3d> >& model_views);
  void getCandidatesScore(RasterObjectModel3DPtr& obj_model_ptr, const cv::Mat& img, 
                          std::vector< std::pair< double, Eigen::Affine3d > >& model_views);
  void getMultiCandidatesScore(std::vector<RasterObjectModel3DPtr> &multi_obj_model_ptr, const cv::Mat& img, 
                          std::vector<std::pair<int, std::pair<double,Eigen::Affine3d> > >& model_views);
private:
  
  struct Model
  {
    std::string model_name;
    RasterObjectModel3DPtr obj_model_ptr;
    double step;
  };
  
  struct PoseCandidate
  {
    double score;
    Eigen::Quaterniond r;
    Eigen::Vector3d t;
  };

  cv_ext::PinholeCameraModel cam_model_;
  std::map<std::string, Model> models_;
  std::string current_model_;
  Eigen::Vector3d plane_normal_;
  Eigen::Quaterniond plane_q_rot_;
  Eigen::Vector3d plane_center_;
  double plane_dist_;
  std::vector< Eigen::Quaterniond > quat_rotations_;
  
  void getObjectMask( const cv::Mat &src_img, cv::Mat &obj_mask );
  double evaluateScore ( cv_ext::ImageStatisticsPtr &img_stats_p,
                         std::vector<cv::Point2f> &raster_pts,
                         const std::vector<float> &normal_directions );
                         
  void extractBestModelViews( DirectionalChamferMatching dc_matching, RasterObjectModel3DPtr &obj_model_ptr,
                                              std::vector<int> &idxs );
                       
  void precomputeBestModelViews( DirectionalChamferMatching dc_matching, RasterObjectModel3DPtr &obj_model_ptr,
                                              const std::vector< Eigen::Quaterniond > &init_rotations, 
                                              const std::vector<Eigen::Vector3d> &init_translations,
                                              std::vector<int> &idxs );
  
  void precomputeModelViews( RasterObjectModel3DPtr &obj_model_ptr, 
                          const std::vector< Eigen::Quaterniond > &init_rotations, 
                          Eigen::Quaterniond &plane_q_rot, Eigen::Vector3d &plane_center,
                          double step_xy = 0.04, double area_max_side = 2.0);
                          
  void precomputeModelViews( RasterObjectModel3DPtr &obj_model_ptr,
                            const std::vector< Eigen::Quaterniond > &init_rotations, 
                            const std::vector<Eigen::Vector3d> &init_translations);
                            
  void extractBestModelViews( const int N, const cv::Mat& img, DirectionalChamferMatching dc_matching, RasterObjectModel3DPtr &obj_model_ptr,
                                              const std::vector< Eigen::Quaterniond > &init_rotations, 
                                              const std::vector<Eigen::Vector3d> &init_translations,
                                              std::vector<std::pair<double,Eigen::Affine3d> > &model_views );
                            
  void precomputeModelViews( RasterModelImagePtr &model_image_ptr, std::vector<int> &idxs,
                            RasterObjectModel3DPtr &obj_model_ptr);
                            
  void precomputeModelViews( RasterModelImagePtr &model_image_ptr, std::vector<int> &idxs, std::vector<int> &idxs_out,
                            RasterObjectModel3DPtr &obj_model_ptr);
                            
  void extractIndicesFromResponseMaps(RasterObjectModel3DPtr &obj_model_ptr,
                                     const std::vector<cv::Mat> &responseMaps,
                                     std::vector<int> &idxs);
                                     
  void extractIndicesFromResponseMaps(RasterModelImagePtr &model_image_ptr,
                                     const std::vector<cv::Mat> &responseMaps,
                                     std::vector<int> &idxs);
                                                       
  void extractIndicesFromLookupTables(RasterObjectModel3DPtr &obj_model_ptr,
                                     const cv::Mat &binary,
                                     const std::vector<float*> &lookup_tables,
                                     std::vector<int> &idxs);
  
  void extractIndicesFromLookupTables(RasterModelImagePtr &model_image_ptr,
                                     const cv::Mat &binary,
                                     const std::vector<float*> &lookup_tables,
                                     std::vector<int> &idxs);
                                      
  void extractIndicesFromTensor(RasterModelImagePtr &model_image_ptr,
                                                   cv_ext::ImageTensorPtr &dist_map_tensor_ptr,
                                                   std::vector<int> &idxs);
                                                   
  void extractIndicesFromTensor(RasterModelImagePtr &model_image_ptr,
                                       cv_ext::ImageTensorPtr &dist_map_tensor_ptr,
                                       std::vector<std::pair<float,int> > &idxs);
  
  void globalMatch(DirectionalChamferMatching dc_matching,
                  RasterObjectModel3DPtr &obj_model_ptr,
                  cv_ext::ImageStatisticsPtr &img_stats_p,
                  std::vector<PoseCandidate> &out_poses,
                  const cv::Mat &mask = cv::Mat() );
                  
                  
  void matchWithInitialGuess(DirectionalChamferMatching dc_matching,
                              RasterObjectModel3DPtr &obj_model_ptr,
                              cv_ext::ImageStatisticsPtr &img_stats_p,
                              std::vector<PoseCandidate> &out_poses,
                              std::vector<int>& view_idxs, 
                              bool save_results=false, std::string save_dir="");
                  
  void testPrecomputedViews( RasterObjectModel3DPtr &obj_model_ptr );
                  
  void computeBinaryImage(const cv::Mat gr_dir, const cv::Mat gr_mag, cv::Mat& binary);
  int getQuantizedOrientation(float dir, int n0);
  void checkInMask(cv::Mat &quant, int r, int c, int T, int q, cv::Mat &bin);
  void computeLookupTable(std::vector<float*> &lookup_tables);
  float quantumToAngle(int q, int n0);
  void computeResponseMaps(cv::Mat &binary, std::vector<float*> &lookup_tables, std::vector<cv::Mat> &responseMaps);
  
  
  void computeStarOfRotations(const Eigen::Matrix3d& initial_orientation, std::vector< Eigen::Quaterniond >& quat_rotations);
  void computeInitGuess(const Eigen::Vector3d& t, const Eigen::Matrix3d& R, std::vector<Eigen::Vector3d>& t_list);
  void project3dEigenPoints(const cv::Mat& K, const double scale, const std::vector<Eigen::Vector3d>& p_list, std::vector<cv::Point2f>& proj_pts);

  
};
