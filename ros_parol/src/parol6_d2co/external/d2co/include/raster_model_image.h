#pragma once    

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "cv_ext/cv_ext.h"


/** @brief Shared pointer typedef */
typedef std::shared_ptr< class RasterModelImage > RasterModelImagePtr;

class RasterModelImage
{
public:
  
  RasterModelImage(){};

  void storeModel(const Eigen::Quaterniond &r, const Eigen::Vector3d &t, std::vector<cv::Point2f> &pts, std::vector<float> &normals);
  void storeModelsFromOther(RasterModelImagePtr &other);
  void saveModels(std::string base_name, int step_points);
  void loadModels(std::string base_name);
  void saveModelsQuantizedNormals(std::string base_name, int step_points);
  void loadModelsQuantizedNormals(std::string base_name);
  void getModel(int idx, Eigen::Quaterniond &r, Eigen::Vector3d &t, std::vector<cv::Point2f> &pts, std::vector<float> &normals);
  void getModel(int idx, Eigen::Quaterniond &r, Eigen::Vector3d &t, std::vector<cv::Point2f> &pts, std::vector<unsigned char> &quantized_normals);
  int numPrecomputedModelsViews() const {return precomputed_rq_.size(); };
  void clearModel()
  {
    precomputed_pts_.clear();
    precomputed_normals_.clear();
    precomputed_quantized_normals_.clear();
    precomputed_rq_.clear();
    precomputed_t_.clear();
  };
  
private:

  void computeNormalsQuantization(const std::vector<float> &normals, std::vector<unsigned char> &q_normals);
  int getQuantizedOrientation(float dir, int n0);

  std::vector< std::vector<cv::Point2f> > precomputed_pts_;
  std::vector< std::vector<float> > precomputed_normals_;
  std::vector< std::vector<unsigned char> > precomputed_quantized_normals_;
  std::vector<Eigen::Quaternionf, Eigen::aligned_allocator<Eigen::Quaternionf> > precomputed_rq_;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > precomputed_t_;
    
};
