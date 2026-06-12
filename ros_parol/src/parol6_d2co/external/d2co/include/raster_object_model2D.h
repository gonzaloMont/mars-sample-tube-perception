#pragma once

#include "raster_object_model.h"

/** @brief Shared pointer typedef */
typedef std::shared_ptr< class RasterObjectModel2D > RasterObjectModel2DPtr;

class RasterObjectModel2D : public RasterObjectModel
{
public:
  
  RasterObjectModel2D();
  ~RasterObjectModel2D(){};
    
  virtual bool setModelFile( const std::string &filename );
  virtual bool allVisiblePoints() const{ return true; };
  virtual void computeRaster();
  virtual void updateRaster();

  virtual const std::vector<cv::Point3f> &getPoints( bool only_visible_points = true ) const;
  virtual const std::vector<cv::Point3f> &getDPoints( bool only_visible_points = true) const;
  virtual const std::vector<cv::Vec6f> &getSegments( bool only_visible_segments = true ) const;
  virtual const std::vector<cv::Point3f> &getDSegments( bool only_visible_segments = true ) const;

  virtual const std::vector<cv::Point3f> &getPrecomputedPoints( int idx ) const;
  virtual const std::vector<cv::Point3f> &getPrecomputedDPoints( int idx ) const;
  virtual const std::vector<cv::Vec6f> &getPrecomputedSegments( int idx ) const;
  virtual const std::vector<cv::Point3f> &getPrecomputedDSegments( int idx ) const;

protected:

  virtual void storeModel();
  virtual void retreiveModel( int idx );
  virtual void savePrecomputedModels( cv::FileStorage &fs ) const;
  virtual void loadPrecomputedModels( cv::FileStorage &fs );
  
private:
  
  inline void addLine( cv::Point3f &p0, cv::Point3f &p1 );
  inline void addCircleArc( cv::Point3f &center, float radius, float start_ang, float end_ang );
  inline void addEllipseArc( cv::Point3f &center, cv::Point3f &major_axis_ep, 
                                    float minor_major_ratio, float start_ang, float end_ang );

  std::vector<cv::Point3f> pts_;
  std::vector<cv::Point3f> d_pts_;

  std::vector<cv::Vec6f> segs_;
  std::vector<cv::Point3f> d_segs_;
  
  /* Pimpl idiom */
  class CadModel; 
  std::shared_ptr< CadModel > cad_ptr_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
