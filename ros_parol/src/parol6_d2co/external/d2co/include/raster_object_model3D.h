#pragma once

#include "raster_object_model.h"

/** @brief Shared pointer typedef */
typedef std::shared_ptr< class RasterObjectModel3D > RasterObjectModel3DPtr;

class RasterObjectModel3D : public RasterObjectModel
{
public:
  
  RasterObjectModel3D();
  ~RasterObjectModel3D();
    
  virtual bool setModelFile( const std::string& filename );
  virtual bool allVisiblePoints() const{ return false; };
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

  void getDepthBufferData(int idx,std::vector<float>& dept_buffer_data) const;
  void getDepthBufferData(int idx, int idx2, float& dept_buffer_data_value) const;
  
  void setRenderZNear( float dist ){ render_z_near_ = dist; }
  void setRenderZFar( float dist ){ render_z_far_ = dist; }
  void setNormalEpsilon( float epsilon ){ normal_epsilon_ = epsilon; };

  /**
   * @brief This method request to load and use the vertex color information (if available)
   * 
   * After loading the model, it is possible to check if the vertex color information has been loaded
   * by calling the vertexColorsEnabled() method
   * 
   * @note The methods requestVertexColors() and setVerticesColor() are mutually exclusive
   */
  void requestVertexColors ()
  {
    vertex_color_ = cv::Scalar(-1);
    has_color_ = true;
  };

  /**
   * @brief Set a uniform vertex color, to be used for all vertices
   *
   * @param[in] color An RGB color (each component in the range [0,255])
   * 
   * @note The methods requestVertexColors() and setVerticesColor() are mutually exclusive
   */
  void setVerticesColor ( cv::Scalar color )
  {
    vertex_color_ = color;
    has_color_ = true;
  };
  
  /**
   * @brief This method request to use lighting in model rendering
   *
   * @param[in] enable If true, add basig shading to the rendered object
   * 
   * @note Lighting in rendering is actually activeted only 
   *       if vertex color has been enabled (see enableVertexColors())
   */
  void requestRenderLighting ()
  {
    light_on_ = true;
  };
  
  /**
   * @brief Set the position of the spot light
   *
   * @param[in] pos 3D light position
   * 
   * The lighting has to be enabled, see requestRenderLighting()
   */  
  void setLightPos( cv::Point3f pos)
  {
    light_pos_ = pos;
  };
  
  float renderZNear() const { return render_z_near_; };
  float renderZFar() const { return render_z_far_; };
  float normalEpsilon() const { return normal_epsilon_; };
  
  /**
   * @brief Return true if vertex color information is used
   */
  bool vertexColorsEnabled() const { return has_color_; }
  
  /**
   * @brief Provide the uniform vertex color possibly set with setVerticesColor()
   * 
   * @return A RGB color (each component in the range [0,255]), or (-1,-1,-1) if no color has been
   *         set with setVerticesColor()
   */  
  cv::Scalar getVerticesColor()
  {
    return vertex_color_;
  }
  
  /**
   * @brief Return true if lighting in model rendering is enabled
   */
  bool renderLightingEnabled () { return light_on_; };
  
  /**
   * @brief Provide the current position of the spot light
   */  
  cv::Point3f getLightPos() { return light_pos_; };
  
  void clearModel()
  {
    precomputed_vis_pts_.clear();
    precomputed_vis_d_pts_.clear();
    precomputed_vis_segs_.clear();
    precomputed_vis_d_segs_.clear();
    precomputed_depth_buffer_data_.clear();
    precomputed_rq_view_.clear();
    precomputed_t_view_.clear();
  }
  
  void enableDepthBufferStoring(bool enable){ depth_buffer_storing_enabled_=enable; };
  
  // project 3D point into the current gl perspective (returns the current depth_buffer_data_ index)
  inline int projectPointToGLPersp(const cv::Point3f& p, cv::Point& proj_p, float& depth);

  /**
   * @brief Provide a depth map of the model
   */  
  cv::Mat getModelDepthMap();
  
  /**
   * @brief Provide an image with the colored render of the model
   * 
   * @param background_color The RGB background color (each component in the range [0,255]) 
   * 
   * @note This method will return an empty image if the vertex color is not enabled (see 
   *       requestVertexColors() or setVerticesColor())
   */
  cv::Mat getRenderedModel( cv::Scalar background_color = cv::Scalar(0) );
  
  virtual cv::Mat getMask();


protected:

  virtual void clearModel2()
  {
    precomputed_vis_pts_.clear();
    precomputed_vis_d_pts_.clear();
    precomputed_vis_segs_.clear();
    precomputed_vis_d_segs_.clear();
    precomputed_depth_buffer_data_.clear();
    precomputed_rq_view_.clear();
    precomputed_t_view_.clear();
  }
  virtual void storeModel();
  virtual void retreiveModel( int idx );
  virtual void savePrecomputedModels( cv::FileStorage &fs ) const;
  virtual void loadPrecomputedModels( cv::FileStorage &fs );

  
private:

  bool initOpenGL();
  void createShader();
  void createImg2GLBufferMap();
  void loadMesh();

  inline void addLine( cv::Point3f &p0, cv::Point3f &p1 );
  inline void addVisibleLine( cv::Point3f &p0, cv::Point3f &p1 );
  bool checkPointOcclusion( cv::Point3f& p );
  inline int denormalizePoint(const cv::Point2f& p, cv::Point& np );

  bool has_color_ = false, light_on_ = false;
  cv::Scalar vertex_color_ = cv::Scalar(-1);
  cv::Point3f light_pos_ = cv::Point3f(1,0,0);
  std::vector<cv::Point3f> pts_, vis_pts_, *vis_pts_p_;
  std::vector<cv::Point3f> d_pts_, vis_d_pts_, *vis_d_pts_p_;

  std::vector<cv::Vec6f> segs_, vis_segs_, *vis_segs_p_;
  std::vector<cv::Point3f> d_segs_, vis_d_segs_, *vis_d_segs_p_;

  std::vector< std::vector<cv::Point3f> > precomputed_vis_pts_;
  std::vector< std::vector<cv::Point3f> > precomputed_vis_d_pts_;
  
  std::vector< std::vector<cv::Vec6f> > precomputed_vis_segs_;
  std::vector< std::vector<cv::Point3f> > precomputed_vis_d_segs_;
  
  std::vector< std::vector<float> > precomputed_depth_buffer_data_;
  
  bool depth_buffer_storing_enabled_ = false;
    
  /* Pimpl idiom */
  class MeshModel; 
  std::shared_ptr< MeshModel > mesh_model_ptr_;
  
  cv::Size render_win_size_ = cv::Size ( 0, 0 );
  cv::Rect buffer_data_roi_ = cv::Rect ( 0, 0, 0, 0 );
  float render_z_near_ = 0.05f, 
        render_z_far_ = 5.0f,
        render_fovy_ = 0.0f;
  float depth_buffer_epsilon_ = 0.001f, 
        normal_epsilon_ = 0.4f;

  std::vector<float> depth_buffer_data_;
  std::vector<cv::Vec4b> color_buffer_data_;

  cv::Mat img2gl_map_;
  float depth_transf_a_, depth_transf_b_, depth_transf_c_;
  bool raster_updated_ = false, raster_initiliazed_ = false;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
