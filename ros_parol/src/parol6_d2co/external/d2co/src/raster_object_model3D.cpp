#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <boost/concept_check.hpp>
#include <boost/thread/locks.hpp>

#include <OpenMesh/Core/IO/MeshIO.hh>
#include "raster_object_model3D.h"

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include "opengl_shaders.h"

using namespace glm;
using namespace std;
using namespace cv;

// Useful for glm backward compatibility (older glm version use degrees in place of radians)
#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
typedef OpenMesh::TriMesh_ArrayKernelT<>  Mesh;

static Mat debug_img, color_debug_img;

void checkFramebufferStatus()
{
  GLenum status = glCheckFramebufferStatus ( GL_FRAMEBUFFER );
  switch ( status )
  {
  case GL_FRAMEBUFFER_COMPLETE:
    // cerr<<"RasterObjectModel3D::GL_FRAMEBUFFER_COMPLETE"<<endl;
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
    cerr<<"RasterObjectModel3D::GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER"<<endl;
    break;
  case GL_FRAMEBUFFER_UNSUPPORTED:
    cerr<<"RasterObjectModel3D::GL_FRAMEBUFFER_UNSUPPORTED"<<endl;
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
    cerr<<"RasterObjectModel3D::GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT "<<endl;
    break;
  default:
    break;
  }
}

class RasterObjectModel3D::MeshModel
{
public:
  Mesh mesh;
  GLFWwindow* window = 0;
  GLuint fbo = 0, depth_rbo = 0, color_rbo = 0;
  vector<GLfloat> vertex_buffer_data;
  vector<GLfloat> color_buffer_data;
  vector<GLfloat> normal_buffer_data;
  GLuint vertex_buffer = 0, color_buffer = 0, normal_buffer;
  GLuint shader_program_id = 0;
  glm::mat4 persp, rt_persp;
};


RasterObjectModel3D::RasterObjectModel3D()
{
  mesh_model_ptr_ = shared_ptr< MeshModel > ( new MeshModel () );
}

RasterObjectModel3D::~RasterObjectModel3D()
{
  if( raster_initiliazed_ )
  {
    glDeleteBuffers ( 1, & ( mesh_model_ptr_->vertex_buffer ) );
    glDeleteBuffers ( 1, & ( mesh_model_ptr_->color_buffer ) );
    glDeleteBuffers ( 1, & ( mesh_model_ptr_->normal_buffer ) );
    glDeleteProgram ( mesh_model_ptr_->shader_program_id );
    glDeleteFramebuffers ( 1, &mesh_model_ptr_->fbo );
    glDeleteRenderbuffers ( 1, &mesh_model_ptr_->depth_rbo );
    glDeleteRenderbuffers ( 1, &mesh_model_ptr_->color_rbo );
    glfwTerminate();
  }
}

bool RasterObjectModel3D::setModelFile ( const string& filename )
{
  mesh_model_ptr_->mesh.clear();

  OpenMesh::IO::Options opt;

  if( has_color_ )
  {
    if( vertex_color_ == cv::Scalar(-1) )
    {
      mesh_model_ptr_->mesh.request_vertex_colors();
      opt += OpenMesh::IO::Options::VertexColor;
    }
    
    if( light_on_ )
    {
      mesh_model_ptr_->mesh.request_face_normals();
      mesh_model_ptr_->mesh.request_vertex_normals();
      opt += OpenMesh::IO::Options::FaceNormal;
      opt += OpenMesh::IO::Options::VertexNormal;
    }
  }

  if ( !OpenMesh::IO::read_mesh ( mesh_model_ptr_->mesh, filename, opt ) )
  {
    cerr<<"RasterObjectModel3D::setModelFile() - Error opening file: "<<filename<<endl;
    return false;
  }

  if( !opt.vertex_has_color() && vertex_color_ == cv::Scalar(-1) )
  {
    has_color_ = light_on_ = false;
  }

  return true;
}

void RasterObjectModel3D::computeRaster()
{
  if(cam_model_.imgWidth() == 0 || cam_model_.imgHeight() == 0 )
  {
    cerr<<"RasterObjectModel3D::computeRaster() - invalid camera model (zero image width or image height)"<<endl;
    return;
  }

  if( initOpenGL() )
  {    
    createShader();
    createImg2GLBufferMap();
    loadMesh();
    updateRaster();
    raster_initiliazed_ = true;
  }
}

bool RasterObjectModel3D::initOpenGL()
{
  // Initialise GLFW
  if ( !glfwInit() )
  {
    cerr<<"RasterObjectModel3D::initOpenGL() - Failed to initialize GLFW"<<endl;
    return false;
  }

  // Check thsi number
  glfwWindowHint ( GLFW_SAMPLES, 4 );
  glfwWindowHint ( GLFW_CONTEXT_VERSION_MAJOR, 2 );
  glfwWindowHint ( GLFW_CONTEXT_VERSION_MINOR, 1 );
  glfwWindowHint ( GLFW_VISIBLE, 0 );
  
  bool has_roi = cam_model_.regionOfInterestEnabled();
  // In case, disable temporarily the RoI ...
  
  if( has_roi )
    cam_model_.enableRegionOfInterest(false);

  // Comute the form factor and the render window size in order to obtain a rendered object with size in
  // pixel close to the actual size when projected into the image plane (actually, not very nice method...)
  cv::Point2f p_tl(0,0), p_br(cam_model_.imgWidth(),cam_model_.imgHeight());
  cv::Point2f np_tl, np_br;
  
  cam_model_.normalize((float*)&p_tl, (float*)&np_tl);
  cam_model_.normalize((float*)&p_br, (float*)&np_br);
    
  float hx1 = fabs(np_tl.x), hx2 = fabs(np_br.x),
        hy1 = fabs(np_tl.y), hy2 = fabs(np_br.y);
  float hx = ( hx1>hx2 )? hx1 : hx2, hy = ( hy1>hy2 ) ? hy1 : hy2;
        
  float fovy = 2.0f*atan(hy);

  cam_model_.denormalizeWithoutDistortion((float*)&np_tl, (float*)&p_tl);
  cam_model_.denormalizeWithoutDistortion((float*)&np_br, (float*)&p_br);
  
  render_win_size_.width = cvCeil(2.0*hx*(p_br.x - p_tl.x)/(hx1+hx2));
  render_win_size_.height = cvCeil(2.0*hy*(p_br.y - p_tl.y)/(hy1+hy2));
  
  glm::mat4 view = glm::lookAt ( glm::vec3 ( 0, 0, 0 ),
                                 glm::vec3 ( 0, 0, 1 ),
                                 glm::vec3 ( 0, 1, 0 ) );
  
  mesh_model_ptr_->persp = glm::perspective ( fovy, float(render_win_size_.width) / float(render_win_size_.height), 
                                              render_z_near_, render_z_far_ ) * view;
  
  if( has_roi ) 
  {
    p_tl = cv::Point2f(cam_model_.regionOfInterest().x, cam_model_.regionOfInterest().y);
    p_br = cv::Point2f(cam_model_.regionOfInterest().x + cam_model_.regionOfInterest().width, 
                      cam_model_.regionOfInterest().y + cam_model_.regionOfInterest().height);
  }
  else
  {
    p_tl = cv::Point2f(0, 0);
    p_br = cv::Point2f(cam_model_.imgWidth(), cam_model_.imgHeight());    
  }
  
  cam_model_.normalize((float *)&p_tl,(float *)&np_tl);
  cam_model_.normalize((float *)&p_br,(float *)&np_br);
 
  glm::vec4 hp_tl ( np_tl.x, np_tl.y, 1.0, 1.0f ), hp_br ( np_br.x, np_br.y, 1.0, 1.0f );
  glm::vec4 proj_hp_tl = mesh_model_ptr_->persp*hp_tl, proj_hp_br = mesh_model_ptr_->persp*hp_br;
  np_tl = Point2f( proj_hp_tl[0], proj_hp_tl[1] );
  np_br = Point2f( proj_hp_br[0], proj_hp_br[1] );

  Point2f tmp_p_tl(np_tl.x*0.5f + 0.5f,np_tl.y*0.5f + 0.5f),
          tmp_p_br(np_br.x*0.5f + 0.5f,np_br.y*0.5f + 0.5f);

  Point roi_tr( cvRound ( tmp_p_tl.x * render_win_size_.width ), cvRound ( tmp_p_tl.y * render_win_size_.height ) );
  Point roi_bl( cvRound ( tmp_p_br.x * render_win_size_.width ), cvRound ( tmp_p_br.y * render_win_size_.height ) );
  
  // TODO Avoid to perform this check improving the code above 
  if( roi_tr.x >= render_win_size_.width ) roi_tr.x = render_win_size_.width - 1;
  if( roi_tr.y < 0 ) roi_tr.y = 0;
  if( roi_bl.x < 0 ) roi_bl.x = 0;
  if( roi_bl.y >= render_win_size_.height ) roi_bl.y = render_win_size_.height - 1;
  
  buffer_data_roi_ = Rect(roi_bl.x, roi_tr.y, roi_tr.x - roi_bl.x, roi_bl.y - roi_tr.y);
  
  depth_buffer_data_.resize ( buffer_data_roi_.width*buffer_data_roi_.height );
  color_buffer_data_.resize ( buffer_data_roi_.width*buffer_data_roi_.height );  
  
  depth_transf_a_ = 2.0f * render_z_near_*render_z_far_;
  depth_transf_b_ = render_z_far_ + render_z_near_;
  depth_transf_c_ = render_z_far_ - render_z_near_;      
  

//   cout<<buffer_data_roi_<<endl;
  // In case, enable the RoI again...
  if( has_roi )
    cam_model_.enableRegionOfInterest(true);
  
  // Create an hidden window
  mesh_model_ptr_->window = glfwCreateWindow ( render_win_size_.width, render_win_size_.height, "Render", NULL, NULL );
  
  if ( mesh_model_ptr_->window == NULL )
  {
    cerr<<"RasterObjectModel3D::initOpenGL() - Failed to open GLFW window"<<endl;
    glfwTerminate();
    return false;
  }

  glfwMakeContextCurrent(mesh_model_ptr_->window);

  // Initialize GLEW
  if ( glewInit() != GLEW_OK )
  {
    cerr<<"RasterObjectModel3D::initOpenGL() - Failed to initialize GLEW"<<endl;
    glfwTerminate();
    return false;
  }
  
  return true;
}

void RasterObjectModel3D::updateRaster()
{
  glfwMakeContextCurrent(mesh_model_ptr_->window);

//   static uint64_t c_timer = 0, n_sample = 1;
//   cv_ext::BasicTimer timer;
  glBindFramebuffer(GL_FRAMEBUFFER, mesh_model_ptr_->fbo );
  glViewport(0,0,render_win_size_.width,render_win_size_.height);

  if( has_color_ )
  {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if( light_on_ )
      glEnable(GL_CULL_FACE);
  }
  else
    glClear ( GL_DEPTH_BUFFER_BIT );

  glUseProgram ( mesh_model_ptr_->shader_program_id );
  
  glm::mat4 model_view=glm::translate ( glm::mat4 ( 1.0f ), glm::vec3 ( ( float ) t_view_ ( 0 ), ( float ) t_view_ ( 1 ), ( float ) t_view_ ( 2 ) ) );
  glm::quat q_view( ( float ) rq_view_.w(), ( float ) rq_view_.x(), ( float ) rq_view_.y(), ( float ) rq_view_.z() );
  model_view=model_view*glm::mat4_cast ( q_view );
  glm::vec3 light_pos = glm::vec3(light_pos_.x,light_pos_.y,light_pos_.z);
                
  mesh_model_ptr_->rt_persp = mesh_model_ptr_->persp * model_view;
  
  GLuint mat_id = glGetUniformLocation ( mesh_model_ptr_->shader_program_id, "rt_persp" );
  glUniformMatrix4fv ( mat_id, 1, GL_FALSE, & ( mesh_model_ptr_->rt_persp[0][0] ) );

  GLuint vertex_pos_id = 0, vertex_color_id = 0, vertex_normal_id = 0; 
  
  // 1rst attribute buffer : vertices
  vertex_pos_id = glGetAttribLocation ( mesh_model_ptr_->shader_program_id, "vertex_pos" ),
  glEnableVertexAttribArray ( vertex_pos_id );
  glBindBuffer ( GL_ARRAY_BUFFER, mesh_model_ptr_->vertex_buffer );
  glVertexAttribPointer ( vertex_pos_id, 3, GL_FLOAT, GL_FALSE, 0, ( void* ) 0 );

  if( has_color_ )
  {
    // 2nd attribute buffer : vertices colors
    vertex_color_id = glGetAttribLocation ( mesh_model_ptr_->shader_program_id, "vertex_color" );
    glEnableVertexAttribArray(vertex_color_id);
    glBindBuffer(GL_ARRAY_BUFFER, mesh_model_ptr_->color_buffer );
    glVertexAttribPointer( vertex_color_id, 3, GL_FLOAT, GL_FALSE, 0, (void*)0 );
    
    if( light_on_ )
    {
      GLuint model_view_id = glGetUniformLocation( mesh_model_ptr_->shader_program_id, "model_view");
      glUniformMatrix4fv(model_view_id, 1, GL_FALSE, &model_view[0][0]);
      
      GLuint light_pos_world_id = glGetUniformLocation( mesh_model_ptr_->shader_program_id, "light_pos_world");
      glUniform3f(light_pos_world_id, light_pos.x, light_pos.y, light_pos.z);
      
      // 3rd attribute buffer : vertices normals
      vertex_normal_id = glGetAttribLocation ( mesh_model_ptr_->shader_program_id, "vertex_normal" );
      glEnableVertexAttribArray(vertex_normal_id);
      glBindBuffer(GL_ARRAY_BUFFER, mesh_model_ptr_->normal_buffer);
      glVertexAttribPointer( vertex_normal_id, 3, GL_FLOAT, GL_FALSE, 0, (void*)0 );
    }
  }
  // Draw the triangles
  glDrawArrays ( GL_TRIANGLES, 0, mesh_model_ptr_->vertex_buffer_data.size() );

  glDisableVertexAttribArray ( vertex_pos_id );
  glDisableVertexAttribArray ( vertex_color_id );
  glDisableVertexAttribArray ( vertex_normal_id );
  
  glReadPixels ( buffer_data_roi_.x, buffer_data_roi_.y, buffer_data_roi_.width, buffer_data_roi_.height, 
                 GL_DEPTH_COMPONENT, GL_FLOAT, ( GLvoid * ) depth_buffer_data_.data() );
  
//   Mat dbg_depth_buf(Size(buffer_data_roi_.width, buffer_data_roi_.height), cv::DataType< float >::type, 
//                     depth_buffer_data_.data(), Mat::AUTO_STEP);
//   cv_ext::showImage(dbg_depth_buf, "depth_buffer", true, 1);

  if( has_color_ )
  {
    glReadPixels ( buffer_data_roi_.x, buffer_data_roi_.y, buffer_data_roi_.width, buffer_data_roi_.height, 
                   GL_BGRA, GL_UNSIGNED_BYTE, ( GLubyte * ) color_buffer_data_.data() );
    
//     Mat dbg_color_buf(Size(buffer_data_roi_.width, buffer_data_roi_.height),
//                       cv::DataType< Vec4b >::type, color_buffer_data_.data(), Mat::AUTO_STEP );
//     cv_ext::showImage(dbg_color_buf, "color_buffer", true, 1);
  }
  
  glBindFramebuffer(GL_FRAMEBUFFER, 0  );

  Mesh &mesh = mesh_model_ptr_->mesh;

  vis_pts_.clear();
  vis_d_pts_.clear();
  vis_segs_.clear();
  vis_d_segs_.clear();

  vis_pts_p_ = &vis_pts_;
  vis_d_pts_p_ = &vis_d_pts_;
  vis_segs_p_ = &vis_segs_;
  vis_d_segs_p_ = &vis_d_segs_;

  // Iterate over all faces

  Eigen::Matrix4d RT ( Eigen::Matrix4d::Identity() );
  RT.block<3,3> ( 0,0 ) =rq_view_.toRotationMatrix();
  RT.col ( 3 ).head ( 3 ) =t_view_;

//   cv_ext::BasicTimer t;
  for ( Mesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it )
  {
    Mesh::FaceHalfedgeIter fh_it = mesh.fh_iter ( *f_it );
    Mesh::Normal face_normal = mesh.calc_face_normal ( *f_it );

    Eigen::Vector3d rotated_face_normal_eigen ( ( double ) face_normal[0], ( double ) face_normal[1], ( double ) face_normal[2] );
    rotated_face_normal_eigen=RT.block<3,3> ( 0,0 ) *rotated_face_normal_eigen;
    //Vec3f rotated_face_normal((float)rotated_face_normal_eigen(0),(float)rotated_face_normal_eigen(1),(float)rotated_face_normal_eigen(2));

    for ( ; fh_it.is_valid(); ++fh_it )
    {
      const OpenMesh::VertexHandle from_ph = mesh.from_vertex_handle ( *fh_it ),
                                   to_ph = mesh.to_vertex_handle ( *fh_it );

      const OpenMesh::DefaultTraits::Point &from_p = mesh.point ( from_ph ),
                                            &to_p = mesh.point ( to_ph );

      Point3f p0 ( from_p[0], from_p[1], from_p[2] ),
              p1 ( to_p[0], to_p[1], to_p[2] );
              
      Mesh::HalfedgeHandle opposite_heh = mesh.opposite_halfedge_handle ( *fh_it );
      Mesh::FaceHandle opposite_fh = mesh.face_handle ( opposite_heh );
      
      if( !opposite_fh.is_valid() )
      {
        addVisibleLine ( p0, p1 );
      }
      else
      {

        Mesh::Normal near_face_normal = mesh.calc_face_normal ( opposite_fh );
        OpenMesh::Vec3f normals_diff = near_face_normal - face_normal;
        float norm_dist = float ( std::sqrt ( normals_diff[0]*normals_diff[0] + 
                                              normals_diff[1]*normals_diff[1] + 
                                              normals_diff[2]*normals_diff[2] ) );
        
        if ( norm_dist > normal_epsilon_ )
        {
          addVisibleLine ( p0, p1 );
        }
        else
        {
          Eigen::Vector3f p0_e ( from_p[0], from_p[1], from_p[2] ),
                          p1_e ( to_p[0], to_p[1], to_p[2] );
          Eigen::Vector4d pfrom ( ( double ) p0_e ( 0 ), ( double ) p0_e ( 1 ), ( double ) p0_e ( 2 ),1 );
          Eigen::Vector4d pto ( ( double ) p1_e ( 0 ), ( double ) p1_e ( 1 ), ( double ) p1_e ( 2 ),1 );
          pfrom=RT*pfrom;
          pto=RT*pto;
          pfrom=pfrom/pfrom ( 3 );
          pto=pto/pto ( 3 );
          Eigen::Vector3d centr ( ( pfrom+pto ).head ( 3 ) /2 );
          double dot=centr.dot ( rotated_face_normal_eigen );

          Eigen::Vector3d rotated_near_face_normal_eigen ( ( double ) near_face_normal[0], ( double ) near_face_normal[1], ( double ) near_face_normal[2] );
          rotated_near_face_normal_eigen=RT.block<3,3> ( 0,0 ) *rotated_near_face_normal_eigen;
          //Vec3f rotated_near_face_normal((float)rotated_near_face_normal_eigen(0),(float)rotated_near_face_normal_eigen(1),(float)rotated_near_face_normal_eigen(2));
          double dot_near=centr.dot ( rotated_near_face_normal_eigen );
          if ( ( dot*dot_near ) < 0 )
          {
            addVisibleLine ( p0, p1 );
          }
        }
      }
    }
  }
  raster_updated_ = true;
//   cout<<"Projection time: "<<t.elapsedTimeUs()<<endl;
//   cout<<"updateRaster() time: "<<(c_timer += timer.elapsedTimeUs())/n_sample++<<endl;
}

const vector< cv::Point3f >& RasterObjectModel3D::getPoints(bool only_visible_points) const
{
  if(only_visible_points)
    return *vis_pts_p_;
  else
    return pts_;
}

const vector< cv::Point3f >& RasterObjectModel3D::getDPoints(bool only_visible_points) const
{
  if(only_visible_points)
    return *vis_d_pts_p_;
  else
    return d_pts_;
}

const vector< cv::Vec6f >& RasterObjectModel3D::getSegments(bool only_visible_segments) const
{
  if(only_visible_segments)
    return *vis_segs_p_;
  else
    return segs_;
}

const vector< cv::Point3f >& RasterObjectModel3D::getDSegments(bool only_visible_segments) const
{
  if(only_visible_segments)
    return *vis_d_segs_p_;
  else
    return d_segs_;
}

const vector< cv::Point3f >& RasterObjectModel3D::getPrecomputedPoints(int idx) const
{
  return precomputed_vis_pts_[idx];
}

const vector< cv::Point3f >& RasterObjectModel3D::getPrecomputedDPoints(int idx) const
{
  return precomputed_vis_d_pts_[idx];
}

const vector< cv::Vec6f >& RasterObjectModel3D::getPrecomputedSegments(int idx) const
{
  return precomputed_vis_segs_[idx];
}

const vector< cv::Point3f >& RasterObjectModel3D::getPrecomputedDSegments(int idx) const
{
  return precomputed_vis_d_segs_[idx];
}

void RasterObjectModel3D::getDepthBufferData ( int idx, vector<float>& dept_buffer_data ) const
{
  dept_buffer_data = precomputed_depth_buffer_data_[idx];
}

void RasterObjectModel3D::getDepthBufferData ( int idx, int idx2, float& dept_buffer_data_value ) const
{
  dept_buffer_data_value = precomputed_depth_buffer_data_[idx][idx2];
}

void RasterObjectModel3D::createShader()
{
  glGenFramebuffers ( 1, &mesh_model_ptr_->fbo );
  glBindFramebuffer ( GL_FRAMEBUFFER, mesh_model_ptr_->fbo );

  // Render buffer as depth buffer
  glGenRenderbuffers ( 1, &mesh_model_ptr_->depth_rbo );
  glBindRenderbuffer ( GL_RENDERBUFFER, mesh_model_ptr_->depth_rbo );
  glRenderbufferStorage ( GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, render_win_size_.width, render_win_size_.height );
  glBindRenderbuffer(GL_RENDERBUFFER, 0);
  // Attach render buffer to the fbo as depth buffer
  glFramebufferRenderbuffer ( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER,  mesh_model_ptr_->depth_rbo );

  if( has_color_ )
  {
    // Render buffer as color buffer
    glGenRenderbuffers(1, &mesh_model_ptr_->color_rbo );
    glBindRenderbuffer(GL_RENDERBUFFER, mesh_model_ptr_->color_rbo );
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, render_win_size_.width, render_win_size_.height );
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    // Attach render buffer to the fbo as color buffer
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, mesh_model_ptr_->color_rbo );
  }

  checkFramebufferStatus();

  // Enable depth test
  glEnable ( GL_DEPTH_TEST );
  // Accept fragment if it closer to the camera than the former one
  glDepthFunc ( GL_LESS );

  if( has_color_ && light_on_ )
  {
    // Cull triangles which normal is not towards the camera
    glEnable(GL_CULL_FACE);    
  }
  

  GLuint vertex_shader_id = glCreateShader ( GL_VERTEX_SHADER ), fragment_shader_id = 0;
  string shading_version ( ( char* ) glGetString ( GL_SHADING_LANGUAGE_VERSION ) );
  cout<<"RasterObjectModel3D::createShader() - Shader_version: "<<shading_version<<endl;

  string sh_ver="";
  for ( int i=0; i < int(shading_version.size()); i++ )
    if ( isdigit ( shading_version[i] ) ) sh_ver.push_back ( shading_version[i] );

  string vertex_shader_code;

  if( has_color_ )
  {
    if( light_on_ )
      vertex_shader_code = string( SHADED_OBJECT_VERTEX_SHADER_CODE ( sh_ver ) );  
    else
      vertex_shader_code = string( COLORED_OBJECT_VERTEX_SHADER_CODE ( sh_ver ) );    
  }  
  else
    vertex_shader_code = string( OBJECT_VERTEX_SHADER_CODE ( sh_ver ) );

  GLint result = GL_FALSE;

  // Compile Vertex Shader
  char const * vertex_source_ptr = vertex_shader_code.c_str();
  glShaderSource ( vertex_shader_id, 1, &vertex_source_ptr , NULL );
  glCompileShader ( vertex_shader_id );

  // Check Vertex Shader
  int info_log_length;
  glGetShaderiv ( vertex_shader_id, GL_COMPILE_STATUS, &result );
  glGetShaderiv ( vertex_shader_id, GL_INFO_LOG_LENGTH, &info_log_length );
  if (info_log_length > 1 )
  {
    vector<char> vertex_shader_error_message ( info_log_length + 1 );
    glGetShaderInfoLog ( vertex_shader_id, info_log_length, NULL, &vertex_shader_error_message[0] );
    cout<<"RasterObjectModel3D::createShader() - ShaderInfoLog Vertex Shader: "<<&vertex_shader_error_message[0]<<endl;
  }

  // Link the program
  mesh_model_ptr_->shader_program_id = glCreateProgram();

  glAttachShader ( mesh_model_ptr_->shader_program_id, vertex_shader_id );

  if( has_color_ )
  {
    fragment_shader_id = glCreateShader ( GL_FRAGMENT_SHADER );
    string fragment_shader_code;
    if( light_on_ )
      fragment_shader_code = string( SHADED_OBJECT_FRAGMENT_SHADER_CODE( sh_ver ) );
    else
      fragment_shader_code = string( COLORED_OBJECT_FRAGMENT_SHADER_CODE( sh_ver ) );

    // Compile Fragment Shader
    char const * fragment_source_ptr = fragment_shader_code.c_str();
    glShaderSource ( fragment_shader_id, 1, &fragment_source_ptr , NULL );
    glCompileShader ( fragment_shader_id );

    // Check Fragment Shader
    glGetShaderiv(fragment_shader_id, GL_COMPILE_STATUS, &result);
    glGetShaderiv(fragment_shader_id, GL_INFO_LOG_LENGTH, &info_log_length);
    if ( info_log_length > 1 )
    {
      vector<char> fragment_shader_error_message(info_log_length + 1);
      glGetShaderInfoLog(fragment_shader_id, info_log_length, NULL, &fragment_shader_error_message[0]);
      cout<<"RasterObjectModel3D::createShader() - ShaderInfoLog Fragment Shader: "<<&fragment_shader_error_message[0]<<endl;
    }
    glAttachShader(mesh_model_ptr_->shader_program_id, fragment_shader_id);
//     glBindFragDataLocation(mesh_model_ptr_->shader_program_id, 0, "fragment_out");
  }

  glLinkProgram ( mesh_model_ptr_->shader_program_id );

  // Check the program
  glGetProgramiv(mesh_model_ptr_->shader_program_id, GL_LINK_STATUS, &result);
  glGetProgramiv(mesh_model_ptr_->shader_program_id, GL_INFO_LOG_LENGTH, &info_log_length);
  if ( info_log_length > 1 ){
    vector<char> program_error_message(info_log_length + 1);
    glGetProgramInfoLog(mesh_model_ptr_->shader_program_id, info_log_length, NULL, &program_error_message[0]);
    cout<<"RasterObjectModel3D::createShader() - ShaderInfoLog Program: "<<&program_error_message[0]<<endl;
  }

  glDeleteShader ( vertex_shader_id );
  glDeleteShader ( fragment_shader_id );

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void RasterObjectModel3D::createImg2GLBufferMap()
{
  int rows = cam_model_.imgHeight(), cols = cam_model_.imgWidth();
  img2gl_map_ = Mat( rows, cols , DataType<int>::type );
  for( int r = 0; r < rows; r++ )
  {
    int *map_p = img2gl_map_.ptr<int>(r);

    Point2f p, np;
    Point gl_p;
    for( int c = 0; c < cols; c++, map_p++ )
    {
      p.x = float(c); p.y = float(r);
      cam_model_.normalize((float *)&p,(float *)&np);

      glm::vec4 hp ( np.x, np.y, 1.0, 1.0f );
      glm::vec4 proj_hp = mesh_model_ptr_->persp*hp;
      np = Point2f( proj_hp[0], proj_hp[1] );

      *map_p = denormalizePoint(np, gl_p);
    }
  }
}

void RasterObjectModel3D::loadMesh()
{
  mesh_model_ptr_->vertex_buffer_data.clear();
  mesh_model_ptr_->color_buffer_data.clear();
  mesh_model_ptr_->normal_buffer_data.clear();

  Mesh &mesh = mesh_model_ptr_->mesh;

  pts_.clear();
  d_pts_.clear();
  segs_.clear();
  d_segs_.clear();

  float multiplier = 1.0f/unit_meas_;

  // (linearly) iterate over all vertices
  for ( Mesh::VertexIter v_it = mesh.vertices_sbegin(); v_it!=mesh.vertices_end(); ++v_it )
  {
    OpenMesh::DefaultTraits::Point &p = mesh.point ( *v_it );
    p *= multiplier;
  }

  if ( centroid_orig_offset_ == CENTROID_ORIG_OFFSET )
  {
    OpenMesh::DefaultTraits::Point mean(0,0,0);
    int n_vtx = 0;
    for ( Mesh::VertexIter v_it = mesh.vertices_sbegin(); v_it!=mesh.vertices_end(); ++v_it )
    {
      OpenMesh::DefaultTraits::Point &p = mesh.point ( *v_it );
      mean += p;
      n_vtx++;
    }
    mean[0] /= float ( n_vtx );
    mean[1] /= float ( n_vtx );
    mean[2] /= float ( n_vtx );

    orig_offset_ = -Point3f ( mean[0], mean[1], mean[2] );
  }
  else if ( centroid_orig_offset_ == BOUNDING_BOX_CENTER_ORIG_OFFSET )
  {
    OpenMesh::DefaultTraits::Point min(numeric_limits<float>::max(),
                                       numeric_limits<float>::max(),
                                       numeric_limits<float>::max()),
                                   max(-numeric_limits<float>::max(),
                                       -numeric_limits<float>::max(),
                                       -numeric_limits<float>::max());

    for ( Mesh::VertexIter v_it = mesh.vertices_sbegin(); v_it!=mesh.vertices_end(); ++v_it )
    {
      OpenMesh::DefaultTraits::Point &p = mesh.point ( *v_it );
      if ( p[0] > max[0] )  max[0] = p[0];
      else if ( p[0] < min[0] ) min[0] = p[0];
      if ( p[1] > max[1] )  max[1] = p[1];
      else if ( p[1] < min[1] ) min[1] = p[1];
      if ( p[2] > max[2] )  max[2] = p[2];
      else if ( p[2] < min[2] ) min[2] = p[2];
    }

    orig_offset_ = -Point3f ( min[0] + ( max[0] - min[0] ) /2.0f,
                                  min[1] + ( max[1] - min[1] ) /2.0f,
                                  min[2] + ( max[2] - min[2] ) /2.0f );
  }

  OpenMesh::DefaultTraits::Point orig_offset ( orig_offset_.x, orig_offset_.y, orig_offset_.z );
  for ( Mesh::VertexIter v_it = mesh.vertices_sbegin(); v_it!=mesh.vertices_end(); ++v_it )
  {
    OpenMesh::DefaultTraits::Point &p = mesh.point ( *v_it );
    p += orig_offset;
  }

  // Iterate over all faces
  for ( Mesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it )
  {
    Mesh::FaceVertexIter fv_it = mesh.fv_iter ( *f_it );
    Point3f p0, p1, p2;

    OpenMesh::DefaultTraits::Point &v0 = mesh.point ( *fv_it++ );

    mesh_model_ptr_->vertex_buffer_data.push_back ( v0[0] );
    mesh_model_ptr_->vertex_buffer_data.push_back ( v0[1] );
    mesh_model_ptr_->vertex_buffer_data.push_back ( v0[2] );

    p0.x = v0[0];
    p0.y = v0[1];
    p0.z = v0[2];

    OpenMesh::DefaultTraits::Point &v1 = mesh.point ( *fv_it++ );

    mesh_model_ptr_->vertex_buffer_data.push_back ( v1[0] );
    mesh_model_ptr_->vertex_buffer_data.push_back ( v1[1] );
    mesh_model_ptr_->vertex_buffer_data.push_back ( v1[2] );

    p1.x = v1[0];
    p1.y = v1[1];
    p1.z = v1[2];

    OpenMesh::DefaultTraits::Point &v2 = mesh.point ( *fv_it++ );

    mesh_model_ptr_->vertex_buffer_data.push_back ( v2[0] );
    mesh_model_ptr_->vertex_buffer_data.push_back ( v2[1] );
    mesh_model_ptr_->vertex_buffer_data.push_back ( v2[2] );

    p2.x = v2[0];
    p2.y = v2[1];
    p2.z = v2[2];

    addLine ( p0, p1 );
    addLine ( p1, p2 );
    addLine ( p2, p0 );
  }

  // cout<<pts_.size()<<endl;

  glDeleteBuffers ( 1, & ( mesh_model_ptr_->vertex_buffer ) );
  glGenBuffers ( 1, & ( mesh_model_ptr_->vertex_buffer ) );
  glBindBuffer ( GL_ARRAY_BUFFER, mesh_model_ptr_->vertex_buffer );
  glBufferData ( GL_ARRAY_BUFFER, sizeof ( GL_FLOAT ) *mesh_model_ptr_->vertex_buffer_data.size(),
                 mesh_model_ptr_->vertex_buffer_data.data(), GL_STATIC_DRAW );

  if( has_color_ )
  {
    if( vertex_color_ == cv::Scalar(-1) )
    {
      for ( Mesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it )
      {
        Mesh::FaceVertexIter fv_it = mesh.fv_iter ( *f_it );

        const OpenMesh::DefaultTraits::Color &c0 = mesh.color( *fv_it++ );

        mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(c0[0])/255.0f );
        mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(c0[1])/255.0f );
        mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(c0[2])/255.0f );

        const OpenMesh::DefaultTraits::Color &c1 = mesh.color( *fv_it++ );

        mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(c1[0])/255.0f );
        mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(c1[1])/255.0f );
        mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(c1[2])/255.0f );

        const OpenMesh::DefaultTraits::Color &c2 = mesh.color( *fv_it++ );

        mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(c2[0])/255.0f );
        mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(c2[1])/255.0f );
        mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(c2[2])/255.0f );

      }
    }
    else
    {
      Scalar v_c = vertex_color_/=255.0;
      for ( Mesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it )
      {
        for( int i = 0; i < 3; i++ )
        {
          mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(v_c[0]));
          mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(v_c[1]));
          mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(v_c[2]));
        }
      }
    }

    glDeleteBuffers ( 1, & ( mesh_model_ptr_->color_buffer ) );
    glGenBuffers ( 1, & ( mesh_model_ptr_->color_buffer ) );
    glBindBuffer ( GL_ARRAY_BUFFER, mesh_model_ptr_->color_buffer );
    glBufferData ( GL_ARRAY_BUFFER, sizeof ( GL_FLOAT ) *mesh_model_ptr_->color_buffer_data.size(),
                  mesh_model_ptr_->color_buffer_data.data(), GL_STATIC_DRAW );
    
    if( light_on_ )
    {
      for ( Mesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it )
      {
        Mesh::Normal face_normal = mesh.calc_face_normal ( *f_it );
        
        mesh_model_ptr_->normal_buffer_data.push_back ( GLfloat(face_normal[0]) );
        mesh_model_ptr_->normal_buffer_data.push_back ( GLfloat(face_normal[1]) );
        mesh_model_ptr_->normal_buffer_data.push_back ( GLfloat(face_normal[2]) );
        
        mesh_model_ptr_->normal_buffer_data.push_back ( GLfloat(face_normal[0]) );
        mesh_model_ptr_->normal_buffer_data.push_back ( GLfloat(face_normal[1]) );
        mesh_model_ptr_->normal_buffer_data.push_back ( GLfloat(face_normal[2]) );
        
        mesh_model_ptr_->normal_buffer_data.push_back ( GLfloat(face_normal[0]) );
        mesh_model_ptr_->normal_buffer_data.push_back ( GLfloat(face_normal[1]) );
        mesh_model_ptr_->normal_buffer_data.push_back ( GLfloat(face_normal[2]) );
        
      }

      glDeleteBuffers ( 1, & ( mesh_model_ptr_->normal_buffer ) );
      glGenBuffers ( 1, & ( mesh_model_ptr_->normal_buffer ) );
      glBindBuffer ( GL_ARRAY_BUFFER, mesh_model_ptr_->normal_buffer );
      glBufferData ( GL_ARRAY_BUFFER, sizeof ( GL_FLOAT ) *mesh_model_ptr_->normal_buffer_data.size(),
                     mesh_model_ptr_->normal_buffer_data.data(), GL_STATIC_DRAW );
            
    }
  }
}


void RasterObjectModel3D::addLine ( Point3f& p0, Point3f& p1 )
{
  Point3f dir = p1 - p0;
  float len = std::sqrt ( dir.dot ( dir ) );
  if ( !len )
    return;

  int n_steps = len/step_;
  Point3f dp ( p0 + epsilon_*dir );
  if ( len >= min_seg_len_ )
  {
    segs_.push_back ( Vec6f ( p0.x, p0.y, p0.z, p1.x, p1.y, p1.z ) );
    d_segs_.push_back ( dp );
  }

  if ( !n_steps )
  {
    // Push at least the first point
    pts_.push_back ( p0 );
    d_pts_.push_back ( dp );
  }
  else
  {
    float dist_step = 1.0/n_steps;

    for ( int i = 0; i <= n_steps; i++ )
    {
      pts_.push_back ( p0 + ( i*dist_step ) *dir );
      d_pts_.push_back ( p0 + ( i*dist_step + epsilon_ ) *dir );
    }
  }
}

void RasterObjectModel3D::addVisibleLine ( Point3f& p0, Point3f& p1 )
{
  Point3f dir = p1 - p0;
  float len = std::sqrt ( dir.dot ( dir ) );
  if ( !len )
    return;

  int init_i = vis_pts_.size();
  bool single_point = false;
  // sub_segs is used to collect unoccluded sub-segments along a segment
  vector< Point3f > sub_segs;

  int n_steps = len/step_;
  if ( !n_steps )
  {
    // Try to push at least one point
    if ( checkPointOcclusion ( p0 ) )
    {
      vis_pts_.push_back ( p0 );
      vis_d_pts_.push_back ( p0 + epsilon_*dir );
      single_point = true;
    }
  }
  else
  {
    float dist_step = 1.0/n_steps;
    bool start_point = true;
    for ( int i = 0; i <= n_steps; i++ )
    {
      Point3f p = p0 + ( i*dist_step ) *dir;
      if ( checkPointOcclusion ( p ) )
      {
        // First point of an unoccluded sub-segment
        if ( start_point )
        {
          sub_segs.push_back ( p );
          start_point = false;
        }
        vis_pts_.push_back ( p );
        vis_d_pts_.push_back ( p+ epsilon_*dir );
      }
      else
      {
        // Last point of an unoccluded sub-segment
        if ( !start_point )
        {
          Point3f end_p = p0 + ( ( i-1 ) *dist_step ) *dir;
          sub_segs.push_back ( end_p );
          start_point = true;
        }
      }
    }
    // In case, "close" the last subsegment
    if ( !start_point )
      sub_segs.push_back ( p1 );
  }

  if ( single_point )
  {
    if ( len >= min_seg_len_ && checkPointOcclusion ( p1 ) )
    {
      vis_segs_.push_back ( Vec6f ( p0.x, p0.y, p0.z, p1.x, p1.y, p1.z ) );
      vis_d_segs_.push_back ( p0 + epsilon_*dir );
    }
  }
  else if ( init_i < int(vis_pts_.size()) )
  {
    // Iterate for each subsegment
    for ( int i = 0; i < int(sub_segs.size()); i += 2 )
    {
      Point3f s_seg = sub_segs[i], e_seg = sub_segs[i+1];
      Point3f diff = s_seg - e_seg;
      len = std::sqrt ( diff.dot ( diff ) );
      if ( len >= min_seg_len_ )
      {
        vis_segs_.push_back ( Vec6f ( s_seg.x, s_seg.y, s_seg.z, e_seg.x, e_seg.y, e_seg.z ) );
        vis_d_segs_.push_back ( s_seg + epsilon_*dir );
      }
    }
  }
}


// bool RasterObjectModel3D::checkPointOcclusion( Point3f& p )
// {
//   Point proj_p;
//   glm::vec4 hp(p.x, p.y, p.z, 1.0f);
//   glm::vec4 proj_hp = mesh_model_ptr_->rt_persp*hp;
//   float depth = proj_hp[3];
//   proj_hp /= proj_hp[3];
//   proj_hp = proj_hp * 0.5f + 0.5f;
//   proj_p.x = cvRound(proj_hp[0] * render_win_size_.width);
//   proj_p.y = render_win_size_.height - cvRound(proj_hp[1] * render_win_size_.height);
//   if( proj_p.x < 1 || proj_p.y < 1 || proj_p.x > render_win_size_.width + 2 || proj_p.y > render_win_size_.height + 2 )
//     return false;
//
//   float closest_depth0 = depth_transf_a_ / (depth_transf_b_ - (2.0f * depth_buffer_.at<float>(proj_p.y, proj_p.x) - 1.0f) * (depth_transf_c_));
//   float closest_depth1 = depth_transf_a_ / (depth_transf_b_ - (2.0f * depth_buffer_.at<float>(proj_p.y + 1, proj_p.x) - 1.0f) * (depth_transf_c_));
//   float closest_depth2 = depth_transf_a_ / (depth_transf_b_ - (2.0f * depth_buffer_.at<float>(proj_p.y - 1, proj_p.x) - 1.0f) * (depth_transf_c_));
//   float closest_depth3 = depth_transf_a_ / (depth_transf_b_ - (2.0f * depth_buffer_.at<float>(proj_p.y, proj_p.x + 1) - 1.0f) * (depth_transf_c_));
//   float closest_depth4 = depth_transf_a_ / (depth_transf_b_ - (2.0f * depth_buffer_.at<float>(proj_p.y, proj_p.x - 1) - 1.0f) * (depth_transf_c_));
//
//   if( depth <= closest_depth0 + depth_buffer_epsilon_ ||
//       depth <= closest_depth1 + depth_buffer_epsilon_ ||
//       depth <= closest_depth2 + depth_buffer_epsilon_ ||
//       depth <= closest_depth3 + depth_buffer_epsilon_ ||
//       depth <= closest_depth4 + depth_buffer_epsilon_ )
//   {
//     // color_debug_img.at<Vec3b>(proj_p.y, proj_p.x) = Vec3b(0,0,255);
//     return true;
//   }
//   else
//     return false;
// }

inline int RasterObjectModel3D::projectPointToGLPersp ( const Point3f& p, Point& proj_p, float& depth )
{
  glm::vec4 hp ( p.x, p.y, p.z, 1.0f );
  glm::vec4 proj_hp = mesh_model_ptr_->rt_persp*hp;

  Point2f tmp_p( proj_hp[0]/proj_hp[3], proj_hp[1]/proj_hp[3] );

  int i = denormalizePoint(tmp_p, proj_p);
  if( i >= 0 ) depth = proj_hp[3];
  return i;
}

inline int RasterObjectModel3D::denormalizePoint(const cv::Point2f& p, cv::Point& dp )
{  
  Point2f tmp_p(p.x*0.5f + 0.5f,p.y*0.5f + 0.5f);

  dp.x = cvRound ( tmp_p.x * render_win_size_.width ) - buffer_data_roi_.x;
  dp.y = cvRound ( tmp_p.y * render_win_size_.height ) - buffer_data_roi_.y;

  // WARNING Workaround to check the neighbours
  if ( dp.x < 1 || dp.y < 1 || dp.x > buffer_data_roi_.width - 2 || dp.y > buffer_data_roi_.height - 2 )
    return -1;

  return dp.y*buffer_data_roi_.width + dp.x;
}

Mat RasterObjectModel3D::getRenderedModel( Scalar background_color )
{
  if( !raster_updated_ )
    updateRaster();

  int rows = cam_model_.imgHeight(), cols = cam_model_.imgWidth();
  Mat render_img( Size(cols, rows) , DataType<Vec3b>::type, 
                  Scalar(background_color[2],background_color[1],background_color[0]) );
  for( int r = 0; r < rows; r++ )
  {
    int *map_p = img2gl_map_.ptr<int>(r);
    Vec3b *img_p = render_img.ptr<Vec3b>(r);

    for( int c = 0; c < cols; c++, map_p++, img_p++ )
    {
      int idx = *map_p;
      if( idx >= 0 && depth_buffer_data_[idx] < 1.0f )
        *img_p = *(reinterpret_cast<Vec3b*>(&(color_buffer_data_[idx])));
    }
  }
  return render_img;
}

Mat RasterObjectModel3D::getModelDepthMap()
{
  if( !raster_updated_ )
    updateRaster();

  int rows = cam_model_.imgHeight(), cols = cam_model_.imgWidth();
  Mat depth_img( rows, cols, DataType<float>::type);

  for( int r = 0; r < rows; r++ )
  {
    int *map_p = img2gl_map_.ptr<int>(r);
    float *img_p = depth_img.ptr<float>(r);

    for( int c = 0; c < cols; c++, map_p++, img_p++ )
    {
      int idx = *map_p;
      float d;
      if( idx >= 0 && ( d = depth_buffer_data_[idx]) < 1.0f )
        *img_p = depth_transf_a_ / ( depth_transf_b_ - ( 2.0f * d - 1.0f ) * depth_transf_c_ );
      else
        *img_p = -1;
    }
  }
  return depth_img;
}

// TODO Optimize this!
bool RasterObjectModel3D::checkPointOcclusion ( Point3f& p )
{
  Point proj_p;
  float depth;
  int idx0 = projectPointToGLPersp ( p, proj_p, depth );
  if ( idx0 == -1 )
    return false;

  int idx1 = (proj_p.y + 1)*buffer_data_roi_.width + proj_p.x;
  int idx2 = (proj_p.y - 1)*buffer_data_roi_.width + proj_p.x;
  int idx3 = (proj_p.y)*buffer_data_roi_.width + proj_p.x + 1;
  int idx4 = (proj_p.y)*buffer_data_roi_.width + proj_p.x - 1;
  
  // TODO Create a macro here?
  float closest_depth0 = depth_transf_a_ / ( depth_transf_b_ - ( 2.0f * depth_buffer_data_[idx0] - 1.0f ) * depth_transf_c_ );
  float closest_depth1 = depth_transf_a_ / ( depth_transf_b_ - ( 2.0f * depth_buffer_data_[idx1] - 1.0f ) * depth_transf_c_ );
  float closest_depth2 = depth_transf_a_ / ( depth_transf_b_ - ( 2.0f * depth_buffer_data_[idx2] - 1.0f ) * depth_transf_c_ );
  float closest_depth3 = depth_transf_a_ / ( depth_transf_b_ - ( 2.0f * depth_buffer_data_[idx3] - 1.0f ) * depth_transf_c_ );
  float closest_depth4 = depth_transf_a_ / ( depth_transf_b_ - ( 2.0f * depth_buffer_data_[idx4] - 1.0f ) * depth_transf_c_ );

  if ( depth <= closest_depth0 + depth_buffer_epsilon_ ||
       depth <= closest_depth1 + depth_buffer_epsilon_ ||
       depth <= closest_depth2 + depth_buffer_epsilon_ ||
       depth <= closest_depth3 + depth_buffer_epsilon_ ||
       depth <= closest_depth4 + depth_buffer_epsilon_ )
  {
    // color_debug_img.at<Vec3b>(proj_p.y, proj_p.x) = Vec3b(0,0,255);
    return true;
  }
  else
    return false;
}

void RasterObjectModel3D::retreiveModel ( int idx )
{
  vis_pts_p_ = &(precomputed_vis_pts_[idx]);
  vis_d_pts_p_ = &(precomputed_vis_d_pts_[idx]);
  vis_segs_p_ = &(precomputed_vis_segs_[idx]);
  vis_d_segs_p_ = &(precomputed_vis_d_segs_[idx]);

  if ( int(precomputed_depth_buffer_data_.size()) > idx )
    depth_buffer_data_=precomputed_depth_buffer_data_[idx];
  
  raster_updated_ = false;
}

void RasterObjectModel3D::storeModel()
{
  precomputed_vis_pts_.push_back ( vis_pts_ );
  precomputed_vis_d_pts_.push_back ( vis_d_pts_ );
  precomputed_vis_segs_.push_back ( vis_segs_ );
  precomputed_vis_d_segs_.push_back ( vis_d_segs_ );

  if ( depth_buffer_storing_enabled_ )
    precomputed_depth_buffer_data_.push_back ( depth_buffer_data_ );
}

void RasterObjectModel3D::loadPrecomputedModels ( FileStorage& fs )
{
  precomputed_vis_pts_.clear();
  precomputed_vis_d_pts_.clear();
  precomputed_vis_segs_.clear();
  precomputed_vis_d_segs_.clear();

  int n_precomputed_views = precomputed_rq_view_.size();

  for ( int i = 0; i < n_precomputed_views; i++ )
  {
    precomputed_vis_pts_.push_back ( vector<Point3f>() );
    precomputed_vis_d_pts_.push_back ( vector<Point3f>() );

    Mat model_views_pts;
    stringstream sname;
    sname<<"model_views_pts_"<<i;
    fs[sname.str().data()] >> model_views_pts;

    for ( int j = 0; j < model_views_pts.rows; j++ )
    {
      Point3f pt, d_pt;
      pt.x = model_views_pts.at<float> ( j,0 );
      pt.y = model_views_pts.at<float> ( j,1 );
      pt.z = model_views_pts.at<float> ( j,2 );

      d_pt.x = model_views_pts.at<float> ( j,3 );
      d_pt.y = model_views_pts.at<float> ( j,4 );
      d_pt.z = model_views_pts.at<float> ( j,5 );

      precomputed_vis_pts_[i].push_back ( pt );
      precomputed_vis_d_pts_[i].push_back ( d_pt );
    }

    precomputed_vis_segs_.push_back ( vector<Vec6f>() );
    precomputed_vis_d_segs_.push_back ( vector<Point3f>() );

    sname.clear();
    sname<<"model_views_segs_"<<i;
    Mat model_views_segs;
    fs[sname.str().data()] >> model_views_segs;

    for ( int j = 0; j < model_views_segs.rows; j++ )
    {
      Vec6f seg;
      Point3f d_seg;
      seg[0] = model_views_segs.at<float> ( j,0 );
      seg[1] = model_views_segs.at<float> ( j,1 );
      seg[2] = model_views_segs.at<float> ( j,2 );
      seg[3] = model_views_segs.at<float> ( j,3 );
      seg[4] = model_views_segs.at<float> ( j,4 );
      seg[5] = model_views_segs.at<float> ( j,5 );

      d_seg.x = model_views_pts.at<float> ( j,6 );
      d_seg.y = model_views_pts.at<float> ( j,7 );
      d_seg.z = model_views_pts.at<float> ( j,8 );

      precomputed_vis_segs_[i].push_back ( seg );
      precomputed_vis_d_segs_[i].push_back ( d_seg );
    }
  }
}

void RasterObjectModel3D::savePrecomputedModels ( FileStorage& fs ) const
{
  for ( int i = 0; i < int(precomputed_vis_pts_.size()); i++ )
  {
    const vector<Point3f> &pts = precomputed_vis_pts_[i],
                                    &d_pts = precomputed_vis_d_pts_[i];
    Mat model_views_pts ( precomputed_vis_pts_[i].size(), 6, DataType<float>::type );
    for ( int j = 0; j < int(precomputed_vis_pts_[i].size()); j++ )
    {
      model_views_pts.at<float> ( j,0 ) = pts[j].x;
      model_views_pts.at<float> ( j,1 ) = pts[j].y;
      model_views_pts.at<float> ( j,2 ) = pts[j].z;

      model_views_pts.at<float> ( j,3 ) = d_pts[j].x;
      model_views_pts.at<float> ( j,4 ) = d_pts[j].y;
      model_views_pts.at<float> ( j,5 ) = d_pts[j].z;
    }

    stringstream sname;
    sname<<"model_views_pts_"<<i;
    fs << sname.str().data() << model_views_pts;

    const vector<Vec6f> &segs = precomputed_vis_segs_[i];
    const vector<Point3f> &d_segs = precomputed_vis_d_segs_[i];
    Mat model_views_segs ( precomputed_vis_segs_[i].size(), 9, DataType<float>::type );
    for ( int j = 0; j < int(precomputed_vis_segs_[i].size()); j++ )
    {
      model_views_segs.at<float> ( j,0 ) = segs[j][0];
      model_views_segs.at<float> ( j,1 ) = segs[j][1];
      model_views_segs.at<float> ( j,2 ) = segs[j][2];
      model_views_segs.at<float> ( j,3 ) = segs[j][3];
      model_views_segs.at<float> ( j,4 ) = segs[j][4];
      model_views_segs.at<float> ( j,5 ) = segs[j][5];

      model_views_segs.at<float> ( j,6 ) = d_segs[j].x;
      model_views_segs.at<float> ( j,7 ) = d_segs[j].y;
      model_views_segs.at<float> ( j,8 ) = d_segs[j].z;
    }
    sname.clear();
    sname<<"model_views_segs_"<<i;
    fs << sname.str().data() << model_views_segs;
  }
}

Mat RasterObjectModel3D::getMask()
{
  if( !raster_updated_ )
    updateRaster();

  int rows = cam_model_.imgHeight(), cols = cam_model_.imgWidth();
  Mat mask( Size(cols, rows) , DataType<uchar>::type, Scalar(0));
  for( int r = 0; r < rows; r++ )
  {
    int *map_p = img2gl_map_.ptr<int>(r);
    uchar *img_p = mask.ptr<uchar>(r);

    for( int c = 0; c < cols; c++, map_p++, img_p++ )
    {
      int idx = *map_p;
      if( idx >= 0 && depth_buffer_data_[idx] < 1.0f )
        *img_p = 255;
    }
  }
  return mask;
}