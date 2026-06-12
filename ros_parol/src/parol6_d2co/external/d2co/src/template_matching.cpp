#include "template_matching.h"

using namespace std;
using namespace cv;

TemplateMatching :: TemplateMatching () : 
  optimizer_num_iterations_(50),
  verbouse_mode_(false),
  parallelism_enabled_(false),
  update_optimizer_(false)
{
  transf_ << 1.0, 0, 0, 0, 0, 0, 0, 0;
}

void TemplateMatching::setTemplateModel ( const RasterObjectModelPtr& model_ptr )
{
  if( !model_ptr )
    throw runtime_error("Null RasterObjectModel");
  
  model_ptr_ = model_ptr;
  cam_model_ = model_ptr->cameraModel();
  
  /* camera_matrix_ and dist_coeff_  are redundant information used for efficiency issuess
   * i.e., to use directly openCV functions (e.g., solvePnP()) */
  camera_matrix_ = cam_model_.cameraMatrix();
  if( cam_model_.hasDistCoeff() )
    dist_coeff_ = cam_model_.distorsionCoeff();
  else
    dist_coeff_ = Mat();
  
  update_optimizer_ = true;
}

double TemplateMatching::refinePosition( double r_quat[4], double t_vec[3] )
{
  if( !model_ptr_ )
    throw runtime_error("RasterObjectModel not set");

  setPos( r_quat, t_vec );
  if(update_optimizer_ || !model_ptr_->allVisiblePoints())
  {
    model_ptr_->setModelView(transf_.data(), transf_.block<3,1>(4,0).data()); 
    updateOptimizer(-1);
    update_optimizer_ = false;
  }
  double res = optimize();
  getPos( r_quat, t_vec );
  
  return res;
}

double TemplateMatching::refinePosition ( Eigen::Quaterniond& r_quat,
                                          Eigen::Vector3d& t_vec )
{
  if( !model_ptr_ )
    throw runtime_error("RasterObjectModel not set");

  setPos( r_quat, t_vec );
  if(update_optimizer_ || !model_ptr_->allVisiblePoints())
  {
    model_ptr_->setModelView(transf_.data(), transf_.block<3,1>(4,0).data()); 
    updateOptimizer(-1);
    update_optimizer_ = false;
  }
  double res = optimize();
  getPos( r_quat, t_vec );
  
  return res;
}

double TemplateMatching::refinePosition( Mat_<double> &r_vec, Mat_<double> &t_vec )
{
  if( !model_ptr_ )
    throw runtime_error("RasterObjectModel not set");

  setPos( r_vec, t_vec );
  if(update_optimizer_ || !model_ptr_->allVisiblePoints())
  {
    model_ptr_->setModelView(transf_.data(), transf_.block<3,1>(4,0).data()); 
    updateOptimizer(-1);
    update_optimizer_ = false;
  }
  double res = optimize();
  getPos( r_vec, t_vec );
  
  return res;
}

double TemplateMatching::refinePosition ( int idx, double r_quat[4], double t_vec[3] )
{
  if( !model_ptr_ )
    throw runtime_error("RasterObjectModel not set");

  Eigen::Quaterniond quat;
  Eigen::Vector3d t;
  model_ptr_->modelView(idx, quat, t);
  setPos ( quat, t );
  
  if(update_optimizer_ || !model_ptr_->allVisiblePoints())
  {
    updateOptimizer(idx);
    update_optimizer_ = false;
  }
  
  double res = optimize();
  getPos (r_quat, t_vec );
  
  return res;
}

double TemplateMatching::refinePosition ( int idx, Eigen::Quaterniond& r_quat,
                                               Eigen::Vector3d& t_vec  )
{
  if( !model_ptr_ )
    throw runtime_error("RasterObjectModel not set");

  Eigen::Quaterniond quat;
  Eigen::Vector3d t;
  model_ptr_->modelView(idx, quat, t);
  setPos ( quat, t );
  
  if(update_optimizer_ || !model_ptr_->allVisiblePoints())
  {
    updateOptimizer(idx);
    update_optimizer_ = false;
  }
  
  double res = optimize();
  getPos (r_quat, t_vec );
  
  return res;
}

double TemplateMatching::refinePosition ( int idx, Mat_< double >& r_vec,
                                               Mat_< double >& t_vec )
{
  if( !model_ptr_ )
    throw runtime_error("RasterObjectModel not set");

  Eigen::Quaterniond quat;
  Eigen::Vector3d t;
  model_ptr_->modelView(idx, quat, t);
  setPos ( quat, t );
  
  if(update_optimizer_ || !model_ptr_->allVisiblePoints())
  {
    updateOptimizer(idx);
    update_optimizer_ = false;
  }
  
  double res = optimize();
  getPos (r_vec, t_vec );
  
  return res;
}

double TemplateMatching::getAvgDistance( const double r_quat[4], const double t_vec[3] )
{
  if( !model_ptr_ )
    throw runtime_error("RasterObjectModel not set");

  setPos( r_quat, t_vec );
  if(!model_ptr_->allVisiblePoints())
    model_ptr_->setModelView(transf_.data(), transf_.block<3,1>(4,0).data()); 
    
  return avgDistance(-1);
}

double TemplateMatching::getAvgDistance( const Eigen::Quaterniond& r_quat, 
                                         const Eigen::Vector3d& t_vec )
{
  if( !model_ptr_ )
    throw runtime_error("RasterObjectModel not set");

  setPos( r_quat, t_vec );
  if(!model_ptr_->allVisiblePoints())
    model_ptr_->setModelView(transf_.data(), transf_.block<3,1>(4,0).data()); 
  
  return avgDistance(-1);
}


double TemplateMatching::getAvgDistance( const Mat_< double >& r_vec, 
                                         const Mat_< double >& t_vec )
{
  if( !model_ptr_ )
    throw runtime_error("RasterObjectModel not set");

  setPos( r_vec, t_vec );
  if(!model_ptr_->allVisiblePoints())
    model_ptr_->setModelView(transf_.data(), transf_.block<3,1>(4,0).data()); 
  
  return avgDistance(-1);
}

double TemplateMatching::getAvgDistance( int idx )
{
  if( !model_ptr_ )
    throw runtime_error("RasterObjectModel not set");

  Eigen::Quaterniond quat;
  Eigen::Vector3d t;
  model_ptr_->modelView(idx, quat, t);
  setPos ( quat, t );
  
  return avgDistance(idx);
}

void TemplateMatching::setPos ( const double r_quat[4], const double t_vec[3] )
{
  transf_(0,0) = r_quat[0];
  transf_(1,0) = r_quat[1];
  transf_(2,0) = r_quat[2];
  transf_(3,0) = r_quat[3];

  transf_(4,0) = t_vec[0];
  transf_(5,0) = t_vec[1];
  transf_(6,0) = t_vec[2];
}

void TemplateMatching::setPos ( const Eigen::Quaterniond& r_quat, 
                                const Eigen::Vector3d& t_vec )
{
  cv_ext::eigenQuat2Quat( r_quat, transf_.data() );
  transf_(4,0) = t_vec[0];
  transf_(5,0) = t_vec[1];
  transf_(6,0) = t_vec[2];
}

void TemplateMatching::setPos ( const Mat_< double >& r_vec, 
                                const Mat_< double >& t_vec )
{
  double angle_axis[3] = { r_vec ( 0,0 ) , r_vec ( 1,0 ) , r_vec ( 2,0 ) };
  ceres::AngleAxisToQuaternion<double> ( angle_axis, transf_.data() );

  transf_(4,0) = t_vec ( 0,0 );
  transf_(5,0) = t_vec ( 1,0 );
  transf_(6,0) = t_vec ( 2,0 );
}

void TemplateMatching::getPos ( double r_quat[4], double t_vec[3] ) const
{
  r_quat[0] = transf_(0,0);
  r_quat[1] = transf_(1,0);
  r_quat[2] = transf_(2,0);
  r_quat[3] = transf_(3,0);

  t_vec[0] = transf_(4,0);
  t_vec[1] = transf_(5,0);
  t_vec[2] = transf_(6,0);
}

void TemplateMatching::getPos ( Eigen::Quaterniond& r_quat, 
                                Eigen::Vector3d& t_vec ) const
{
  cv_ext::quat2EigenQuat(transf_.data(), r_quat );
  t_vec( 0,0 ) = transf_(4,0);
  t_vec( 1,0 ) = transf_(5,0);
  t_vec( 2,0 ) = transf_(6,0);
}


void TemplateMatching::getPos ( Mat_< double >& r_vec, 
                                Mat_< double >& t_vec ) const
{
  double angle_axis[3];
  ceres::QuaternionToAngleAxis<double> ( transf_.data(), angle_axis );
  
  r_vec ( 0,0 ) = angle_axis[0];
  r_vec ( 1,0 ) = angle_axis[1];
  r_vec ( 2,0 ) = angle_axis[2];

  t_vec ( 0,0 ) = transf_(4,0);
  t_vec ( 1,0 ) = transf_(5,0);
  t_vec ( 2,0 ) = transf_(6,0);
}

void TemplateMatching::normalizeMatch( TemplateMatch &m )
{
  if( m.img_offset == Point(0,0) )
    return;
  vector <Point3f> obj_pts = model_ptr_->getPrecomputedPoints(m.id);
  vector <Point2f> proj_pts;
  Point2f off_p(m.img_offset.x, m.img_offset.y);
  
  model_ptr_->projectRasterPoints( m.id, proj_pts );
  for( auto &p : proj_pts )
    p += off_p;
  
  cv::Mat r_vec(3,1,cv::DataType<double>::type);
  cv::Mat t_vec(3,1,cv::DataType<double>::type);
  
  cv::solvePnP( obj_pts, proj_pts, camera_matrix_, dist_coeff_, r_vec, t_vec );
  
  double r_quat[4];
  ceres::AngleAxisToQuaternion( (const double*)(r_vec.data), r_quat) ;
  m.r_quat = Eigen::Quaterniond( r_quat[0], r_quat[1], r_quat[2], r_quat[3] );
  m.t_vec = Eigen::Map< const Eigen::Vector3d>( (const double*)(t_vec.data) );
  m.img_offset = Point(0,0);
}
