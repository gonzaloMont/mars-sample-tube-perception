#include <cstdio>
#include <string>
#include <sstream>
#include <boost/program_options.hpp>

#include "cv_ext/cv_ext.h"
#include "raster_object_model3D.h"
#include "raster_object_model2D.h"

#include "apps_utils.h"

namespace po = boost::program_options;
using namespace std;
using namespace cv;

static void quickGuide()
{
  cout << "Use the keyboard to move the object model:" << endl<< endl;
  
  objectPoseControlsHelp();
}

int main(int argc, char **argv)
{
  string app_name( argv[0] ), model_filename, pre_models_filename, 
        camera_filename, image_filename;
  double scale_factor = 1.0;
  int top_boundary = -1, bottom_boundary = -1, left_boundary = -1, rigth_boundary = -1;
  double delta_rp = -1, rp_step = 0, 
         delta_yaw = -1, yaw_step = 0,
         delta_z = 0, z_step = 0;
  
  po::options_description desc ( "OPTIONS" );
  desc.add_options()
  ( "help,h", "Print this help messages" )
  ( "model_filename,m", po::value<string > ( &model_filename )->required(),
    "STL, PLY, OBJ, ...  model file" )
  ( "pre_models_filename,p", po::value<string > ( &pre_models_filename)->required(),
    "Input/output precomputed models file" )
  ( "camera_filename,c", po::value<string > ( &camera_filename )->required(),
    "A YAML file that stores all the camera parameters (see the PinholeCameraModel object)" )
  ( "image_filename,i", po::value<string > ( &image_filename ),
    "Optional background image file" )
  ( "scale_factor,s", po::value<double > ( &scale_factor ),
    "Optional scale factor" )
  ( "d_rp", po::value<double > ( &delta_rp ),
    "Roll/pitch tollerance [0-90, or -1 full dimensions] in degrees (i.e., current view +/- drp), used to generate a neighborhood of views" )
  ( "rp_s", po::value<double > ( &rp_step ),
    "Roll/pitch sample step, used to generate a neighborhood of views" )
  ( "d_yaw", po::value<double > ( &delta_yaw ),
    "Yaw tollerance [0-90, or -1 full dimension] in degrees (i.e., current view +/- dyaw), used to generate a neighborhood of views" )
  ( "yaw_s", po::value<double > ( &yaw_step ),
    "Yaw sample step in degrees, used to generate a neighborhood of views" )
  ( "d_z", po::value<double > ( &delta_z ),
    "Z axis tollerance in meters (i.e., current view +/- dz), used to generate a neighborhood of views" )
  ( "z_s", po::value<double > ( &z_step ),
    "Z axis sample step in meters, used to generate a neighborhood of views" )  
  ( "tb", po::value<int> ( &top_boundary ),
    "Optional region of interest: top boundary " )
  ( "bb", po::value<int> ( &bottom_boundary ),
    "Optional region of interest: bottom boundary " )
  ( "lb", po::value<int> ( &left_boundary ),
  "Optional region of interest: left boundary " )  
  ( "rb", po::value<int> ( &rigth_boundary ),
  "Optional region of interest: rigth boundary" );

  po::variables_map vm;
  try
  {
    po::store ( po::parse_command_line ( argc, argv, desc ), vm );

    if ( vm.count ( "help" ) )
    {
      cout << "USAGE: "<<app_name<<" OPTIONS"
                << endl << endl<<desc;
      return 0;
    }

    po::notify ( vm );
  }

  catch ( boost::program_options::required_option& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    cout << "USAGE: "<<app_name<<" OPTIONS"
              << endl << endl<<desc;
    return -1;
  }

  catch ( boost::program_options::error& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    cout << "USAGE: "<<app_name<<" OPTIONS"
              << endl << endl<<desc;
    return -1;
  }
  
  Mat r_vec = (Mat_<double>(3,1) << 0,0,0),
      t_vec = (Mat_<double>(3,1) << 0,0,1.0);

  cv_ext::PinholeCameraModel cam_model;
  cam_model.readFromFile(camera_filename);
  cam_model.setSizeScaleFactor(scale_factor);
  int img_w = cam_model.imgWidth(), img_h = cam_model.imgHeight();
  
  bool has_roi = false;
  cv::Rect roi;
  
  if( top_boundary != -1 || bottom_boundary != -1 || 
      left_boundary != -1 || rigth_boundary != -1 )
  {
    Point tl(0,0), br(img_w, img_h);
    
    if( top_boundary != -1 ) tl.y = top_boundary;
    if( left_boundary != -1 ) tl.x = left_boundary;
    if( bottom_boundary != -1 ) br.y = bottom_boundary;
    if( rigth_boundary != -1 ) br.x = rigth_boundary;
    
    has_roi = true;
    roi = cv::Rect(tl, br);
    cam_model.setRegionOfInterest(roi);
    cam_model.enableRegionOfInterest(true);
    roi = cam_model.regionOfInterest();
  }
   
  cout << "Loading model from file : "<<model_filename<< endl;
  cout << "Loading precomputed models from file : "<<pre_models_filename<< endl;  
  cout << "Loading camera parameters from file : "<<camera_filename<< endl;
  if( !image_filename.empty() )
    cout << "Loading background image from file : "<<image_filename<< endl;
  cout << "Scale factor : "<<scale_factor<< endl;
  if(has_roi)
    cout << "Region of interest : "<<roi<< endl;
  
  if( delta_rp > 90 ) delta_rp = 90;
  if( delta_yaw > 90 ) delta_yaw = 90;
  if( rp_step < 0 ) rp_step = 0;
  if( yaw_step < 0 ) yaw_step = 0;
  
  cout << "Roll/pitch tollerance : ";
  if( delta_rp < 0 )
    cout << "full dimensions";
  else
    cout << delta_rp;
  cout <<" sample step "<<rp_step<<" [degrees]"<< endl;
  cout << "Yaw tollerance : ";
  if( delta_yaw < 0 )
    cout <<"full dimension";
  else
    cout <<delta_yaw;
  cout <<" sample step "<<yaw_step<<" [degrees]"<< endl;
  cout << "Z tollerance : "<<delta_z<<" sample step "<<z_step<<" [meters]"<< endl;

  delta_rp *= M_PI/180.0; rp_step *= M_PI/180.0;
  delta_yaw *= M_PI/180.0; yaw_step *= M_PI/180.0;
  delta_z *= M_PI/180.0; z_step *= M_PI/180.0;
         
  RasterObjectModel3D obj_model;

  obj_model.setCamModel( cam_model );
  obj_model.setStepMeters(0.001);
  obj_model.setUnitOfMeasure(RasterObjectModel::MILLIMETER);

  if(!obj_model.setModelFile( model_filename ))
    return -1;

  // obj_model.setMinSegmentsLen(0.01);
  obj_model.computeRaster();
  obj_model.loadPrecomputedModelsViews(pre_models_filename);
  

  
  cv::Mat background_img;
  if(!image_filename.empty())
  {
    background_img = cv::imread(image_filename, cv::IMREAD_COLOR);
    cv::resize(background_img, background_img, Size(img_w, img_h));
  }
  
  vector<Vec4f> raster_segs;

  bool view_mode = obj_model.numPrecomputedModelsViews() != 0, 
       draw_axis = false, store_model = false, store_multi_model = false, save_models = false;
  int cur_pre_model = 0;
  
  quickGuide();

  bool exit_now = false;
  Point text_org_guide1 (20, 20), text_org_guide2(20, 40), text_org_model (20, 70),
        text_org_x(20, img_h - 80), text_org_y(20, img_h - 60), text_org_z(20, img_h - 40);
  Scalar text_color;
  
  while( !exit_now )
  { 
    stringstream sstr_guide1, sstr_guide2, sstr_model;
    if(view_mode)
    {
      sstr_guide1<<"Use [9] an [0] to switch the previuois and next models, [SPACE] to add more views";
      sstr_guide2<<"";
      obj_model.setModelView(cur_pre_model);
      sstr_model<<"VIEW MODE - Model ";
      sstr_model<<cur_pre_model + 1;
      sstr_model<<" of ";
      sstr_model<<obj_model.numPrecomputedModelsViews();
      
      text_color = Scalar(0x44,0x62,0x35);
    }
    else
    {
      sstr_guide1<<"Use [ENTER] to store the current model, [9] to store a set of views close to the current one, [0] to save all the views,";
      sstr_guide2<<"[SPACE] to show all stored models";
      obj_model.setModelView(r_vec, t_vec);
      
      if( store_model )
      {
        obj_model.storeModelView();
        store_model = false;
      }
      else if( store_multi_model )
      {
        Eigen::Quaterniond ref_qr;
        Eigen::Vector3d ref_t;
        obj_model.modelView(ref_qr,ref_t);
        std::vector<cv::Point3d> cap_points;
        cv_ext::vector_Quaterniond quat_rotations;
        
        if( rp_step )
        {
          if( delta_rp < 0 )
            cv_ext::createIcosphere( cap_points, rp_step, 1.0 );
          else
            cv_ext::createIcospherePolarCap( cap_points, rp_step, M_PI/2 - delta_rp, 1.0, true );
          
          cout<<cap_points.size()<<endl;
          if( cap_points.size() )
          {
            if( delta_yaw != 0 && yaw_step != 0 )
            {
              if( delta_yaw <= 0 )
              {
                int num_rotations = round(2*M_PI/yaw_step);
                cv_ext::sampleRotationsAroundVectors( cap_points, quat_rotations, 
                                                      num_rotations , cv_ext::COORDINATE_Z_AXIS );
              }
              else
                cv_ext::sampleRotationsAroundVectors( cap_points, quat_rotations, 
                                                      -delta_yaw, delta_yaw, yaw_step,
                                                      cv_ext::COORDINATE_Z_AXIS );
            }
            else
              cv_ext::sampleRotationsAroundVectors( cap_points, quat_rotations, 1, cv_ext::COORDINATE_Z_AXIS );
          
            cout<<quat_rotations.size()<<endl;
            Eigen::Isometry3d cur_tr, ref_tr, final_tr;
            
            int n_z_step;
            
            if( delta_z != 0 && z_step != 0 )
            {
              n_z_step = cvRound(2.0*delta_z/z_step);
              z_step = 2.0*delta_z/double(n_z_step);
            }
            else
              n_z_step = 0;
            
            ref_t(2) -= delta_z;
            for(int i_z = 0; i_z <= n_z_step; i_z++, ref_t(2)+= z_step )
            {
              ref_tr.fromPositionOrientationScale( ref_t, ref_qr, Eigen::Vector3d::Ones());
              for( auto &q : quat_rotations)
              {
                cur_tr.fromPositionOrientationScale( Eigen::Vector3d::Zero(), q, Eigen::Vector3d::Ones());
                final_tr = ref_tr*cur_tr;
                obj_model.setModelView( Eigen::Quaterniond(final_tr.rotation()), ref_t );
                obj_model.storeModelView();  
              }
            }
          }
        }
        store_multi_model = false;
      }

      sstr_model<<"EDIT MODE - Stored ";
      sstr_model<<obj_model.numPrecomputedModelsViews();
      sstr_model<<" models";
      
      if(save_models)
      {
        obj_model.savePrecomputedModelsViews(pre_models_filename);
        save_models = false;
        sstr_model<<" (Saved)";
      }
      
      text_color = Scalar(0x1A,0x2A,0xAD);
    }
    
    Mat background, draw_img;
    if( background_img.empty() )
      background = Mat( Size(img_w,img_h),
                        DataType<Vec3b>::type, CV_RGB( 0,0,0));
    else
      background = background_img.clone();

    if( has_roi )
      draw_img = background(roi);
    else
      draw_img = background;
    
    obj_model.projectRasterSegments( raster_segs );
    cv_ext::drawSegments( draw_img, raster_segs );
        
    if( draw_axis )
    {
      vector< Vec4f > proj_segs_x, proj_segs_y,  proj_segs_z;
      obj_model.projectAxes ( proj_segs_x, proj_segs_y, proj_segs_z );

      cv_ext::drawSegments( draw_img, proj_segs_x, Scalar(0, 0, 255) );
      cv_ext::drawSegments( draw_img, proj_segs_y, Scalar(0, 255, 0) );
      cv_ext::drawSegments( draw_img, proj_segs_z, Scalar(255, 0, 0) );
    }
    
    if( has_roi )
    {
      cv::Point dbg_tl = roi.tl(), dbg_br = roi.br();
      dbg_tl.x -= 1; dbg_tl.y -= 1;
      dbg_br.x += 1; dbg_br.y += 1;
      cv::rectangle( background, dbg_tl, dbg_br, cv::Scalar(255,255,255));
    }
    
    putText(background, sstr_guide1.str(), text_org_guide1, CV_FONT_HERSHEY_PLAIN, 1, 
        text_color, 1, 8);
    putText(background, sstr_guide2.str(), text_org_guide2, CV_FONT_HERSHEY_PLAIN, 1, 
        text_color, 1, 8);
    putText(background, sstr_model.str(), text_org_model, CV_FONT_HERSHEY_SIMPLEX, 1, 
            text_color, 1, 8);
    stringstream sstr_x, sstr_y, sstr_z;
    sstr_x<<"X:"; sstr_y<<"Y:"; sstr_z<<"Y:";
    sstr_x<<t_vec.at<double>(0); sstr_y<<t_vec.at<double>(1); sstr_z<<t_vec.at<double>(2);
    putText(background, sstr_x.str(), text_org_x, CV_FONT_HERSHEY_PLAIN, 1, text_color, 1, 8);
    putText(background, sstr_y.str(), text_org_y, CV_FONT_HERSHEY_PLAIN, 1, text_color, 1, 8);
    putText(background, sstr_z.str(), text_org_z, CV_FONT_HERSHEY_PLAIN, 1, text_color, 1, 8);
    
    imshow("Generate Models", background);
    int key = cv_ext::waitKeyboard();
    
    if( !view_mode )
    {
      parseObjectPoseControls( key, r_vec, t_vec );
      switch(key)
      {
        case '\n':
          store_model = true;
          break;
        case '9':
          store_multi_model = true;
          break;             
        case '0':
          save_models = true;
          break;
      }      
    }
    else
    {
      switch(key)
      {      
        case '0':
          cur_pre_model++;
          if( cur_pre_model >= obj_model.numPrecomputedModelsViews() )
            cur_pre_model = 0;
          break;
        case '9':
          cur_pre_model--;
          if( cur_pre_model < 0 )
            cur_pre_model = obj_model.numPrecomputedModelsViews() - 1;
          break;
      }
    }
    
    switch(key)
    {
      case 'r':
        draw_axis = !draw_axis;
        break;
      case ' ':
        view_mode = !view_mode && obj_model.numPrecomputedModelsViews();
        break;        
      case cv_ext::KEY_ESCAPE:
        exit_now = true;
        break;
    }
  }

  return 0;
}


