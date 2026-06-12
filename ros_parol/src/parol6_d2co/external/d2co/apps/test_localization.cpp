#include <iostream>
#include <fstream>
#include <string>
#include <boost/program_options.hpp>

#include "cv_ext/cv_ext.h"
#include "raster_object_model3D.h"
#include "chamfer_matching.h"
#include "direct_matching.h"
#include "apps_utils.h"


namespace po = boost::program_options;
using namespace std;
using namespace cv;
using namespace cv_ext;

static const double OUTER_POINT_SCORE = 0.6;


double evaluateScore ( cv_ext::ImageStatisticsPtr &img_stats_p,
                       vector<cv::Point2f> &raster_pts,
                       const vector<float> &normal_directions )
{
  boost::shared_ptr< vector<float> > g_dir_p =  img_stats_p->getGradientDirections ( raster_pts );
  boost::shared_ptr< vector<float> > g_mag_p =  img_stats_p->getGradientMagnitudes ( raster_pts );
  
  vector<float> &g_dir = *g_dir_p;
  vector<float> &g_mag = *g_mag_p;

  if ( !g_dir.size() || !g_mag_p->size() )
    return 0;

  double score = 0;
  for ( int i = 0; i < int(g_dir.size()); i++ )
  {
    float &direction = g_dir[i], magnitude = g_mag[i];
    if ( img_stats_p->outOfImage ( direction ) )
      score += OUTER_POINT_SCORE;
    else
      score += ((magnitude > 0.01)?1.0:magnitude ) * abs ( cos ( double ( direction ) - normal_directions[i] ) );
  }

  return score/g_dir.size();
}

static void quickGuide()
{
  cout << "Use the keyboard to move the object model in a reasonable initial position:" << endl<< endl;
  
  objectPoseControlsHelp();
  
  cout << endl << "Perform the registration starting from the position indicated by the user"<<endl;
  cout << endl << "using one of the available algorithms:"<< endl<<endl;
  cout << "[1] D2CO " << endl;
  cout << "[2] DC-ICP optimization" << endl;
  cout << "[3] Simple Chamfer Matching optimization" << endl;
  cout << "[4] C-ICP optimization" << endl;
  cout << "[5] Direct optimization" << endl;
  cout << "[g] evaluate score of the model" << endl;
  cout << "[SPACE] undo optimization" << endl;
  cout << "[ESC] exit" << endl<<endl;
}

static std::string stripKnownExtension( const std::string &filename )
{
  if( filename.size() > 4 )
  {
    std::string ext = filename.substr( filename.size() - 4 );
    std::transform( ext.begin(), ext.end(), ext.begin(), ::tolower );
    if( ext == ".yml" || ext == ".xml" )
      return filename.substr( 0, filename.size() - 4 );
  }
  return filename;
}

static void ensureYmlFromXmlIfNeeded( const std::string &base_filename )
{
  std::string yml_filename = base_filename + ".yml";
  std::string xml_filename = base_filename + ".xml";

  std::ifstream yml_stream( yml_filename.c_str(), std::ios::binary );
  if( yml_stream.good() )
    return;

  std::ifstream xml_stream( xml_filename.c_str(), std::ios::binary );
  if( !xml_stream.good() )
    return;

  std::ofstream out_stream( yml_filename.c_str(), std::ios::binary );
  if( !out_stream.good() )
    return;

  out_stream << xml_stream.rdbuf();
}

static void buildRuntimePrecomputedViews( const RasterObjectModel3DPtr &obj_model_ptr )
{
  const double deg2rad = M_PI/180.0;
  std::vector<double> roll_vals = {-30, 0, 30};
  std::vector<double> pitch_vals = {-30, 0, 30};
  std::vector<double> yaw_vals = {-135, -45, 45, 135};

  std::vector<double> x_vals = {-0.08, 0.0, 0.08};
  std::vector<double> y_vals = {-0.08, 0.0, 0.08};
  std::vector<double> z_vals = {-0.20, -0.14, -0.10, 0.10};

  for( const auto &z : z_vals )
  {
    for( const auto &x : x_vals )
    {
      for( const auto &y : y_vals )
      {
        cv::Mat_<double> t_vec = (cv::Mat_<double>(3,1) << x, y, z);
        for( const auto &roll_deg : roll_vals )
        {
          for( const auto &pitch_deg : pitch_vals )
          {
            for( const auto &yaw_deg : yaw_vals )
            {
              double rx = roll_deg*deg2rad, ry = pitch_deg*deg2rad, rz = yaw_deg*deg2rad;

              double cx = cos(rx), sx = sin(rx);
              double cy = cos(ry), sy = sin(ry);
              double cz = cos(rz), sz = sin(rz);

              cv::Mat_<double> rx_mat = (cv::Mat_<double>(3,3) <<
                                         1, 0, 0,
                                         0, cx, -sx,
                                         0, sx, cx);
              cv::Mat_<double> ry_mat = (cv::Mat_<double>(3,3) <<
                                         cy, 0, sy,
                                         0, 1, 0,
                                         -sy, 0, cy);
              cv::Mat_<double> rz_mat = (cv::Mat_<double>(3,3) <<
                                         cz, -sz, 0,
                                         sz, cz, 0,
                                         0, 0, 1);

              cv::Mat_<double> rmat = rz_mat*ry_mat*rx_mat;
              cv::Mat_<double> r_vec(3,1);
              cv::Rodrigues( rmat, r_vec );

              obj_model_ptr->setModelView( r_vec, t_vec );
              obj_model_ptr->storeModelView();
            }
          }
        }
      }
    }
  }
}

int main(int argc, char **argv)
{
  string app_name( argv[0] ), model_filename, camera_filename, image_filename, edge_detector_type("CANNY");
  string pre_models_filename;
  double scale_factor = 1.0;
  int top_boundary = -1, bottom_boundary = -1, left_boundary = -1, rigth_boundary = -1;
  bool auto_init = false;
  int auto_match_step = 4;
  int auto_num_matches = 12;
  
  po::options_description desc ( "OPTIONS" );
  desc.add_options()
  ( "help,h", "Print this help messages" )
  ( "model_filename,m", po::value<string > ( &model_filename )->required(),
    "STL, PLY, OBJ, ...  model file" )
  ( "camera_filename,c", po::value<string > ( &camera_filename )->required(),
    "A YAML file that stores all the camera parameters (see the PinholeCameraModel object)" )
  ( "image_filename,i", po::value<string > ( &image_filename )->required(),
    "Image file" )
  ( "pre_models_filename,p", po::value<string > ( &pre_models_filename ),
    "Optional precomputed model views file (required for --auto_init)" )
  ( "auto_init", po::bool_switch( &auto_init ),
    "Enable automatic pose initialization using global DCM matching" )
  ( "auto_match_step", po::value<int> ( &auto_match_step )->default_value(4),
    "Image step used by exhaustive matching in --auto_init mode [4]" )
  ( "auto_num_matches", po::value<int> ( &auto_num_matches )->default_value(12),
    "Number of top global candidates considered in --auto_init mode [12]" )
  ( "edge_detector_type,e", po::value<string> ( &edge_detector_type ),
    "Edge detector types, options: [CANNY], LSD" )
  ( "scale_factor,s", po::value<double > ( &scale_factor ),
    "Optional scale factor" )
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

  if( !edge_detector_type.empty() )
  {
    std::transform( edge_detector_type.begin(), 
                    edge_detector_type.end(),
                    edge_detector_type.begin(), ::toupper);
    
    if( edge_detector_type.compare("CANNY") && 
        edge_detector_type.compare("LSD") )
    {
      cerr<<"Unrecognized edge detector type ("<<edge_detector_type<<")"<<endl; 
      return -1;
    }
  }

  if( auto_init )
  {
    if( !pre_models_filename.empty() )
    {
      pre_models_filename = stripKnownExtension( pre_models_filename );
      ensureYmlFromXmlIfNeeded( pre_models_filename );
    }
    if( auto_match_step < 1 )
      auto_match_step = 1;
    if( auto_num_matches < 1 )
      auto_num_matches = 1;
  }

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
  cout << "Loading camera parameters from file : "<<camera_filename<< endl;
  cout << "Loading background image from file : "<<image_filename<< endl;
  cout << "Edge detector type: "<<edge_detector_type<< endl;
  cout << "Scale factor : "<<scale_factor<< endl;
  if( auto_init )
  {
    cout << "Auto initialization enabled" << endl;
    if( !pre_models_filename.empty() )
      cout << "Precomputed models file : " << pre_models_filename << endl;
    else
      cout << "Precomputed models file : not provided (using runtime coarse views)" << endl;
    cout << "Auto match step : " << auto_match_step << endl;
    cout << "Auto num matches : " << auto_num_matches << endl;
  }
  if( has_roi )
    cout << "Region of interest : "<<roi<< endl;
  
  quickGuide();
  
  // Initial pose
  cv::Mat_<double> r_vec = (Mat_<double>(3,1) << 0,0,0),
                   t_vec = (Mat_<double>(3,1) << 0,0,1.0),
                   prev_r_vec, prev_t_vec;

  cv_ext::BasicTimer timer;
    
  // Load the 3D object model from file
  RasterObjectModel3DPtr obj_model_ptr( new RasterObjectModel3D() );
  obj_model_ptr->setCamModel( cam_model );
  obj_model_ptr->setStepMeters ( 0.001 ); //meters
  obj_model_ptr->setBBCenterOrigOffset();
  if ( !obj_model_ptr->setModelFile ( model_filename ) )
    return -1;
  
  obj_model_ptr->computeRaster();
  if( auto_init )
  {
    if( !pre_models_filename.empty() )
      obj_model_ptr->loadPrecomputedModelsViews( pre_models_filename );

    if( !obj_model_ptr->numPrecomputedModelsViews() )
    {
      cout << "No valid precomputed views found, generating runtime coarse views..." << endl;
      buildRuntimePrecomputedViews( obj_model_ptr );
      cout << "Generated runtime precomputed views: " << obj_model_ptr->numPrecomputedModelsViews() << endl;
    }
  }
  
  vector<cv::Point2f> proj_pts;
  vector<float> normals;
      
  cv::Mat src_img = imread ( image_filename ), scaled_img, background_img;
  cv::resize(src_img , scaled_img , cv::Size(img_w, img_h));
  
  bool exit_now = false, optimize_c = false, optimize_dc = false, optimize_hdc = false,
       optimize_direct = false, optimize_icp_c = false,
       optimize_icp_dc = false;

  if( scaled_img.channels() == 3 )
    background_img = scaled_img;
  else
    cv::cvtColor ( scaled_img,  background_img, cv::COLOR_GRAY2BGR );
  
  const int num_directions = 60;
  const double tensor_lambda = 6.0;
  
  ImageTensorPtr dist_map_tensor_ptr, /*x_dist_map_tensor_ptr, y_dist_map_tensor_ptr,*/ edgels_map_tensor_ptr;
  DistanceTransform dc;
  dc.enableParallelism(true);
  
  Mat input_img;
  if( has_roi )
    input_img = scaled_img(roi);
  else
    input_img = scaled_img;
  
  ///// Image statistics used in the evaluateScore function /////
  cv_ext::ImageStatisticsPtr img_stats_p =
    cv_ext::ImageStatistics::createImageStatistics ( input_img, true );
    
  if( !edge_detector_type.compare("LSD") )
  {
    LSDEdgeDetectorUniquePtr edge_detector_ptr( new LSDEdgeDetector());
    edge_detector_ptr->setPyrNumLevels(3);
    edge_detector_ptr->setImage(input_img);
    Mat edge_map;
    edge_detector_ptr->getEdgeMap(edge_map);
    cv_ext::showImage(edge_map, "LSD edge_map", true, 1);
    dc.setEdgeDetector(move(edge_detector_ptr));
  }
  else if( !edge_detector_type.compare("CANNY") )
  {
    CannyEdgeDetectorUniquePtr edge_detector_ptr( new CannyEdgeDetector() );
    edge_detector_ptr->setLowThreshold(40);
    edge_detector_ptr->setRatio(2);
    edge_detector_ptr->enableRGBmodality(true);    
    edge_detector_ptr->setImage(input_img);
    Mat edge_map;
    edge_detector_ptr->getEdgeMap(edge_map);
    cv_ext::showImage(edge_map, "Canny edge_map", true, 1);
    dc.setEdgeDetector(move(edge_detector_ptr));
  }
  
  timer.reset();
  cv::Mat dist_map, closest_edgels_map;
  dc.computeDistanceMap( input_img, dist_map, closest_edgels_map );
  ChamferMatching c_matching;
  c_matching.setTemplateModel( obj_model_ptr );
  c_matching.enableVerbouseMode ( false);
  c_matching.setInput( dist_map );
  
  ICPChamferMatching icp_c_matching;
  icp_c_matching.setTemplateModel( obj_model_ptr );
  icp_c_matching.enableVerbouseMode ( false);
  icp_c_matching.setInput( closest_edgels_map );
  
  dc.computeDistanceMapTensor ( input_img, dist_map_tensor_ptr, edgels_map_tensor_ptr, num_directions, tensor_lambda);
  cout<<"\ncomputeDistanceMapTensor : "<<timer.elapsedTimeMs() <<" ms"<<endl;
  
  DirectionalChamferMatching dc_matching;
  dc_matching.setNumDirections(num_directions);
  dc_matching.setTemplateModel( obj_model_ptr );
  dc_matching.enableVerbouseMode ( false);
  dc_matching.setInput( dist_map_tensor_ptr );

  if( auto_init )
  {
    vector< TemplateMatch > matches;
    dc_matching.setupExhaustiveMatching();
    timer.reset();
    dc_matching.match( auto_num_matches, matches, auto_match_step );
    if( matches.empty() )
    {
      cerr << "Automatic initialization failed: no candidate matches were found" << endl;
      return -1;
    }

    cv::Mat_<double> best_r_vec, best_t_vec;
    double best_score = -1.0;
    double best_rank = -1.0;
    int best_idx = 0;
    int num_candidates = std::min( int(matches.size()), std::max(1, auto_num_matches) );

    for( int i = 0; i < num_candidates; i++ )
    {
      cv::Mat_<double> cand_r_vec, cand_t_vec;
      obj_model_ptr->setModelView( matches[i].r_quat, matches[i].t_vec );
      obj_model_ptr->modelView( cand_r_vec, cand_t_vec );

      dc_matching.refinePosition( cand_r_vec, cand_t_vec );
      obj_model_ptr->setModelView( cand_r_vec, cand_t_vec );
      obj_model_ptr->projectRasterPoints( proj_pts, normals );
      double cand_score = evaluateScore( img_stats_p, proj_pts, normals );

      int valid_proj_pts = 0;
      for( const auto &p : proj_pts )
      {
        if( unsigned(cvRound(p.x)) < unsigned(input_img.cols) &&
            unsigned(cvRound(p.y)) < unsigned(input_img.rows) )
          valid_proj_pts++;
      }

      double visible_ratio = proj_pts.empty() ? 0.0 : double(valid_proj_pts)/double(proj_pts.size());
      double tx = cand_t_vec(0,0), ty = cand_t_vec(1,0);
      double lateral_penalty = 1.0/(1.0 + 10.0*sqrt(tx*tx + ty*ty));
      double cand_rank = cand_score*(0.5 + 0.5*visible_ratio)*lateral_penalty;

      if( cand_rank > best_rank )
      {
        best_score = cand_score;
        best_rank = cand_rank;
        best_idx = i;
        best_r_vec = cand_r_vec.clone();
        best_t_vec = cand_t_vec.clone();
      }
    }

    r_vec = best_r_vec.clone();
    t_vec = best_t_vec.clone();

    cout << "\nAutomatic initialization (DCM) elapsed time: " << timer.elapsedTimeMs() << " ms" << endl;
    cout << "Selected candidate index: " << best_idx << endl;
    cout << "Selected candidate distance: " << matches[best_idx].distance << endl;
    cout << "Selected candidate score: " << best_score << endl;
    cout << "Selected candidate rank: " << best_rank << endl;
    cout << "Initial pose (in camera reference frame): " << endl;
    cout << " - Rotation (axis-angle): " << r_vec << endl;
    cout << " - Translation (meters): " << t_vec << endl;
  }

  HybridDirectionalChamferMatching hdc_matching;
  hdc_matching.setNumDirections(num_directions);
  hdc_matching.setTemplateModel( obj_model_ptr );
  hdc_matching.enableVerbouseMode ( false);
  hdc_matching.setInput( dist_map_tensor_ptr, edgels_map_tensor_ptr );

  ICPDirectionalChamferMatching icp_dc_matching;
  icp_dc_matching.setNumDirections(num_directions);
  icp_dc_matching.setTemplateModel( obj_model_ptr );
  icp_dc_matching.enableVerbouseMode ( false);
  icp_dc_matching.setInput( edgels_map_tensor_ptr );
  
  ScaledImagesListPtr g_mag_pyr_ptr;
  computeGradientMagnitudePyr(input_img, g_mag_pyr_ptr, 1 );
  DirectMatching direct_matching;
  direct_matching.setTemplateModel( obj_model_ptr );
  direct_matching.enableVerbouseMode ( false);
  direct_matching.setInput( g_mag_pyr_ptr );
  
//   int model_idx=0;
  while ( !exit_now )
  {
//     obj_model_ptr=obj_model_ptr_vec[model_idx];
    prev_r_vec = r_vec.clone();
    prev_t_vec = t_vec.clone();

    if ( optimize_c )
    {
      timer.reset();
      c_matching.refinePosition ( r_vec, t_vec );
      cout<<endl<<"Simple Chamfer Optimization (elapsed time): "<<timer.elapsedTimeMs() <<" ms"<<endl;
      obj_model_ptr->setModelView(r_vec, t_vec);
      obj_model_ptr->projectRasterPoints ( proj_pts, normals );
      cout<<"Score : "<<evaluateScore ( img_stats_p, proj_pts, normals ) <<endl;
      cout<<"Object Pose (in camera reference frame): "<<endl;
      cout<<" - Rotation (axis-angle): "<<r_vec<<endl;
      cout<<" - Translation (meters): "<<t_vec<<endl;
      optimize_c = false;
    }
    else if ( optimize_dc )
    {
      timer.reset();
      dc_matching.refinePosition ( r_vec, t_vec );
      cout<<endl<<"D2CO (elapsed time): "<<timer.elapsedTimeMs() <<" ms"<<endl;
      obj_model_ptr->setModelView(r_vec, t_vec);
      obj_model_ptr->projectRasterPoints ( proj_pts, normals );
      cout<<"Score : "<<evaluateScore ( img_stats_p, proj_pts, normals ) <<endl;
      cout<<"Object Pose (in camera reference frame): "<<endl;
      cout<<" - Rotation (axis-angle): "<<r_vec<<endl;
      cout<<" - Translation (meters): "<<t_vec<<endl;
      optimize_dc = false;
    }
    else if ( optimize_hdc )
    {
      timer.reset();
      hdc_matching.refinePosition ( r_vec, t_vec );
      cout<<endl<<"Hybrid D2CO (elapsed time): "<<timer.elapsedTimeMs() <<" ms"<<endl;
      obj_model_ptr->setModelView(r_vec, t_vec);
      obj_model_ptr->projectRasterPoints ( proj_pts, normals );
      cout<<"Score : "<<evaluateScore ( img_stats_p, proj_pts, normals ) <<endl;
      cout<<"Object Pose (in camera reference frame): "<<endl;
      cout<<" - Rotation (axis-angle): "<<r_vec<<endl;
      cout<<" - Translation (meters): "<<t_vec<<endl;
      optimize_hdc = false;
    }
    else if( optimize_direct )
    {
      timer.reset();
      direct_matching.refinePosition ( r_vec, t_vec );
      cout<<endl<<"Direct (elapsed time): "<<timer.elapsedTimeMs() <<" ms"<<endl;
      obj_model_ptr->setModelView(r_vec, t_vec);
      obj_model_ptr->projectRasterPoints ( proj_pts, normals );
      cout<<"Score : "<<evaluateScore ( img_stats_p, proj_pts, normals ) <<endl;
      cout<<"Object Pose (in camera reference frame): "<<endl;
      cout<<" - Rotation (axis-angle): "<<r_vec<<endl;
      cout<<" - Translation (meters): "<<t_vec<<endl;
      optimize_direct = false;      
    }
    else if( optimize_icp_c )
    {
      timer.reset();
      icp_c_matching.refinePosition ( r_vec, t_vec );
      cout<<endl<<"C-ICP (elapsed time): "<<timer.elapsedTimeMs() <<" ms"<<endl;
      obj_model_ptr->setModelView(r_vec, t_vec);
      obj_model_ptr->projectRasterPoints ( proj_pts, normals );
      cout<<"Score : "<<evaluateScore ( img_stats_p, proj_pts, normals ) <<endl;
      cout<<"Object Pose (in camera reference frame): "<<endl;
      cout<<" - Rotation (axis-angle): "<<r_vec<<endl;
      cout<<" - Translation (meters): "<<t_vec<<endl;
      optimize_icp_c = false;
    }
    else if( optimize_icp_dc )
    {
      timer.reset();
      icp_dc_matching.refinePosition ( r_vec, t_vec );
      cout<<endl<<"DC-ICP (elapsed time): "<<timer.elapsedTimeMs() <<" ms"<<endl;
      obj_model_ptr->setModelView(r_vec, t_vec);
      obj_model_ptr->projectRasterPoints ( proj_pts, normals );
      cout<<"Score : "<<evaluateScore ( img_stats_p, proj_pts, normals ) <<endl;
      cout<<"Object Pose (in camera reference frame): "<<endl;
      cout<<" - Rotation (axis-angle): "<<r_vec<<endl;
      cout<<" - Translation (meters): "<<t_vec<<endl;
      optimize_icp_dc = false;     
    }  

    Mat background, draw_img;

    background = background_img.clone();
    obj_model_ptr->setModelView(r_vec, t_vec);
    obj_model_ptr->projectRasterPoints ( proj_pts, normals );
    if( has_roi )
    {
      cv::Point dbg_tl = roi.tl(), dbg_br = roi.br();
      dbg_tl.x -= 1; dbg_tl.y -= 1;
      dbg_br.x += 1; dbg_br.y += 1;
      cv::rectangle( background, dbg_tl, dbg_br, cv::Scalar(255,255,255));      
      draw_img = background(roi);
    }
    else
      draw_img = background;
    cv_ext::drawPoints ( draw_img, proj_pts,cv::Scalar ( 0,0,255 ) );

    char key = 27;
    cv::imshow ( "test_localization", background );
    key = cv::waitKey();

    parseObjectPoseControls( key, r_vec, t_vec );
        
    switch ( key )
    {

    case 'g' : //evaluate Score
    case 'G' :
      cout<<"Score : "<<evaluateScore ( img_stats_p, proj_pts, normals ) <<endl;
      break;
    case '1':
//       cout<<"D2CO"<<endl;
      optimize_dc = true;
      break;
    case '2':
//       cout<<"Directional ICP Chamfer"<<endl;
      optimize_icp_dc = true;
      break;
    case '3':
//       cout<<"Direct Chamfer"<<endl;
      optimize_c = true;
      break;      
    case '4':
//       cout<<"ICP Chamfer"<<endl;
      optimize_icp_c = true;
      break;
    case '5':
//       cout<<"Gradient magnitude"<<endl;
      optimize_direct = true;
      break;      
    case '6':
      optimize_hdc = true;
      break;
    case ' ':
      cout<<"Undo"<<endl;
      r_vec = prev_r_vec.clone();
      t_vec = prev_t_vec.clone();
      break;
    case 27: // ESC
      exit_now=true;
      break;
    }

  }

  return 0;
}
