#include <cstdio>
#include <string>
#include <sstream>
#include <algorithm>
#include <map>
#include <boost/program_options.hpp>
#include "boost/filesystem.hpp"
#include <opencv2/rgbd.hpp>
#include "cv_ext/cv_ext.h"
#include "raster_object_model3D.h"
#include "raster_object_model2D.h"
#include "chamfer_matching.h"

#include "apps_utils.h"

namespace po = boost::program_options;
namespace filesystem = boost::filesystem;
using namespace std;
using namespace cv;
using namespace cv_ext;

struct AppOptions
{
  string models_filename, camera_filename, input_image_dir, algorithm_type, output_type;
  double scale_factor, match_threshold;
  bool has_roi = false;
  cv::Rect roi;
  int top_boundary, bottom_boundary, left_boundary, rigth_boundary;
};

void parseCommandLine( int argc, char **argv, AppOptions &options )
{
  string app_name( argv[0] );

  options.scale_factor = 1.0;
  options.match_threshold = 2.5;
  options.algorithm_type = "DCM";
  options.output_type = "VIDEO";
  options.top_boundary = options.bottom_boundary = 
    options.left_boundary = options.rigth_boundary = -1;
  
  po::options_description desc ( "OPTIONS" );
  desc.add_options()
  ( "help,h", "Print this help messages" )
  ( "models_filename,m", po::value<string > ( &options.models_filename )->required(),
    "Precomputed models file" )
  ( "camera_filename,c", po::value<string > ( &options.camera_filename )->required(),
    "A YAML file that stores all the camera parameters (see the PinholeCameraModel object)" )
  ( "input_image_dir,d", po::value<string > ( &options.input_image_dir)->required(),
    "input_image_dir" )
  ( "scale_factor,s", po::value<double> ( &options.scale_factor ),
    "Scale factor [1]" )
  ( "match_threshold,t", po::value<double > ( &options.match_threshold ),
    "Match threshold [2.5]" )  
  ( "algorithm_type,a", po::value<string> ( &options.algorithm_type ),
    "Used algorithm, options: [DCM], OCM" )
  ( "output_type,o", po::value<string> ( &options.output_type ),
    "Output type, options: [VIDEO], FILE, STREAMING" )
  ( "tb", po::value<int> ( &options.top_boundary ),
    "Optional region of interest: top boundary " )
  ( "bb", po::value<int> ( &options.bottom_boundary ),
    "Optional region of interest: bottom boundary " )
  ( "lb", po::value<int> ( &options.left_boundary ),
  "Optional region of interest: left boundary " )  
  ( "rb", po::value<int> ( &options.rigth_boundary ),
  "Optional region of interest: rigth boundary" );

  po::variables_map vm;

  try
  {
    po::store ( po::parse_command_line ( argc, argv, desc ), vm );

    if ( vm.count ( "help" ) )
    {
      cout << "USAGE: "<<app_name<<" OPTIONS"
                << endl << endl<<desc;                
      exit(EXIT_SUCCESS);
    }

    po::notify ( vm );
  }

  catch ( boost::program_options::required_option& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    cout << "USAGE: "<<app_name<<" OPTIONS"
              << endl << endl<<desc;
    exit(EXIT_FAILURE);
  }

  catch ( boost::program_options::error& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    cout << "USAGE: "<<app_name<<" OPTIONS"
              << endl << endl<<desc;
    exit(EXIT_FAILURE);
  }
  
  std::transform(options.algorithm_type.begin(), 
                 options.algorithm_type.end(),
                 options.algorithm_type.begin(), ::toupper);

  std::transform(options.output_type.begin(), 
                 options.output_type.end(),
                 options.output_type.begin(), ::toupper);
  
  if( options.algorithm_type.compare("DCM") && 
      options.algorithm_type.compare("OCM") )
  {
    cerr<<"Unrecognized algorithm type ("<<options.algorithm_type<<")"<<endl; 
    exit(EXIT_FAILURE);
  }
  
  if( options.output_type.compare("VIDEO") && 
      options.output_type.compare("FILE") && 
      options.output_type.compare("STREAMING") )
  {
    cerr<<"Unrecognized output type ("<<options.output_type<<")"<<endl; 
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char **argv)
{
  AppOptions options;
  parseCommandLine( argc, argv, options );
  
  Mat r_vec, t_vec;

  cv_ext::PinholeCameraModel cam_model;
  cam_model.readFromFile(options.camera_filename);
  cam_model.setSizeScaleFactor(options.scale_factor);
  int img_w = cam_model.imgWidth(), img_h = cam_model.imgHeight();

  bool has_roi = false;
  cv::Rect roi;
  
  if( options.top_boundary != -1 || options.bottom_boundary != -1 || 
      options.left_boundary != -1 || options.rigth_boundary != -1 )
  {
    Point tl(0,0), br(img_w, img_h);
    
    if( options.top_boundary != -1 ) tl.y = options.top_boundary;
    if( options.left_boundary != -1 ) tl.x = options.left_boundary;
    if( options.bottom_boundary != -1 ) br.y = options.bottom_boundary;
    if( options.rigth_boundary != -1 ) br.x = options.rigth_boundary;
    
    has_roi = true;
    roi = cv::Rect(tl, br);
    cam_model.setRegionOfInterest(roi);
    cam_model.enableRegionOfInterest(true);
    roi = cam_model.regionOfInterest();
  }  
  
  cout << "Loading precomputed models from file : "<<options.models_filename<< endl;
  cout << "Loading camera parameters from file : "<<options.camera_filename<< endl;
  cout << "Loading input images from directory : "<<options.input_image_dir<< endl;
  cout << "Scale factor : "<<options.scale_factor<< endl;
  cout << "Algorithm type : "<<options.algorithm_type<< endl;
  cout << "Output type : "<<options.output_type<< endl;
  cout << "Match threshold : "<<options.match_threshold<< endl;
  if( has_roi )
    cout << "Region of interest : "<<roi<< endl;  
  
  RasterObjectModel3DPtr obj_model_ptr( new RasterObjectModel3D() );
  RasterObjectModel3D &obj_model = *obj_model_ptr;

  obj_model.setCamModel( cam_model );
  obj_model.loadPrecomputedModelsViews(options.models_filename);
  
  DistanceTransform dc;
  dc.setDistThreshold(30);
  dc.enableParallelism(true);
  
  CannyEdgeDetectorUniquePtr edge_detector_ptr( new CannyEdgeDetector() );
  edge_detector_ptr->setLowThreshold(40);
  edge_detector_ptr->setRatio(2);
  edge_detector_ptr->enableRGBmodality(true);

  dc.setEdgeDetector(std::move(edge_detector_ptr));
  
  const int num_directions = 60, increment = 4;
  const double tensor_lambda = 6.0;
  const bool smooth_tensor = false;
  
  DirectionalChamferMatching dcm;
  OrientedChamferMatching ocm;
  
  if( !options.algorithm_type.compare("DCM") )
  {
    dcm.setTemplateModel(obj_model_ptr);
    dcm.setNumDirections(num_directions);
    dcm.enableParallelism(true);
    dcm.setupExhaustiveMatching();    
  }
  else if( !options.algorithm_type.compare("OCM") )
  {
    ocm.setNumDirections(num_directions);
    ocm.setTemplateModel(obj_model_ptr);
    ocm.enableParallelism(true);
    ocm.setupExhaustiveMatching();    
  }  

  
  filesystem::path images_path (options.input_image_dir);
  filesystem::directory_iterator end_itr;
  vector< TemplateMatch > matches;
  for (filesystem::directory_iterator itr(images_path); itr != end_itr; ++itr)
  {
    if ( !filesystem::is_regular_file(itr->path()))
      continue;
    
    string current_file = itr->path().string();
 
    Mat src_img = imread ( current_file, cv::IMREAD_COLOR ), input_img;
    
    if( src_img.empty())
      continue;
      
    cv::resize(src_img,src_img,cv::Size(img_w, img_h));
    
    if( has_roi )
      input_img = src_img(roi);
    else
      input_img = src_img;
    
    if( !options.algorithm_type.compare("DCM") )
    {
      cv_ext::BasicTimer timer;
      ImageTensorPtr dst_map_tensor_ptr;
      dc.computeDistanceMapTensor ( input_img, dst_map_tensor_ptr, num_directions, tensor_lambda, smooth_tensor);
      dcm.setInput( dst_map_tensor_ptr );
      dcm.match(3, matches, (int)increment);
      cout<<"DCM elapsed time ms : "<<timer.elapsedTimeMs()<<endl;
    }
    else if( !options.algorithm_type.compare("OCM") )
    {
      cv_ext::BasicTimer timer;
      Mat dist_map, dir_map;
      dc.computeDistDirMap(input_img, dist_map, dir_map, num_directions);
      ocm.setInput( dist_map, dir_map );
      ocm.match(3, matches, (int)increment);
      cout<<"OCM elapsed time ms : "<<timer.elapsedTimeMs()<<endl;
    }


    int i_m = 0;
    for( auto iter = matches.begin(); iter != matches.end(); iter++, i_m++ )
    {
      TemplateMatch &match = *iter;
//       cout<<match.img_offset<<" "<<match.distance<<endl;
    
      vector<Point2f> proj_pts;
      const vector<Point3f> &pts = obj_model.getPrecomputedPoints(match.id);
      cv_ext::PinholeSceneProjector proj( obj_model.cameraModel() );
      proj.setTransformation( match.r_quat, match.t_vec );
      proj.projectPoints( pts, proj_pts ); 
      
      cv::Mat display = src_img.clone(), draw_img;
      if( has_roi )
      {
        cv::Point dbg_tl = roi.tl(), dbg_br = roi.br();
        dbg_tl.x -= 1; dbg_tl.y -= 1;
        dbg_br.x += 1; dbg_br.y += 1;
        cv::rectangle( display, dbg_tl, dbg_br, cv::Scalar(255,255,255));      
        draw_img = display(roi);
      }
      else
        draw_img = display;
      
      cv_ext::drawPoints( draw_img, proj_pts, match.distance > options.match_threshold ? Scalar(0,0,255):Scalar(0,255,0) );
      
      if( !options.output_type.compare("VIDEO") )
      {
        cv_ext::showImage(display,"display", true, 100);
      }
      else if( !options.output_type.compare("FILE") )
      {
        size_t found = current_file.find_last_of("/");
        string path = current_file.substr(0,found);
        string filename = current_file.substr(found+1);
        found = filename.find_last_of(".");
        string basename = filename.substr(0,found);
        string extension = filename.substr(found+1);
        
        stringstream sstr;
        sstr<<path;
        sstr<<"/results/";
        sstr<<basename;
        sstr<<"_";
        sstr<<i_m;
        sstr<<".";
        sstr<<extension;

        cout<<"Saving result image to: "<<sstr.str()<<endl;        
        imwrite(sstr.str(), display);
      }
      else if( !options.output_type.compare("STREAMING") )
      {
        cout<<"Warning: STREAMING output type not yet implemented"<<endl;
      }
    }
  }

  return 0;
}
