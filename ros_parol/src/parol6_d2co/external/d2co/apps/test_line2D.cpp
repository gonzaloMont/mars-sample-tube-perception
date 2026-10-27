#include <cstdio>
#include <string>
#include <sstream>
#include <map>
#include <boost/program_options.hpp>
#include <opencv2/rgbd.hpp>
// #include <opencv2/cnn_3dobj.hpp>
#include "omp.h"

#include "cv_ext/cv_ext.h"

#include "raster_object_model3D.h"
#include "raster_object_model2D.h"
#include "chamfer_matching.h"

#include "apps_utils.h"

extern "C"
{
#include "lsd.h"
}

#define TEST_CM 0
#define TEST_OCM 0
#define TEST_DCM 1
#define TEST_LINE2D 0
#define TEST_LINE2D_GL 0

#define PARALLELISM_ENABLED 1

#define MAX_TEMPLATE_PTS 64
#define RGB_MODALITY_ENABLED 1

namespace po = boost::program_options;
using namespace std;
using namespace cv;
using namespace cv_ext;

struct Pose
{
  Pose(Mat r, Mat t, Rect b)
  {
    r_vec = r.clone();
    t_vec = t.clone();
    bb = b;
  };
  Pose(){};
  Mat r_vec, t_vec;
  Rect bb;
};


void normalizePose( const RasterObjectModel3D &obj_model, const PinholeCameraModel &cam_model, const Point &disp, Pose &p, int id )
{
  Point2f off_p(disp.x - p.bb.x, disp.y - p.bb.y);

  if( cv_ext::norm2D(off_p) == 0 )
    return;

  vector <Point3f> obj_pts = obj_model.getPrecomputedPoints(id);
  vector <Point2f> proj_pts;
  
  obj_model.projectRasterPoints( id, proj_pts );
  for( auto &p : proj_pts )
    p += off_p;
  
  cv::Mat r_vec(3,1,cv::DataType<double>::type);
  cv::Mat t_vec(3,1,cv::DataType<double>::type);
  
  cv::solvePnP( obj_pts, proj_pts, cam_model.cameraMatrix(), cam_model.distorsionCoeff(), r_vec, t_vec );
  
  p.r_vec = r_vec;
  p.t_vec = t_vec;
}


void drawResponse(const std::vector<cv::linemod::Template>& templates,
                  int num_modalities, cv::Mat& dst, cv::Point offset, int T)
{
  static const cv::Scalar COLORS[5] = { CV_RGB(0, 0, 255),
                                        CV_RGB(0, 255, 0),
                                        CV_RGB(255, 255, 0),
                                        CV_RGB(255, 140, 0),
                                        CV_RGB(255, 0, 0) };

  for (int m = 0; m < num_modalities; ++m)
  {
    cv::Scalar color = COLORS[m];

    for (int i = 0; i < (int)templates[m].features.size(); ++i)
    {
      cv::linemod::Feature f = templates[m].features[i];
      cv::Point pt(f.x + offset.x, f.y + offset.y);
      cv::circle(dst, pt, T / 2, color);
    }
  }
}


void loadACCVDataset( const string input_image_dir, vector<string> &image_names,
                       vector<Mat> &gt_r_vecs, vector<Mat> &gt_t_vecs )
{
  Mat gt_r_vec = Mat_<double>(3,1), gt_t_vec = Mat_<double>(3,1);
  int image_idx = 0;
  while( true )
  {
//     if(!(image_idx % 100))
//       cout<<"Loading data #"<<image_idx<<endl;

    std::string rot_name = cv::format("%s/rot%d.rot", input_image_dir.c_str(), image_idx),
                tra_name = cv::format("%s/tra%d.tra", input_image_dir.c_str(), image_idx);
    FILE *rot_file = fopen(rot_name.c_str(),"r"),
         *tra_file = fopen(tra_name.c_str(),"r");
    if( rot_file == NULL || tra_file == NULL )
      break;

    Eigen::Matrix3d gt_rot;
    Eigen::Vector3d gt_tra;
    double val0,val1;
    CV_Assert(fscanf(rot_file,"%lf %lf", &val0,&val1) == 2);
    CV_Assert(fscanf(rot_file,"%lf %lf %lf", &gt_rot(0,0),&gt_rot(0,1),&gt_rot(0,2)) == 3);
    CV_Assert(fscanf(rot_file,"%lf %lf %lf", &gt_rot(1,0),&gt_rot(1,1),&gt_rot(1,2)) == 3);
    CV_Assert(fscanf(rot_file,"%lf %lf %lf", &gt_rot(2,0),&gt_rot(2,1),&gt_rot(2,2)) == 3);

    CV_Assert(fscanf(tra_file,"%lf %lf", &val0, &val1) == 2);
    CV_Assert(fscanf(tra_file,"%lf %lf %lf", &gt_tra(0),&gt_tra(1),&gt_tra(2)) == 3);

    cv_ext::rotMat2Exp( gt_rot, gt_r_vec );

    gt_t_vec.at<double>(0) = gt_tra(0)/100.0;
    gt_t_vec.at<double>(1) = gt_tra(1)/100.0;
    gt_t_vec.at<double>(2) = gt_tra(2)/100.0;

    gt_r_vecs.push_back(gt_r_vec.clone());
    gt_t_vecs.push_back(gt_t_vec.clone());

    fclose(rot_file);
    fclose(tra_file);

    std::string image_name = cv::format("%s/color%d.jpg", input_image_dir.c_str(), image_idx);
    image_names.push_back(image_name);
    image_idx++;
  }
}

struct AppOptions
{
  string models_filename, camera_filename, input_image_dir;
};

void parseCommandLine( int argc, char **argv, AppOptions &options )
{
  string app_name( argv[0] );

  po::options_description desc ( "OPTIONS" );
  desc.add_options()
  ( "help,h", "Print this help messages" )
  ( "model_filename,m", po::value<string > ( &options.models_filename )->required(),
    "DXF, STL or PLY model file" )
  ( "camera_filename,c", po::value<string > ( &options.camera_filename )->required(),
    "A YAML file that stores all the camera parameters (see the PinholeCameraModel object)" )
  ( "input_image_dir,d", po::value<string > ( &options.input_image_dir)->required(),
    "input_image_dir" );

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
    exit(EXIT_FAILURE);
  }

  catch ( boost::program_options::error& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    exit(EXIT_FAILURE);
  }

  cout << "Loading model from file : "<<options.models_filename<< endl;
  cout << "Loading camera parameters from file : "<<options.camera_filename<< endl;
  cout << "Loading input images from directory : "<<options.input_image_dir<< endl;
}

void logResults(string name, vector<double> fbm, uint64_t total_time, int num_images )
{
  double results_acc = 0.0;
  cout<<name<<" =[";
  for(int i = 0; i < fbm.size(); i++ )
  {
    fbm[i] /= double(num_images);
    results_acc += fbm[i];
    cout<<results_acc<<" ";
  }
  cout<<"]"<<" avg runtime : "<<total_time /num_images<<endl;
}

int main(int argc, char **argv)
{
  AppOptions options;
  parseCommandLine( argc, argv, options );
  
  const float dist_transform_thresh = 30;
  const int canny_low_thresh = 80, canny_ratio = 2;
  const int max_template_pts = MAX_TEMPLATE_PTS;
  const int search_step = 4;
  
  cv_ext::PinholeCameraModel cam_model;
  cam_model.readFromFile(options.camera_filename);

  int width = cam_model.imgWidth(), height = cam_model.imgHeight();

  RasterObjectModel3DPtr obj_model_ptr( new RasterObjectModel3D() );
  RasterObjectModel3D &obj_model = *obj_model_ptr;

  obj_model.setCamModel( cam_model );
  obj_model.setStepMeters(0.005);
  obj_model.setUnitOfMeasure(RasterObjectModel::MILLIMETER);
  obj_model.requestVertexColors();

  if(!obj_model.setModelFile( options.models_filename ))
    return -1;

  obj_model.computeRaster();


  vector<string> image_names;
  vector<Mat> gt_r_vecs, gt_t_vecs;
  vector<Pose> gt_poses;
  loadACCVDataset( options.input_image_dir, image_names, gt_r_vecs, gt_t_vecs );

#if 1
  for( int i = 0; i < int(image_names.size()); i++ )
  {
    obj_model.setModelView(gt_r_vecs[i], gt_t_vecs[i]);
    obj_model.storeModelView();
    vector<Point2f> raster_pts;
    vector<float> raster_normals_dirs;

    obj_model.projectRasterPoints( raster_pts, raster_normals_dirs );
    Point tl(width,height), br(0,0);

    for( int j = 0; j < int(raster_pts.size()); j++ )
    {
      int x = cvRound(raster_pts[j].x), y = cvRound(raster_pts[j].y);

      if( unsigned(x) < unsigned(width) && unsigned(y) < unsigned(height) )
      {
        if( x < tl.x ) tl.x = x;
        if( y < tl.y ) tl.y = y;
        if( x > br.x ) br.x = x;
        if( y > br.y ) br.y = y;
      }
    }
    br.x++;
    br.y++;
    cv::Rect bb(tl,br);
    gt_poses.push_back(Pose(gt_r_vecs[i],gt_t_vecs[i], bb));
  }
  
//   obj_model.savePrecomputedModelsViews("benchviseblue_models");

#else
// WARNING Debug code!
  obj_model.loadPrecomputedModelsViews("benchviseblue_models");
  for( int i = 0; i < obj_model.numPrecomputedModelsViews(); i++ )
  {
    obj_model.setModelView(i);  
    vector<Point2f> raster_pts;
    vector<float> raster_normals_dirs;

    obj_model.projectRasterPoints( raster_pts, raster_normals_dirs );
    Point tl(width,height), br(0,0);

    for( int j = 0; j < raster_pts.size(); j++ )
    {
      int x = cvRound(raster_pts[j].x), y = cvRound(raster_pts[j].y);

      if( unsigned(x) < unsigned(width) && unsigned(y) < unsigned(height) )
      {
        if( x < tl.x ) tl.x = x;
        if( y < tl.y ) tl.y = y;
        if( x > br.x ) br.x = x;
        if( y > br.y ) br.y = y;
      }
    }
    br.x++;
    br.y++;
    cv::Rect bb(tl,br);
    gt_poses.push_back(Pose(gt_r_vecs[i],gt_t_vecs[i], bb));
  }
  
#endif

// // Test draw ground truths
//   for( int i = 0; i < image_names.size(); i++ )
//   {
//     Mat src_img = imread ( image_names[i].c_str(), cv::IMREAD_COLOR );
//
//     vector<Point2f> raster_pts;
//     obj_model.projectRasterPoints( i, raster_pts );
//     cv_ext::drawPoints(src_img, raster_pts, Scalar(0,0,255));
//     cv::rectangle(src_img, gt_poses[i].bb,CV_RGB(0, 0, 255));
//
//     cv_ext::showImage(src_img);
//   }
  
  
#if TEST_LINE2D || TEST_LINE2D_GL
  
  const int n_threads = ( PARALLELISM_ENABLED ? omp_get_max_threads() : 1 );  
    
//   cout<<"Linemod models"<<endl;
    const int linemod_matching_threshold = 60;
//  const int T_DEFAULTS[] = {5,8};
  const int T_DEFAULTS[] = {search_step};
  std::vector< Ptr<linemod::Modality> > modalities;

  modalities.push_back(new linemod::ColorGradient(10.0f,63, 55.0f));
  std::vector<int> T_pyramid(T_DEFAULTS, T_DEFAULTS + 1);

  vector<Ptr<linemod::Detector> > lm_detector, lm_detector_gl;
  
  for( int i = 0; i < n_threads; i++ )
  {
    lm_detector.push_back( (new  linemod::Detector(modalities, T_pyramid)) );
    lm_detector_gl.push_back( (new  linemod::Detector(modalities, T_pyramid)) );
  }

  map<string, Pose> pose_map, pose_map_gl;
  map<string, int> lm_detector_map, template_map;
  int num_classes = 0, th_id = 0;

  for( int i = 0; i < obj_model.numPrecomputedModelsViews(); i++ )
  {
    std::string class_id = cv::format("rot_%d", num_classes);
    cv::Rect bb, bb_gl;
    vector<Mat> templates, templates_gl;
    obj_model.setModelView(i);
    Mat gray_template, src_template = obj_model.getRenderedModel(), src_template_gl;
    cvtColor( src_template, gray_template, cv::COLOR_BGR2GRAY );
    cvtColor( gray_template, src_template_gl, cv::COLOR_GRAY2BGR );
    templates.push_back(src_template);
    templates_gl.push_back(src_template_gl);

    if ( lm_detector[th_id]->addTemplate(templates, class_id, cv::Mat(), &bb) != -1 &&
         lm_detector_gl[th_id]->addTemplate(templates_gl, class_id, cv::Mat(), &bb_gl) != -1 )
    {
      lm_detector_map[class_id] = th_id;
      template_map[class_id] = i;
//       if(!(num_classes % 100))
//         cout<<"Added template #"<<num_classes<<endl;
      Mat_<double> r_vec, t_vec;
      obj_model.modelView(r_vec, t_vec);
      pose_map[class_id] = Pose(r_vec,t_vec, bb);
      pose_map_gl[class_id] = Pose(r_vec,t_vec, bb_gl);
      num_classes++;

//       Mat src_img = imread ( image_names[i].c_str(), cv::IMREAD_COLOR );
// 
//       vector<Point2f> raster_pts;
// //       obj_model.projectRasterPoints( i, raster_pts );
// //       cv_ext::drawPoints(src_img, raster_pts, Scalar(0,0,255));
//       const std::vector<cv::linemod::Template>& templates_vec = detector[th_id]->getTemplates(class_id, 0);
//       drawResponse(templates_vec,1,src_img,Point(bb.x, bb.y),detector[th_id]->getT(0));
//       cv::rectangle(src_img, bb,CV_RGB(0, 0, 255));
//       cv::rectangle(src_img, bb_gl,CV_RGB(0, 255, 0));
// 
//       cv_ext::showImage(src_img);
      ++th_id;
      th_id %= n_threads;
    }
    else
      cout<<"Failed to add gt template "<<i<<endl;
  }
#endif

  const double max_rot_diff_deg = 5, max_t_diff = 0.05;
  const int num_test_images = image_names.size();
  const int max_num_matches = 10;
  
  cout<<num_test_images<<endl;

  vector<double> cm_fbm(max_num_matches,0.0), ocm_fbm(max_num_matches,0.0), 
                 dcm_fbm(max_num_matches,0.0), line_fbm(max_num_matches,0.0),
                 line_fbm_gl(max_num_matches,0.0);
                 
  uint64_t avg_timer = 0;
  
#if TEST_CM
  DistanceTransform dc;
  dc.setDistThreshold(dist_transform_thresh);

  CannyEdgeDetectorUniquePtr edge_detector_ptr( new CannyEdgeDetector() );
//   Best so far...
  edge_detector_ptr->setLowThreshold(canny_low_thresh);
  edge_detector_ptr->setRatio(canny_ratio);
  edge_detector_ptr->enableRGBmodality(RGB_MODALITY_ENABLED);

  dc.enableParallelism(true);
  dc.setEdgeDetector(std::move(edge_detector_ptr));
  
  ChamferMatching cm;
  cm.setTemplateModel(obj_model_ptr);
  cm.enableParallelism(true);
  cm.setupExhaustiveMatching( max_template_pts );
  
  avg_timer = 0;
  
  for( int i = 0; i < num_test_images; i++ )
  {
//     if(!(i%100))
//       cout<<"Image "<<i<<" of "<<num_test_images<<endl;
    Mat src_img = imread ( image_names[i].c_str(),cv::IMREAD_COLOR );
    cv_ext::BasicTimer timer;
    Mat dist_map;
    dc.computeDistanceMap(src_img, dist_map);
    cm.setInput( dist_map );
    vector< TemplateMatch > matches;
    cm.match(max_num_matches, matches, search_step);
    avg_timer += timer.elapsedTimeMs();

    int i_m = 0;
    for( auto iter = matches.begin(); iter != matches.end(); iter++, i_m++ )
    {
      TemplateMatch &match = *iter;

      Eigen::Matrix3d est_rot_mat = match.r_quat.toRotationMatrix(), rot_mat_gt;
      cv_ext::exp2RotMat(gt_poses[i].r_vec, rot_mat_gt);

      cv::Point3d p_t_diff( match.t_vec(0) - gt_poses[i].t_vec.at<double>(0),
                            match.t_vec(1) - gt_poses[i].t_vec.at<double>(1),
                            match.t_vec(2) - gt_poses[i].t_vec.at<double>(2) );
      
      double rot_diff = 180.0*cv_ext::rotationDist(est_rot_mat, rot_mat_gt)/M_PI, 
             t_diff  = cv_ext::norm3D(p_t_diff);
             
      if( rot_diff < max_rot_diff_deg && t_diff < max_t_diff )
      {
        cm_fbm[i_m]++;
        
//         obj_model.setModelView(match.r_quat, match.t_vec);
//         vector<Point2f> proj_pts;
//         obj_model.projectRasterPoints(proj_pts);        
//         cv::Mat display = src_img.clone();
//         cv_ext::drawPoints( display, proj_pts, Scalar(0,255,0) );
//         cv_ext::showImage(display,"display");
        
        break;
      }
    }
  }
  stringstream sstr;
  sstr<<"cm_fbm";
  sstr<<max_template_pts;
  if( !RGB_MODALITY_ENABLED )
    sstr<<" gl";
  logResults(sstr.str(), cm_fbm, avg_timer, num_test_images );
  
#endif
  
#if TEST_OCM
  DistanceTransform dc;
  dc.setDistThreshold(dist_transform_thresh);

  CannyEdgeDetectorUniquePtr edge_detector_ptr( new CannyEdgeDetector() );
//   Best so far...
  edge_detector_ptr->setLowThreshold(canny_low_thresh);
  edge_detector_ptr->setRatio(canny_ratio);
  edge_detector_ptr->enableRGBmodality(RGB_MODALITY_ENABLED);

//   edge_detector_ptr->setLowThreshold(80);
//   edge_detector_ptr->setRatio(3);

//   LSDEdgeDetectorUniquePtr edge_detector_ptr( new LSDEdgeDetector() );
//   edge_detector_ptr->setPyrNumLevels(2);
//   edge_detector_ptr->setScale(4);

  dc.enableParallelism(true);
  dc.setEdgeDetector(std::move(edge_detector_ptr));

  const int num_directions = 8;
  
  OrientedChamferMatching ocm;
  ocm.setNumDirections(num_directions);
  ocm.setTemplateModel(obj_model_ptr);
  ocm.enableParallelism(true);
  ocm.setupExhaustiveMatching( max_template_pts );
  
    
  const double tensor_lambda = 6.0;
  const bool smooth_tensor = false;
 
  avg_timer = 0;
  
  for( int i = 0; i < num_test_images; i++ )
  {
    if(!(i%100))
      cout<<"Image "<<i<<" of "<<num_test_images<<endl;
    Mat src_img = imread ( image_names[i].c_str(),cv::IMREAD_COLOR );
    cv_ext::BasicTimer timer;
    Mat dist_map, dir_map;
    dc.computeDistDirMap(src_img, dist_map, dir_map, num_directions);
//     cout<<"CM elapsed time computeDistanceMapTensor ms : "<<timer.elapsedTimeMs()<<endl;
    ocm.setInput( dist_map, dir_map );
    vector< TemplateMatch > matches;
    ocm.match(max_num_matches, matches, search_step);
    avg_timer += timer.elapsedTimeMs();

    int i_m = 0;
    for( auto iter = matches.begin(); iter != matches.end(); iter++, i_m++ )
    {
      TemplateMatch &match = *iter;

      Eigen::Matrix3d est_rot_mat = match.r_quat.toRotationMatrix(), rot_mat_gt;
      cv_ext::exp2RotMat(gt_poses[i].r_vec, rot_mat_gt);

      cv::Point3d p_t_diff( match.t_vec(0) - gt_poses[i].t_vec.at<double>(0),
                            match.t_vec(1) - gt_poses[i].t_vec.at<double>(1),
                            match.t_vec(2) - gt_poses[i].t_vec.at<double>(2) );
      
      double rot_diff = 180.0*cv_ext::rotationDist(est_rot_mat, rot_mat_gt)/M_PI, 
             t_diff  = cv_ext::norm3D(p_t_diff);
             
      if( rot_diff < max_rot_diff_deg && t_diff < max_t_diff )
      {
        ocm_fbm[i_m]++;
//         obj_model.setModelView(match.r_quat, match.t_vec);
//         vector<Point2f> proj_pts;
//         obj_model.projectRasterPoints(proj_pts);        
//         cv::Mat display = src_img.clone();
//         cv_ext::drawPoints( display, proj_pts, Scalar(0,255,0) );
//         cv_ext::showImage(display,"display");        
        
        break;
      }
    }
  }
  stringstream sstr;
  sstr<<"ocm_fbm";
  sstr<<max_template_pts;
  if( !RGB_MODALITY_ENABLED )
    sstr<<" gl";  
  logResults(sstr.str(), ocm_fbm, avg_timer, num_test_images );

#endif
  
  
#if TEST_DCM
  DistanceTransform dc;
  dc.setDistThreshold(dist_transform_thresh);

  CannyEdgeDetectorUniquePtr edge_detector_ptr( new CannyEdgeDetector() );
//   Best so far...
  edge_detector_ptr->setLowThreshold(canny_low_thresh);
  edge_detector_ptr->setRatio(canny_ratio);
  edge_detector_ptr->enableRGBmodality(RGB_MODALITY_ENABLED);

  dc.enableParallelism(true);
//   dc.setDistType(CV_DIST_L1);
//   dc.setMaskSize(3);
  dc.setEdgeDetector(std::move(edge_detector_ptr));
  
  int num_directions = 60;
  double tensor_lambda = 6.0;
  const bool smooth_tensor = false;
  
  DirectionalChamferMatching dcm;
  dcm.setTemplateModel(obj_model_ptr);
  dcm.setNumDirections(num_directions);
  dcm.enableParallelism(true);
  dcm.setupExhaustiveMatching( max_template_pts );
  
  avg_timer = 0;
  
  for( int i = 0; i < num_test_images; i++ )
  {
//     if(!(i%100))
//       cout<<"Image "<<i<<" of "<<num_test_images<<endl;
    Mat src_img = imread ( image_names[i].c_str(),cv::IMREAD_COLOR );
    cv_ext::BasicTimer timer;
    ImageTensorPtr dst_map_tensor_ptr;
    dc.computeDistanceMapTensor ( src_img, dst_map_tensor_ptr, num_directions, tensor_lambda, smooth_tensor);
    dcm.setInput( dst_map_tensor_ptr );
    vector< TemplateMatch > matches;
    dcm.match(max_num_matches, matches, search_step);
    avg_timer += timer.elapsedTimeMs();

    int i_m = 0;
    for( auto iter = matches.begin(); iter != matches.end(); iter++, i_m++ )
    {
      TemplateMatch &match = *iter;
      
      Eigen::Matrix3d est_rot_mat = match.r_quat.toRotationMatrix(), rot_mat_gt;
      cv_ext::exp2RotMat(gt_poses[i].r_vec, rot_mat_gt);

      cv::Point3d p_t_diff( match.t_vec(0) - gt_poses[i].t_vec.at<double>(0),
                            match.t_vec(1) - gt_poses[i].t_vec.at<double>(1),
                            match.t_vec(2) - gt_poses[i].t_vec.at<double>(2) );
      
      double rot_diff = 180.0*cv_ext::rotationDist(est_rot_mat, rot_mat_gt)/M_PI, 
             t_diff  = cv_ext::norm3D(p_t_diff);

      if( rot_diff < max_rot_diff_deg && t_diff < max_t_diff )
      {
        dcm_fbm[i_m]++;

//         obj_model.setModelView(match.r_quat, match.t_vec);
//         vector<Point2f> proj_pts;
//         obj_model.projectRasterPoints(proj_pts);        
//         cv::Mat display = src_img.clone();
//         cv_ext::drawPoints( display, proj_pts, Scalar(0,255,0) );
//         cv_ext::showImage(display,"display");

        break;

      }
    }
  }
  stringstream sstr;
  sstr<<"dcm_fbm ";
  sstr<<max_template_pts;
  if( !RGB_MODALITY_ENABLED )
    sstr<<" gl";
  logResults(sstr.str(), dcm_fbm, avg_timer, num_test_images );
  
#endif

#if TEST_LINE2D
  
  avg_timer = 0;
  
  for( int i = 0; i < num_test_images; i++ )
  {
    Mat src_img = imread ( image_names[i].c_str(), cv::IMREAD_COLOR );

//     if(!(i%100))
//       cout<<"Image "<<i<<" of "<<num_test_images<<endl;
    
    vector<Mat> sources;
    sources.push_back(src_img);
    std::vector<cv::linemod::Match> th_matches[n_threads], matches;

    cv_ext::BasicTimer timer;
    #pragma omp parallel if( PARALLELISM_ENABLED )
    {
      int i_th = omp_get_thread_num();
      lm_detector[i_th]->match(sources, (float)linemod_matching_threshold, th_matches[i_th] );
    }
    
    // Combine the results
    for( int i_th = 0; i_th < n_threads; i_th++ )
      matches.insert(matches.end(), th_matches[i_th].begin(), th_matches[i_th].end());
    std::stable_sort (matches.begin(), matches.end() );
    
    avg_timer += timer.elapsedTimeMs();
    
    for (int i_m = 0; i_m < (int)matches.size() && i_m < max_num_matches; i_m++)
    {
      cv::linemod::Match m = matches[i_m];
      std::string class_id_gt = cv::format("rot_%d", i);
      Pose p_est = pose_map[m.class_id], p_gt = pose_map[class_id_gt];

      normalizePose( obj_model, cam_model, Point(m.x, m.y), p_est, template_map[m.class_id] );
      
      Eigen::Matrix3d est_rot_mat, rot_mat_gt;

      cv_ext::exp2RotMat(p_est.r_vec, est_rot_mat);
      cv_ext::exp2RotMat(p_gt.r_vec, rot_mat_gt);
      
      cv::Point3d p_t_diff( p_est.t_vec.at<double>(0) - gt_poses[i].t_vec.at<double>(0),
                            p_est.t_vec.at<double>(1) - gt_poses[i].t_vec.at<double>(1),
                            p_est.t_vec.at<double>(2) - gt_poses[i].t_vec.at<double>(2) );
      
      double rot_diff = 180.0*cv_ext::rotationDist(est_rot_mat, rot_mat_gt)/M_PI,
             t_diff  = cv_ext::norm3D(p_t_diff);

      if( rot_diff < max_rot_diff_deg && t_diff < max_t_diff )
      {
        line_fbm[i_m]++;

//         // Draw matching template
//         cv::Mat display = sources[0].clone(); 
//         vector <Point2f> proj_pts;
//         obj_model.projectRasterPoints( template_map[m.class_id], proj_pts );
//         Point2f off_p(m.x - p_est.bb.x, m.y - p_est.bb.y);
//         for( auto &p : proj_pts ) p += off_p;
//         cv_ext::drawPoints( display, proj_pts, Scalar(0,255,0) );
//         cv_ext::showImage(display,"display");

        break;
      }
    }
  }

  logResults("line_fbm", line_fbm, avg_timer, num_test_images );

#endif

#if TEST_LINE2D_GL
  
  avg_timer = 0;
  
  for( int i = 0; i < num_test_images; i++ )
  {
//     if(!(i%100))
//       cout<<"Image "<<i<<" of "<<num_test_images<<endl;    
    Mat src_img = imread ( image_names[i].c_str(), cv::IMREAD_COLOR );
    Mat gray_src;
    cvtColor( src_img, gray_src, cv::COLOR_BGR2GRAY );
    cvtColor( gray_src, src_img, cv::COLOR_GRAY2BGR );

    vector<Mat> sources;
    sources.push_back(src_img);
    std::vector<cv::linemod::Match> th_matches[n_threads], matches;

    cv_ext::BasicTimer timer;
    #pragma omp parallel if( PARALLELISM_ENABLED )
    {
      int i_th = omp_get_thread_num();
      lm_detector_gl[i_th]->match(sources, (float)linemod_matching_threshold, th_matches[i_th] );
    }
    
    // Combine the results
    for( int i_th = 0; i_th < n_threads; i_th++ )
      matches.insert(matches.end(), th_matches[i_th].begin(), th_matches[i_th].end());
    std::stable_sort (matches.begin(), matches.end());
    
    avg_timer += timer.elapsedTimeMs();
    
    for (int i_m = 0; i_m < (int)matches.size() && i_m < max_num_matches; ++i_m)
    {
      cv::linemod::Match m = matches[i_m];
      std::string class_id_gt = cv::format("rot_%d", i);
      Pose p_est = pose_map_gl[m.class_id], p_gt = pose_map_gl[class_id_gt];
      
      normalizePose( obj_model, cam_model, Point(m.x, m.y), p_est, template_map[m.class_id]);
      
      Eigen::Matrix3d est_rot_mat, rot_mat_gt;

      cv_ext::exp2RotMat(p_est.r_vec, est_rot_mat);
      cv_ext::exp2RotMat(p_gt.r_vec, rot_mat_gt);

      cv::Point3d p_t_diff( p_est.t_vec.at<double>(0) - gt_poses[i].t_vec.at<double>(0),
                            p_est.t_vec.at<double>(1) - gt_poses[i].t_vec.at<double>(1),
                            p_est.t_vec.at<double>(2) - gt_poses[i].t_vec.at<double>(2) );
      
      double rot_diff = 180.0*cv_ext::rotationDist(est_rot_mat, rot_mat_gt)/M_PI,
             t_diff  = cv_ext::norm3D(p_t_diff);

      if( rot_diff < max_rot_diff_deg && t_diff < max_t_diff )
      {
        line_fbm_gl[i_m]++;
//         // Draw matching template
//         cv::Mat display = sources[0].clone(); 
//         vector <Point2f> proj_pts;
//         obj_model.projectRasterPoints( template_map[m.class_id], proj_pts );
//         Point2f off_p(m.x - p_est.bb.x, m.y - p_est.bb.y);
//         for( auto &p : proj_pts ) p += off_p;
//         cv_ext::drawPoints( display, proj_pts, Scalar(0,255,0) );
//         cv_ext::showImage(display,"display");
        break;

      }
    }
  }

  logResults("line_fbm gl", line_fbm_gl, avg_timer, num_test_images );
  
#endif
  
  

  return 0;
}
