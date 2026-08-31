#include "chamfer_matching.h"

#include <omp.h>
#include <algorithm>
#include <stdexcept>

// Enable Ceres (using local Ceres 1.14 install)
#include <ceres/ceres.h>
#define USE_CERES 1

#include "cv_ext/cv_ext.h"

#if defined(D2CO_USE_SSE) || defined(D2CO_USE_AVX)
#include <immintrin.h>
#endif

#if defined(D2CO_USE_NEON)
#include <arm_neon.h>
#endif




using namespace cv;
using namespace cv_ext;
using namespace std;

#define CHAMFER_MATCHING_FAST_MAP 1

static inline int getDirectionIndex( float direction, int num_directions, float eta_direction )
{
  float dir = direction + M_PI/2;
  if( dir >= M_PI/2 )
    dir -= M_PI;
  dir += M_PI/2;

  int i_dir = cv_ext::roundPositive(eta_direction*dir);
  i_dir %= num_directions;
  return i_dir;
}

inline static void setRegion( Mat &m, int cx, int cy, int spacing_size, uchar val )
{
  int mx = cx + spacing_size, my = cy + spacing_size;
  for( int y = cy - spacing_size; y <= my; y++ )
    for( int x = cx - spacing_size; x <= mx; x++ )
      m.at<uchar>(y,x) = val;
}

// Taken from selectScatteredFeatures() in OpenCV linemod implementation
void selectScatteredPoints( vector<Point> &template_pts, int num_pts )
{
  if( int(template_pts.size()) <= num_pts )
    return;
  
  float distance = float(template_pts.size()) / num_pts + 1.0f;

  vector<Point> selected_pts;
  selected_pts.reserve(num_pts);
  float distance_sq = distance * distance;
  int i = 0;
  while (int(selected_pts.size()) < num_pts)
  {
    Point &new_p = template_pts[i];

    bool keep = true;
    for (int j = 0; (j < (int)selected_pts.size()) && keep; ++j)
    {
      Point &p = selected_pts[j];
      keep = (new_p.x - p.x)*(new_p.x - p.x) + (new_p.y - p.y)*(new_p.y - p.y) >= distance_sq;
    }
    if (keep)
      selected_pts.push_back(new_p);

    if (++i == (int)template_pts.size())
    {
      // Start back at beginning, and relax required distance
      i = 0;
      distance -= 1.0f;
      distance_sq = distance * distance;
    }
  }
  
  template_pts = selected_pts;
}

void selectScatteredPoints( vector<Point> &template_pts, vector<int> &template_i_dir,int num_pts )
{
  if( int(template_pts.size()) <= num_pts )
    return;
  
  float distance = float(template_pts.size()) / num_pts + 1.0f;

  vector<Point> selected_pts;
  vector<int> selected_i_dir;
  selected_pts.reserve(num_pts);
  selected_i_dir.reserve(num_pts);
  float distance_sq = distance * distance;
  int i = 0;
  while (int(selected_pts.size()) < num_pts)
  {
    Point &new_p = template_pts[i];

    bool keep = true;
    for (int j = 0; (j < (int)selected_pts.size()) && keep; ++j)
    {
      Point &p = selected_pts[j];
      keep = (new_p.x - p.x)*(new_p.x - p.x) + (new_p.y - p.y)*(new_p.y - p.y) >= distance_sq;
    }
    if (keep)
    {
      selected_pts.push_back(new_p);
      selected_i_dir.push_back(template_i_dir[i]);
    }

    if (++i == (int)template_pts.size())
    {
      // Start back at beginning, and relax required distance
      i = 0;
      distance -= 1.0f;
      distance_sq = distance * distance;
    }
  }
  
  template_pts = selected_pts;
  template_i_dir = selected_i_dir;
}



static void setupExhaustiveMatching_( const RasterObjectModel &obj_model, vector< vector<Point> > &templates_pts,
                                      vector<Rect> &templates_bb, int spacing_size = 1, int max_num_pts = -1 )
{ 
  int w = obj_model.cameraModel().imgWidth(), h = obj_model.cameraModel().imgHeight();
  int num_precomputed_models = obj_model.numPrecomputedModelsViews();

  templates_pts.clear();
  templates_pts.resize(num_precomputed_models);
  templates_bb.resize(num_precomputed_models);

  Size mask_size( obj_model.cameraModel().imgWidth() + 2*spacing_size,
                  obj_model.cameraModel().imgHeight() + 2*spacing_size );  
  Mat template_mask(mask_size, DataType<uchar>::type, Scalar(255));

  vector<Point2f> proj_pts;
  
  for( int i = 0; i < num_precomputed_models; i++ )
  {
    obj_model.projectRasterPoints( i, proj_pts );
    Point tl(w,h), br(0,0);

    int num_pts = 0;
    for( int j = 0; j < int(proj_pts.size()); j++ )
    {
      int x = cvRound(proj_pts[j].x), y = cvRound(proj_pts[j].y),
              mask_x = x + 1, mask_y = y + 1;

      if( unsigned(x) < unsigned(w) && unsigned(y) < unsigned(h) && template_mask.at<uchar>(mask_y,mask_x) )
      {
        if( x < tl.x ) tl.x = x;
        if( y < tl.y ) tl.y = y;
        if( x > br.x ) br.x = x;
        if( y > br.y ) br.y = y;

        templates_pts[i].push_back(Point(x,y));

        setRegion( template_mask, mask_x, mask_y, spacing_size, 0 );
        num_pts++;
      }
    }
    
    br.x++;
    br.y++;
    templates_bb[i] = Rect(tl,br);
    std::vector<cv::Point> &cur_template = templates_pts[i];

    for( int j = 0; j < int(cur_template.size()); j++ )
    {
      int mask_x = cur_template[j].x + spacing_size, mask_y = cur_template[j].y + spacing_size;
      setRegion( template_mask, mask_x, mask_y, spacing_size, 255 );        
    }
    
    if( max_num_pts > 0 && num_pts > max_num_pts )
      selectScatteredPoints(cur_template, max_num_pts);    
  } 
}

static void setupExhaustiveMatching_( const RasterObjectModel &obj_model, int num_directions, float eta_direction,
                                      vector< vector<Point> > &templates_pts, vector< vector<int> > &templates_i_dir,
                                      vector<Rect> &templates_bb, int spacing_size = 1, int max_num_pts = -1 )
{ 
  int w = obj_model.cameraModel().imgWidth(), h = obj_model.cameraModel().imgHeight();
  int num_precomputed_models = obj_model.numPrecomputedModelsViews();

  templates_pts.clear();
  templates_i_dir.clear();
  templates_pts.resize(num_precomputed_models);
  templates_i_dir.resize(num_precomputed_models);
  templates_bb.resize(num_precomputed_models);

  Size mask_size( obj_model.cameraModel().imgWidth() + 2*spacing_size,
                  obj_model.cameraModel().imgHeight() + 2*spacing_size );
  Mat template_mask( mask_size, DataType<uchar>::type, Scalar(255));

  vector<Point2f> proj_pts;
  vector<float> normal_directions;
  
  for( int i = 0; i < num_precomputed_models; i++ )
  {
    obj_model.projectRasterPoints( i, proj_pts, normal_directions );
    Point tl(w,h), br(0,0);

    int num_pts = 0;
    for( int j = 0; j < int(proj_pts.size()); j++ )
    {
      int x = cvRound(proj_pts[j].x), y = cvRound(proj_pts[j].y),
          mask_x = x + 1, mask_y = y + 1;
          
      if( unsigned(x) < unsigned(w) && unsigned(y) < unsigned(h) && template_mask.at<uchar>(mask_y,mask_x) )
      {
        if( x < tl.x ) tl.x = x;
        if( y < tl.y ) tl.y = y;
        if( x > br.x ) br.x = x;
        if( y > br.y ) br.y = y;

        templates_pts[i].push_back(Point(x,y));
        int i_dir = getDirectionIndex( normal_directions[j], num_directions, eta_direction );
        templates_i_dir[i].push_back(i_dir);

        setRegion( template_mask, mask_x, mask_y, spacing_size, 0 );
        num_pts++;
      }
    }
    
    br.x++;
    br.y++;
    templates_bb[i] = Rect(tl,br);
    std::vector<cv::Point> &cur_template = templates_pts[i];
    

    for( int j = 0; j < int(cur_template.size()); j++ )
    {
      int mask_x = cur_template[j].x + spacing_size, mask_y = cur_template[j].y + spacing_size;
      setRegion( template_mask, mask_x, mask_y, spacing_size, 255 );
    }
    
    if( max_num_pts > 0 && num_pts > max_num_pts )
      selectScatteredPoints(cur_template, templates_i_dir[i], max_num_pts);
    
//     // WARNING DEBUG CODE
//     for( int i_dir = 0; i_dir < num_directions; i_dir++ )
//     {
//       Mat dbg_img( mask_size, DataType<cv::Vec3b>::type, Scalar(0));
//       vector< Point > dir_pts;
//        cv_ext::drawPoints(dbg_img, cur_template, Scalar(255,255,255));
//       for( int j_p = 0; j_p < cur_template.size(); j_p++ )
//       {
// //         if( !i_dir )
// //           cout<<templates_i_dir[i][j_p]<<endl;
//         if( templates_i_dir[i][j_p] == i_dir )
//           dir_pts.push_back(cur_template[j_p]);
//       }
// //       cout<<dir_pts.size()<<endl;
//       cv_ext::drawPoints(dbg_img, dir_pts, Scalar(0,0,255));
//       cv_ext::showImage(dbg_img);
//     }
//     // END DEBUG CODE
    
//       Mat dbg_img( mask_size, DataType<cv::Vec3b>::type, Scalar(0));
//       cv_ext::drawPoints(dbg_img, cur_template, Scalar(255,255,255));
//       cv_ext::showImage(dbg_img);
      
  }
}

class ChamferMatching::Optimizer
{
public:
  Optimizer(){};
  ~Optimizer(){};
  ceres::Problem problem;
};

struct ChamferResidual
{
  ChamferResidual ( const cv_ext::PinholeCameraModel &cam_model, const Mat &distance_map,
                    const Point3f &model_pt ) :
    cam_model_ ( cam_model ),
    dist_map_ ( distance_map )
  {
    model_pt_[0] = model_pt.x;
    model_pt_[1] = model_pt.y;
    model_pt_[2] = model_pt.z;
  }

  template <typename _T>
  bool operator() ( const _T* const pos, _T* residuals ) const
  {

    _T model_pt[3] = { _T ( model_pt_[0] ), _T ( model_pt_[1] ), _T ( model_pt_[2] ) };
    _T proj_pt[2];

    cam_model_.quatRTProject ( pos, pos + 4, model_pt, proj_pt );

    if ( proj_pt[0] < _T ( 1 ) ||  proj_pt[1] < _T ( 1 ) ||
         proj_pt[0] > _T ( cam_model_.imgWidth() - 2 ) ||
         proj_pt[1] > _T ( cam_model_.imgHeight() - 2 ) )
    {
      residuals[0] = _T ( 0 );
      return true;
    }

    residuals[0] = cv_ext::bilinearInterp<float, _T> ( dist_map_, proj_pt[0], proj_pt[1] );
    //residuals[0] = cv_ext::getPix<float, _T> ( dist_map_, proj_pt[0], proj_pt[1] );
    return true;
  }

  static ceres::CostFunction* Create ( const cv_ext::PinholeCameraModel &cam_model,
                                       const Mat &dist_map,
                                       const Point3f &model_pt )
  {
    return ( new ceres::NumericDiffCostFunction<ChamferResidual, ceres::CENTRAL, 1, 7 > (
               new ChamferResidual ( cam_model, dist_map, model_pt ) ) );
  }

  const cv_ext::PinholeCameraModel &cam_model_;
  const Mat &dist_map_;
  double model_pt_[3];
};


ChamferMatching :: ChamferMatching()
{
  
}

void ChamferMatching::setInput ( const Mat& dist_map )
{
  if( dist_map.type() != DataType<float>::type || 
      dist_map.rows != cam_model_.imgHeight()|| 
      dist_map.cols  != cam_model_.imgWidth() )
    throw invalid_argument("Invalid input data");  
  
  dist_map_ =  dist_map;
  
#if CHAMFER_MATCHING_FAST_MAP  
  // TODO Switch each map to ushort type?
  dist_map_.convertTo(fast_dist_map_, cv::DataType<ushort>::type);
#endif
}

void ChamferMatching::setupExhaustiveMatching( int max_template_pts )
{
  setupExhaustiveMatching_( *model_ptr_, templates_pts_, templates_bb_, 1, max_template_pts );
}

void ChamferMatching::match(int num_best_matches, vector< TemplateMatch >& matches, int image_step)
{
  const int n_threads = ( parallelism_enabled_ ? omp_get_max_threads() : 1 );

  vector< multimap< double, TemplateMatch > > best_matches(n_threads);
  vector< float > min_dists( n_threads, std::numeric_limits< float >::max());
  const int w = cam_model_.imgWidth(), h = cam_model_.imgHeight();

#if CHAMFER_MATCHING_FAST_MAP && defined(D2CO_USE_SSE)
  
  // TODO Move from here to top
  
//   cv_ext::BasicTimer timer;
    
  vector<Mat> step_map;
  step_map.reserve(image_step);
  
  int step_map_w = cvCeil(float(w)/image_step), step_map_h = h;
  int step_map_data_step = step_map_w + ((step_map_w%8) ? (8 - (step_map_w%8)) : 0);
  int data_chunk_size = step_map_h * step_map_data_step * sizeof(ushort);
  
  // Allocate just one memory portinon for the whole vector of tensors
  void *step_map_data = _mm_malloc (data_chunk_size*image_step, 16);
  char *step_tensor_data_chunk = (char *)step_map_data;
    
  for( int j = 0; j < image_step; j++ )
  {
    step_map.push_back( Mat( step_map_h , step_map_w, DataType<ushort>::type, 
                             step_tensor_data_chunk, step_map_data_step * sizeof(ushort) ) );
    step_tensor_data_chunk += data_chunk_size;
  }
  
  for( int y = 0; y < h; y++)
  {
    ushort *map_p = fast_dist_map_.ptr<ushort>(y);
    vector <ushort *> step_map_p(image_step);
    for( int j = 0; j < image_step; j++ )
      step_map_p[j] = step_map[j].ptr<ushort>(y);
    
    for( int x = 0; x < w; x++, map_p++)
      *step_map_p[x%image_step]++ = *map_p;
  }
//   cout<<"OBJ_REC_FAST_SSE data creation Elapsed time ms : "<<timer.elapsedTimeMs()<<endl;

#endif

  #pragma omp parallel for schedule(dynamic) if( parallelism_enabled_ )
  for( int templ_i = 0; templ_i < int(templates_pts_.size()); templ_i++ )
  {
    int th_id = omp_get_thread_num();
    auto &th_best_matches = best_matches[th_id];
    float &min_dist = min_dists[th_id];
    
    vector<Point> proj_pts = templates_pts_[templ_i];

    // TODO Remove this??
    if( image_step < 1 )
    {
      float dist = templateDist(proj_pts);

      if( int(th_best_matches.size()) < num_best_matches )
      {
        Eigen::Quaterniond r_quat;
        Eigen::Vector3d t_vec;
        model_ptr_->modelView(templ_i,r_quat,t_vec);
        th_best_matches.insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist,r_quat,t_vec, Point(0,0))) );
        min_dist = th_best_matches.rbegin()->first;
      }
      else if( dist < min_dist )
      {
        Eigen::Quaterniond r_quat;
        Eigen::Vector3d t_vec;
        model_ptr_->modelView(templ_i,r_quat,t_vec);
        th_best_matches.insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist,r_quat,t_vec, Point(0,0))) );
        th_best_matches.erase(--th_best_matches.rbegin().base());
        min_dist = th_best_matches.rbegin()->first;
      }
    }
    else
    {
      Rect &bb = templates_bb_[templ_i];
      Point tl = bb.tl();
      for( int i_p = 0; i_p < int(proj_pts.size()); i_p++ )
        proj_pts[i_p] -= tl;

      int dists_w = (w - bb.width)/image_step + 1, dists_h = (h - bb.height)/image_step + 1;      
      Mat_<float> avg_dists(dists_h, dists_w);
      
#if CHAMFER_MATCHING_FAST_MAP

#if defined(D2CO_USE_SSE)

      #if 1
      int aligned_c_dists_w = dists_w - (dists_w%8), c_dists_data_step = aligned_c_dists_w + ((dists_w%8)?8:0);

      // TODO Move this away (avoid to relocate for each template)?      
      void *c_dists_data = _mm_malloc (dists_h * c_dists_data_step * sizeof(ushort), 16);
      Mat c_dists(dists_h, dists_w, DataType<ushort>::type, c_dists_data, c_dists_data_step * sizeof(ushort));      
      memset(c_dists_data, 0, dists_h * c_dists_data_step * sizeof(ushort));
      
      __m128i map_dist128i, avg_dist128i, *c_dists128i_p, *dist_map128i_p;
      int vec_c_dists_w = aligned_c_dists_w/8;
      for( int i_p = 0; i_p < int(proj_pts.size()); i_p++ )
      {
        int x = proj_pts[i_p].x;
        Mat &dist_map = step_map[x%image_step];

        for( int y = proj_pts[i_p].y, c_dists_y = 0; c_dists_y < dists_h; y += image_step, c_dists_y++ )
        {
          c_dists128i_p = (__m128i *)c_dists.ptr<ushort>(c_dists_y);
          dist_map128i_p = (__m128i *)(dist_map.ptr<ushort>(y) + x/image_step);  

          for(int vec_c_dists_x = 0; vec_c_dists_x < vec_c_dists_w; dist_map128i_p++, c_dists128i_p++, vec_c_dists_x++ )
          {
            map_dist128i = _mm_loadu_si128(dist_map128i_p);
            avg_dist128i = _mm_load_si128(c_dists128i_p);
            avg_dist128i = _mm_adds_epu16(avg_dist128i, map_dist128i);
            _mm_store_si128(c_dists128i_p, avg_dist128i );
          }
          ushort *dist_map_p = (ushort *)dist_map128i_p, *dists_p = (ushort *)c_dists128i_p;
          for( int c_dists_x = aligned_c_dists_w; c_dists_x < dists_w; 
               dist_map_p++, dists_p++, c_dists_x++ )
            *dists_p += *dist_map_p;
        }
      }

#else      
      cv_ext::AlignedMat c_dists(dists_h, dists_w, DataType<ushort>::type, cv_ext::MEM_ALIGN_16);
      __m128i map_dist128i, avg_dist128i, *c_dists128i_p, *dist_map128i_p;
      int vec_c_dists_w = c_dists.step1/8;
      for( int i_p = 0; i_p < int(proj_pts.size()); i_p++ )
      {
        int x = proj_pts[i_p].x;
        Mat &dist_map = step_map[x%image_step];

        for( int y = proj_pts[i_p].y, c_dists_y = 0; c_dists_y < dists_h; y += image_step, c_dists_y++ )
        {
          c_dists128i_p = (__m128i *)c_dists.ptr<ushort>(c_dists_y);
          dist_map128i_p = (__m128i *)(dist_map.ptr<ushort>(y) + x/image_step);  

          for(int vec_c_dists_x = 0; vec_c_dists_x < vec_c_dists_w; dist_map128i_p++, c_dists128i_p++, vec_c_dists_x++ )
          {
            map_dist128i = _mm_loadu_si128(dist_map128i_p);
            avg_dist128i = _mm_load_si128(c_dists128i_p);
            avg_dist128i = _mm_adds_epu16(avg_dist128i, map_dist128i);
            _mm_store_si128(c_dists128i_p, avg_dist128i );
          }
        }
      }
     #endif 
#else // defined(D2CO_USE_SSE)
      // TODO Check here
      Mat c_dists( dists_h, dists_w, DataType<ushort>::type, Scalar(0));
      for( int i_p = 0; i_p < proj_pts.size(); i_p++ )
      {
        for( int y = proj_pts[i_p].y, c_dists_y = 0; c_dists_y < dists_h; y += image_step, c_dists_y++ )
        {
          ushort *dists_p = c_dists.ptr<ushort>(c_dists_y);
          ushort *dist_map_p = fast_dist_map_.ptr<ushort>(y) + proj_pts[i_p].x;

          for(int c_dists_x = 0; c_dists_x < dists_w; dist_map_p += image_step, dists_p++, c_dists_x++ )
            *dists_p += *dist_map_p;
        }
      }
#endif

      float norm_factor = 1.0f/proj_pts.size();
      c_dists.convertTo(avg_dists, cv::DataType< float >::type, norm_factor);   

#else // CHAMFER_MATCHING_FAST_MAP


      // TODO Check here
      avg_dists.setTo(Scalar(0));
      for( int i_p = 0; i_p < proj_pts.size(); i_p++ )
      {
        // TODO center the searched area
        for( int y = proj_pts[i_p].y, avg_dists_y = 0; avg_dists_y < dists_h; y += image_step, avg_dists_y++ )
        {
          float *dists_p = avg_dists.ptr<float>(avg_dists_y);
          float *dist_map_p = dist_map_.ptr<float>(y) + proj_pts[i_p].x;

          for(int avg_dists_x = 0; avg_dists_x < dists_w; dist_map_p += image_step, dists_p++, avg_dists_x++ )
            *dists_p += *dist_map_p;
        }
      }

      avg_dists *= 1.0f/proj_pts.size();
      
#endif
      
      for(int avg_dists_y = 0; avg_dists_y < dists_h; avg_dists_y++ )
      {
        float *dists_p = avg_dists.ptr<float>(avg_dists_y);
        for(int avg_dists_x = 0; avg_dists_x < dists_w; avg_dists_x++, dists_p++ )
        {
          float dist = *dists_p;
          if( int(th_best_matches.size()) < num_best_matches )
          {
            Eigen::Quaterniond r_quat;
            Eigen::Vector3d t_vec;
            model_ptr_->modelView(templ_i,r_quat,t_vec);
            Point offset( avg_dists_x*image_step, avg_dists_y*image_step);
            offset -= bb.tl();
            th_best_matches.insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist,r_quat,t_vec, offset)) );
            min_dist = th_best_matches.rbegin()->first;
          }
          else if( dist < min_dist )
          {
            Eigen::Quaterniond r_quat;
            Eigen::Vector3d t_vec;
            model_ptr_->modelView(templ_i,r_quat,t_vec);
            Point offset( avg_dists_x*image_step, avg_dists_y*image_step);
            offset -= bb.tl();
            th_best_matches.insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist,r_quat,t_vec, offset)) );
            th_best_matches.erase(--th_best_matches.rbegin().base());
            min_dist = th_best_matches.rbegin()->first;
          }
        }
      }      
    }
  }


  
#if CHAMFER_MATCHING_FAST_MAP && defined(D2CO_USE_SSE)
  // TODO Move from here to top
   _mm_free(step_map_data);
#endif
  
  for( int i = 1; i < n_threads; i++ )
    best_matches[0].insert(best_matches[i].begin(), best_matches[i].end());

  matches.clear();
  matches.reserve(num_best_matches);
  int num_matches = 0;
  for( auto iter = best_matches[0].begin();
       iter != best_matches[0].end() && num_matches < num_best_matches;
       iter++, num_matches++ )
  {
//     cout<<iter->second.img_offset<<" "<<iter->second.distance<<endl;
    normalizeMatch(iter->second);
    matches.push_back(iter->second);
  }
//   for( auto iter = best_pts.begin(); iter != best_pts.end(); iter++ )
//   {
//     Mat dbg_img2 = dbg_img.clone();
// //     Mat dbg_img(cam_model_.imgSize(), DataType<uchar>::type, Scalar(255));
//     vector< Point > &pts = iter->second;
//     for( auto &p : pts )
//       cout<<p;
//     cout<<endl;
//     cv_ext::drawPoints(dbg_img2, pts, Scalar(0,0,255));
//     cv_ext::showImage(dbg_img2);
//   }
}

inline float ChamferMatching::templateDist( const std::vector<cv::Point> &proj_pts )
{
  double avg_dist = 0;
  for( int i = 0; i < int(proj_pts.size()); i++ )
#if CHAMFER_MATCHING_FAST_MAP
    avg_dist += fast_dist_map_.at<ushort>( proj_pts[i].y, proj_pts[i].x );
#else  
    avg_dist += dist_map_.at<float>( proj_pts[i].y, proj_pts[i].x );
#endif    
    

  if( proj_pts.size() )
    avg_dist /= proj_pts.size();
  else
    avg_dist = std::numeric_limits< float >::max();

  return float(avg_dist);
}

double ChamferMatching::optimize()
{
  if( !optimizer_ptr_ )
    return -1;

  ceres::Solver::Options options;
  //options.linear_solver_type = ceres::DENSE_SCHUR;
  options.max_num_iterations = optimizer_num_iterations_;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = verbouse_mode_;

  ceres::Solver::Summary summary;
  ceres::Solve ( options, &optimizer_ptr_->problem, &summary );

  return summary.final_cost;
}

double ChamferMatching::avgDistance( int idx )
{
  int n_pts = 0;
  double avg_dist = 0;

  std::vector<Point2f> proj_pts;

  if( idx < 0 )
    model_ptr_->projectRasterPoints( proj_pts );
  else
    model_ptr_->projectRasterPoints(idx, proj_pts );

  for( int i = 0; i < int(proj_pts.size()); i++ )
  {
    const Point2f &coord = proj_pts.at(i);
    if( coord.x >= 0)
    {
      n_pts++;
      int x = cvRound(coord.x), y = cvRound(coord.y);
      avg_dist += dist_map_.at<float>( y, x );
    }
  }

  if( n_pts )
    avg_dist /= n_pts;
  else
    avg_dist = std::numeric_limits< float >::max();

  return avg_dist;
}

void ChamferMatching::updateOptimizer( int idx )
{
  const std::vector<Point3f> &model_pts = ( idx < 0 )?model_ptr_->getPoints():model_ptr_->getPrecomputedPoints(idx);

  optimizer_ptr_ = std::shared_ptr< Optimizer > ( new Optimizer () );

  for ( int i = 0; i < int(model_pts.size()); ++i )
  {
    ceres::CostFunction* cost_function =
      ChamferResidual::Create ( cam_model_, dist_map_, model_pts[i] );

    optimizer_ptr_->problem.AddResidualBlock ( cost_function, new ceres::HuberLoss(1.0), transf_.data() );
  }
}

OrientedChamferMatching::OrientedChamferMatching()
{
  setNumDirections( 60 );  
}

void OrientedChamferMatching::setNumDirections ( int n )
{
  num_directions_ = n;
  eta_direction_ = double(num_directions_)/M_PI;
}

void OrientedChamferMatching::setInput ( const Mat& dist_map, const cv::Mat &closest_dir_map )
{
  if( dist_map.type() != DataType<float>::type ||
      dist_map.rows != cam_model_.imgHeight()||
      dist_map.cols  != cam_model_.imgWidth() ||
      closest_dir_map.type() != DataType<ushort>::type ||
      closest_dir_map.rows != cam_model_.imgHeight()||
      closest_dir_map.cols  != cam_model_.imgWidth() )
    throw invalid_argument("Invalid input data");  
  
  dist_map_ =  dist_map;
  closest_dir_map_ = closest_dir_map;

  #if CHAMFER_MATCHING_FAST_MAP
  // TODO Switch each map to ushort type?
  dist_map_.convertTo(fast_dist_map_, cv::DataType<ushort>::type);  
#endif
}

void OrientedChamferMatching::setupExhaustiveMatching( int max_template_pts )
{
  setupExhaustiveMatching_( *model_ptr_, num_directions_, eta_direction_, 
                            templates_pts_, templates_i_dir_, templates_bb_, 1, max_template_pts );
}

void OrientedChamferMatching::match(int num_best_matches, vector< TemplateMatch >& matches, int image_step)
{
  const int n_threads = ( parallelism_enabled_ ? omp_get_max_threads() : 1 );

  vector< multimap< double, TemplateMatch > > best_matches(n_threads);
  vector< float > min_dists( n_threads, std::numeric_limits< float >::max());
  const int w = cam_model_.imgWidth(), h = cam_model_.imgHeight();

#if CHAMFER_MATCHING_FAST_MAP && defined(D2CO_USE_SSE)
  
  // TODO
  std::cerr<<"IMPLEMENT ME!!"<<std::endl;

#endif

  #pragma omp parallel for schedule(dynamic) if( parallelism_enabled_ )
  for( int templ_i = 0; templ_i < int(templates_pts_.size()); templ_i++ )
  {
    int th_id = omp_get_thread_num();
    auto &th_best_matches = best_matches[th_id];
    float &min_dist = min_dists[th_id];
    
    vector<Point> proj_pts = templates_pts_[templ_i];
    const vector<int> &i_dirs = templates_i_dir_[templ_i];

    // TODO Remove this??
    if( image_step < 1 )
    {
      float dist = templateDist(proj_pts, i_dirs);

      if( int(th_best_matches.size()) < num_best_matches )
      {
        Eigen::Quaterniond r_quat;
        Eigen::Vector3d t_vec;
        model_ptr_->modelView(templ_i,r_quat,t_vec);
        th_best_matches.insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist,r_quat,t_vec, Point(0,0))) );
        min_dist = th_best_matches.rbegin()->first;
      }
      else if( dist < min_dist )
      {
        Eigen::Quaterniond r_quat;
        Eigen::Vector3d t_vec;
        model_ptr_->modelView(templ_i,r_quat,t_vec);
        th_best_matches.insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist,r_quat,t_vec, Point(0,0))) );
        th_best_matches.erase(--th_best_matches.rbegin().base());
        min_dist = th_best_matches.rbegin()->first;
      }
    }
    else
    {
      Rect &bb = templates_bb_[templ_i];
      Point tl = bb.tl();
      for( int i_p = 0; i_p < int(proj_pts.size()); i_p++ )
        proj_pts[i_p] -= tl;

      int dists_w = (w - bb.width)/image_step + 1, dists_h = (h - bb.height)/image_step + 1;      
      Mat_<float> avg_dists(dists_h, dists_w);
      
#if CHAMFER_MATCHING_FAST_MAP

#if defined(D2CO_USE_SSE)
      
  // TODO
  std::cerr<<"IMPLEMENT ME!!"<<std::endl;
  
//       int aligned_c_dists_w = dists_w - (dists_w%8), c_dists_data_step = aligned_c_dists_w + ((dists_w%8)?8:0);
// 
//       // TODO Move this away (avoid to relocate for each template)?      
//       void *c_dists_data = _mm_malloc (dists_h * c_dists_data_step * sizeof(ushort), 16);
//       Mat c_dists(dists_h, dists_w, DataType<ushort>::type, c_dists_data, c_dists_data_step * sizeof(ushort));      
//       memset(c_dists_data, 0, dists_h * c_dists_data_step * sizeof(ushort));
//       
//       __m128i map_dist128i, avg_dist128i, *c_dists128i_p, *dist_map128i_p;
//       int vec_c_dists_w = aligned_c_dists_w/8;
//       for( int i_p = 0; i_p < proj_pts.size(); i_p++ )
//       {
//         int x = proj_pts[i_p].x;
//         Mat &dist_map = step_tensor[x%image_step][i_dirs[i_p]];
// 
//         for( int y = proj_pts[i_p].y, c_dists_y = 0; c_dists_y < dists_h; y += image_step, c_dists_y++ )
//         {
//           c_dists128i_p = (__m128i *)c_dists.ptr<ushort>(c_dists_y);
//           dist_map128i_p = (__m128i *)(dist_map.ptr<ushort>(y) + x/image_step);  
// 
//           for(int vec_c_dists_x = 0; vec_c_dists_x < vec_c_dists_w; dist_map128i_p++, c_dists128i_p++, vec_c_dists_x++ )
//           {
//             map_dist128i = _mm_loadu_si128(dist_map128i_p);
//             avg_dist128i = _mm_load_si128(c_dists128i_p);
//             avg_dist128i = _mm_adds_epu16(avg_dist128i, map_dist128i);
//             _mm_store_si128(c_dists128i_p, avg_dist128i );
//           }
//           ushort *dist_map_p = (ushort *)dist_map128i_p, *dists_p = (ushort *)c_dists128i_p;
//           for( int c_dists_x = aligned_c_dists_w; c_dists_x < dists_w; 
//                dist_map_p++, dists_p++, c_dists_x++ )
//             *dists_p += *dist_map_p;
//         }
//       }
      
#else // defined(D2CO_USE_SSE)

      Mat c_dists( dists_h, dists_w, DataType<ushort>::type, Scalar(0));
      for( int i_p = 0; i_p < proj_pts.size(); i_p++ )
      {
        for( int y = proj_pts[i_p].y, c_dists_y = 0; c_dists_y < dists_h; y += image_step, c_dists_y++ )
        {
          ushort *dists_p = c_dists.ptr<ushort>(c_dists_y);
          ushort *dist_map_p = fast_dist_map_.ptr<ushort>(y) + proj_pts[i_p].x;

          for(int c_dists_x = 0; c_dists_x < dists_w; dist_map_p += image_step, dists_p++, c_dists_x++ )
            *dists_p += *dist_map_p;
        }
      }

      for( int i_p = 0; i_p < proj_pts.size(); i_p++ )
      {
        ushort point_dir = i_dirs[i_p];
        for( int y = proj_pts[i_p].y, c_dists_y = 0; c_dists_y < dists_h; y += image_step, c_dists_y++ )
        {
          ushort *dists_p = c_dists.ptr<ushort>(c_dists_y);
          ushort *dir_map_p = closest_dir_map_.ptr<ushort>(y) + proj_pts[i_p].x;

          for(int c_dists_x = 0; c_dists_x < dists_w; dir_map_p += image_step, dists_p++, c_dists_x++ )
          {
            ushort ang_diff = (point_dir > *dir_map_p)?(point_dir - *dir_map_p):(*dir_map_p - point_dir);
            *dists_p += ang_diff;
          }
        }
      }
      
#endif
// WARNING Commented out for debug
//       float norm_factor = 1.0f/proj_pts.size();
//       c_dists.convertTo(avg_dists, cv::DataType< float >::type, norm_factor);
      
#if defined(D2CO_USE_SSE)
      // TODO Move from here to top
// WARNING Commented out for debug      
//       _mm_free(c_dists_data);
#endif      

#else // CHAMFER_MATCHING_FAST_MAP


      // TODO Check here
      avg_dists.setTo(Scalar(0));
      for( int i_p = 0; i_p < proj_pts.size(); i_p++ )
      {
        int point_dir = i_dirs[i_p];
        // TODO center the searched area
        for( int y = proj_pts[i_p].y, avg_dists_y = 0; avg_dists_y < dists_h; y += image_step, avg_dists_y++ )
        {
          float *dists_p = avg_dists.ptr<float>(avg_dists_y);
          
          float *dist_map_p = dist_map_.ptr<float>(y) + proj_pts[i_p].x;
          ushort *dir_map_p = closest_dir_map_.ptr<ushort>(y) + proj_pts[i_p].x;

          for(int avg_dists_x = 0; avg_dists_x < dists_w; dist_map_p += image_step, 
              dir_map_p += image_step, dists_p++, avg_dists_x++ )
          {
            float ang_diff = point_dir - *dir_map_p;
            *dists_p += *dist_map_p + (ang_diff<0)?-ang_diff:ang_diff;
            }
        }
      }

      avg_dists *= 1.0f/proj_pts.size();
      
#endif
      
      for(int avg_dists_y = 0; avg_dists_y < dists_h; avg_dists_y++ )
      {
        float *dists_p = avg_dists.ptr<float>(avg_dists_y);
        for(int avg_dists_x = 0; avg_dists_x < dists_w; avg_dists_x++, dists_p++ )
        {
          float dist = *dists_p;
          if( int(th_best_matches.size()) < num_best_matches )
          {
            Eigen::Quaterniond r_quat;
            Eigen::Vector3d t_vec;
            model_ptr_->modelView(templ_i,r_quat,t_vec);
            Point offset( avg_dists_x*image_step, avg_dists_y*image_step);
            offset -= bb.tl();

            th_best_matches.insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist,r_quat,t_vec, offset)) );
            //             best_pts[dist] = proj_pts;
            min_dist = th_best_matches.rbegin()->first;
            //             cout<<"INSERTED"<<endl;
          }
          else if( dist < min_dist )
          {
            Eigen::Quaterniond r_quat;
            Eigen::Vector3d t_vec;
            model_ptr_->modelView(templ_i,r_quat,t_vec);
            Point offset( avg_dists_x*image_step, avg_dists_y*image_step);
            offset -= bb.tl();
            th_best_matches.insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist,r_quat,t_vec, offset)) );
            //             best_pts[dist] = proj_pts;
            th_best_matches.erase(--th_best_matches.rbegin().base());
            //             best_pts.erase(--best_pts.rbegin().base());
            min_dist = th_best_matches.rbegin()->first;
            //             cout<<"INSERTED 2"<<endl;
          }
        }
      }      
    }
  }


  
#if CHAMFER_MATCHING_FAST_MAP && defined(D2CO_USE_SSE)
  // TODO Move from here to top
// WARNING Commented out for debug
//    _mm_free(step_tensor_data);
#endif
  
  for( int i = 1; i < n_threads; i++ )
    best_matches[0].insert(best_matches[i].begin(), best_matches[i].end());

  matches.clear();
  matches.reserve(num_best_matches);
  int num_matches = 0;
  for( auto iter = best_matches[0].begin();
       iter != best_matches[0].end() && num_matches < num_best_matches;
       iter++, num_matches++ )
  {
//     cout<<iter->second.img_offset<<" "<<iter->second.distance<<endl;
    normalizeMatch(iter->second);
    matches.push_back(iter->second);
  }
//   for( auto iter = best_pts.begin(); iter != best_pts.end(); iter++ )
//   {
//     Mat dbg_img2 = dbg_img.clone();
// //     Mat dbg_img(cam_model_.imgSize(), DataType<uchar>::type, Scalar(255));
//     vector< Point > &pts = iter->second;
//     for( auto &p : pts )
//       cout<<p;
//     cout<<endl;
//     cv_ext::drawPoints(dbg_img2, pts, Scalar(0,0,255));
//     cv_ext::showImage(dbg_img2);
//   }
}

float OrientedChamferMatching::templateDist ( const vector< Point >& proj_pts, 
                                              const vector< int >& dirs )
{
  std::cerr<<"IMPLEMENT ME!!"<<std::endl;
  return -1;
}

double OrientedChamferMatching::optimize()
{
  std::cerr<<"IMPLEMENT ME!!"<<std::endl;
  return -1;
}

double OrientedChamferMatching::avgDistance( int idx )
{

  std::cerr<<"IMPLEMENT ME!!"<<std::endl;
//   int n_pts = 0;
//   double avg_dist = 0;
// 
//   std::vector<Point2f> proj_pts;
// 
//   if( idx < 0 )
//     model_ptr_->projectRasterPoints( proj_pts );
//   else
//     model_ptr_->projectRasterPoints(idx, proj_pts );
// 
//   for( int i = 0; i < proj_pts.size(); i++ )
//   {
//     const Point2f &coord = proj_pts.at(i);
//     if( coord.x >= 0)
//     {
//       n_pts++;
//       int x = cvRound(coord.x), y = cvRound(coord.y);
//       avg_dist += dist_map_.at<float>( y, x );
//     }
//   }
// 
//   if( n_pts )
//     avg_dist /= n_pts;
//   else
//     avg_dist = std::numeric_limits< float >::max();
// 
//   return avg_dist;
  return -1;
}

void OrientedChamferMatching::updateOptimizer( int idx )
{
  std::cerr<<"IMPLEMENT ME!!"<<std::endl;
}



struct ICPChamferResidual
{
  ICPChamferResidual ( const cv_ext::PinholeCameraModel &cam_model,
                       const Point3f &model_pt,
                       const Point2f &img_pt ) :
    cam_model_ ( cam_model )
  {
    model_pt_[0] = model_pt.x;
    model_pt_[1] = model_pt.y;
    model_pt_[2] = model_pt.z;
    img_pt_[0] = img_pt.x;
    img_pt_[1] = img_pt.y;
  }

  template <typename _T>
  bool operator() ( const _T* const pos, _T* residuals ) const
  {

    _T model_pt[3] = { _T ( model_pt_[0] ), _T ( model_pt_[1] ), _T ( model_pt_[2] ) };
    _T img_pt[2] = {_T ( img_pt_[0] ), _T ( img_pt_[1] ) }, proj_pt[2];

    cam_model_.quatRTProject ( pos, pos + 4, model_pt, proj_pt );

    if ( proj_pt[0] < _T ( 1 ) ||  proj_pt[1] < _T ( 1 ) ||
         proj_pt[0] > _T ( cam_model_.imgWidth() - 2 ) ||
         proj_pt[1] > _T ( cam_model_.imgHeight() - 2 ) )
    {
      residuals[0] = _T ( 0 );
      residuals[1] = _T ( 0 );
      return true;
    }

    residuals[0] = img_pt[0] - proj_pt[0];
    residuals[1] = img_pt[1] - proj_pt[1];

    return true;
  }

  static ceres::CostFunction* Create ( const cv_ext::PinholeCameraModel &cam_model,
                                       const Point3f &model_pt, const Point2f &img_pt )
  {
    return ( new ceres::NumericDiffCostFunction<ICPChamferResidual, ceres::CENTRAL, 2, 7 > (
               new ICPChamferResidual ( cam_model, model_pt, img_pt ) ) );
  }

  const cv_ext::PinholeCameraModel &cam_model_;
  double model_pt_[3], img_pt_[2];
};

class ICPChamferMatching::Optimizer
{
public:
  Optimizer(){};
  ~Optimizer(){};
  ceres::Problem problem;
};

ICPChamferMatching::ICPChamferMatching () :
  num_icp_iterations_(50),
  selected_idx_(-1)
{
}

void ICPChamferMatching::setInput ( const Mat& closest_edgels_map )
{  
  if( closest_edgels_map.type() != DataType<Point2f>::type || 
      closest_edgels_map.rows != cam_model_.imgHeight()|| 
      closest_edgels_map.cols  != cam_model_.imgWidth() )
    throw invalid_argument("Invalid input data");  
  
  closest_edgels_map_ =  closest_edgels_map;
}

void ICPChamferMatching::setupExhaustiveMatching( int max_template_pts )
{
  std::cerr<<"IMPLEMENT ME!!"<<std::endl;
//   exit(-1);
}

void ICPChamferMatching::match(int num_best_matches, vector< TemplateMatch >& matches, int image_step)
{
  std::cerr<<"IMPLEMENT ME!!"<<std::endl;
}

double ICPChamferMatching::avgDistance ( int idx )
{
  int n_pts = 0;
  double avg_dist = 0;

  std::vector<Point2f> proj_pts;

  if( idx < 0 )
    model_ptr_->projectRasterPoints( proj_pts );
  else
    model_ptr_->projectRasterPoints(idx, proj_pts );

  for( int i = 0; i < int(proj_pts.size()); i++ )
  {
    const Point2f &coord = proj_pts.at(i);
    if( coord.x >= 0)
    {
      n_pts++;
      int x = cvRound(coord.x), y = cvRound(coord.y);
      const Point2f &closest_edgel = closest_edgels_map_.at<Point2f>( y, x );
      avg_dist += sqrt( (closest_edgel.x - coord.x)*(closest_edgel.x - coord.x) +
                         (closest_edgel.y - coord.y)*(closest_edgel.y - coord.y) );
    }
  }

  if( n_pts )
    avg_dist /= n_pts;
  else
    avg_dist = std::numeric_limits< float >::max();

  return avg_dist;
}

void ICPChamferMatching::updateOptimizer ( int idx )
{
  selected_idx_ = idx;
  model_pts_ = ( idx < 0 )?model_ptr_->getPoints():model_ptr_->getPrecomputedPoints(idx);
}

double ICPChamferMatching::optimize()
{

  double final_cost = -1.0;
  for( int icp_iter = 0; icp_iter < num_icp_iterations_; icp_iter++)
  {
    std::vector<Point2f> img_pts;
    if( selected_idx_ < 0 )
      model_ptr_->projectRasterPoints( transf_.data(), transf_.block<3,1>(4,0).data(), img_pts );
    else
      model_ptr_->projectRasterPoints( selected_idx_, transf_.data(), transf_.block<3,1>(4,0).data(), img_pts );

    optimizer_ptr_ = std::shared_ptr< Optimizer > ( new Optimizer () );

    int res_size = 0;
    for ( int i = 0; i < int(model_pts_.size()); ++i )
    {
      if( img_pts[i].x >= 0 )
      {
        int x = cvRound(img_pts[i].x), y = cvRound(img_pts[i].y);
        const Point2f &closest_edgel = closest_edgels_map_.at<Point2f>( y, x );
        ceres::CostFunction* cost_function =
          ICPChamferResidual::Create ( cam_model_, model_pts_[i], closest_edgel );

        optimizer_ptr_->problem.AddResidualBlock ( cost_function, new ceres::HuberLoss(1.0), transf_.data() );
        res_size++;
      }
    }
    if( res_size >= 3 )
    {
      ceres::Solver::Options options;
      //options.linear_solver_type = ceres::DENSE_SCHUR;
      options.max_num_iterations = optimizer_num_iterations_;
      options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
      options.minimizer_progress_to_stdout = verbouse_mode_;

      ceres::Solver::Summary summary;
      ceres::Solve ( options, &optimizer_ptr_->problem, &summary );
      final_cost = summary.final_cost;
    }
  }
  return final_cost;
}

class DirectionalChamferMatching::Optimizer
{
public:
  Optimizer(){};
  ~Optimizer(){};
  ceres::Problem problem;
};


struct DirectionalChamferResidual
{
  DirectionalChamferResidual ( const cv_ext::PinholeCameraModel &cam_model,
                               const ImageTensorPtr &dist_map_tensor_ptr,
                               const Point3f &model_pt, const Point3f &model_dpt ) :
    cam_model_ ( cam_model ),
    dist_map_tensor_ptr_ ( dist_map_tensor_ptr ),
    dist_map_tensor_( *dist_map_tensor_ptr_ )
  {
    model_pt_[0] = model_pt.x;
    model_pt_[1] = model_pt.y;
    model_pt_[2] = model_pt.z;

    model_dpt_[0] = model_dpt.x;
    model_dpt_[1] = model_dpt.y;
    model_dpt_[2] = model_dpt.z;

    eta_direction_ = double(dist_map_tensor_.depth())/M_PI;

    //imshow("tensor[0]", dist_map_tensor_[23]);
    //waitKey(0);
  }

  template <typename _T>
  bool operator() ( const _T* const pos, _T* residuals ) const
  {

    _T model_pt[3] = { _T ( model_pt_[0] ), _T ( model_pt_[1] ), _T ( model_pt_[2] ) };
    _T model_dpt[3] = { _T ( model_dpt_[0] ), _T ( model_dpt_[1] ), _T ( model_dpt_[2] ) };
    _T proj_pt[2], proj_dpt[2];

    cam_model_.quatRTProject ( pos, pos + 4, model_pt, proj_pt );
    cam_model_.quatRTProject ( pos, pos + 4, model_dpt, proj_dpt );

   if ( proj_pt[0] < _T ( 1 ) ||  proj_pt[1] < _T ( 1 ) ||
        proj_pt[0] > _T ( cam_model_.imgWidth() - 2 ) ||
        proj_pt[1] > _T ( cam_model_.imgHeight() - 2 ) ||
        proj_dpt[0] < _T ( 1 ) ||  proj_dpt[1] < _T ( 1 ) ||
        proj_dpt[0] > _T ( cam_model_.imgWidth() - 2 ) ||
        proj_dpt[1] > _T ( cam_model_.imgHeight() - 2 ) )
    {
      residuals[0] = _T ( 0 );
      return true;
    }

    _T diff[2] = { proj_dpt[0] - proj_pt[0], proj_dpt[1] - proj_pt[1] };
    _T direction;

    if( diff[0] != _T(0) )
      direction = atan( diff[1]/diff[0] );
    else
      direction = _T(-M_PI/2);

    _T z = _T(eta_direction_) * ( direction + _T(M_PI/2) );
    //residuals[0] = cv_ext::tensorGetPix<float, _T> ( dist_map_tensor_, proj_pt[0], proj_pt[1], z );
    residuals[0] = cv_ext::tensorbilinearInterp<float, _T> ( dist_map_tensor_, proj_pt[0], proj_pt[1], z );
    return true;
  }

  static ceres::CostFunction* Create ( const cv_ext::PinholeCameraModel &cam_model,
                                       const ImageTensorPtr &dist_map_tensor_ptr,
                                       const Point3f &model_pt, const Point3f &model_dpt)
  {
    return ( new ceres::NumericDiffCostFunction<DirectionalChamferResidual, ceres::CENTRAL, 1, 7 > (
               new DirectionalChamferResidual( cam_model, dist_map_tensor_ptr, model_pt, model_dpt ) ) );
  }

  const cv_ext::PinholeCameraModel &cam_model_;
  const ImageTensorPtr dist_map_tensor_ptr_;
  const ImageTensor &dist_map_tensor_;
  double model_pt_[3], model_dpt_[3];
  double eta_direction_;
};


DirectionalChamferMatching::DirectionalChamferMatching()
{
  setNumDirections(60);
}

void DirectionalChamferMatching::setNumDirections(int n )
{
  num_directions_ = n;
  eta_direction_ = double(num_directions_)/M_PI;
}

void DirectionalChamferMatching::setInput( const ImageTensorPtr& dist_map_tensor_ptr )
{
  if( dist_map_tensor_ptr->depth() != num_directions_ ||
      dist_map_tensor_ptr->at(0).type() != DataType<float>::type || 
      dist_map_tensor_ptr->at(0).rows != cam_model_.imgHeight()|| 
      dist_map_tensor_ptr->at(0).cols  != cam_model_.imgWidth() )
    throw invalid_argument("Invalid input data");  
  
  dist_map_tensor_ptr_ = dist_map_tensor_ptr;

  // Peor valor presente en el tensor: es el coste que se asigna a los puntos
  // del modelo que caen FUERA de la imagen (ver getDistance). Sin el, sacar
  // puntos del encuadre ABARATABA la puntuacion en vez de encarecerla.
  out_of_image_penalty_ = 0.0f;
  for( int i = 0; i < num_directions_; i++ )
  {
    double min_v, max_v;
    cv::minMaxLoc( dist_map_tensor_ptr_->at(i), &min_v, &max_v );
    if( float(max_v) > out_of_image_penalty_ )
      out_of_image_penalty_ = float(max_v);
  }

#if CHAMFER_MATCHING_FAST_MAP
  // TODO Switch each map to ushort type?
  fast_dist_map_tensor_.create(dist_map_tensor_ptr->at(0).rows, dist_map_tensor_ptr->at(0).cols, 
                               num_directions_, cv::DataType<ushort>::type);
  
  for( int i = 0; i < num_directions_; i++ )
    dist_map_tensor_ptr_->at(i).convertTo(fast_dist_map_tensor_[i], cv::DataType<ushort>::type);
#endif
}

void DirectionalChamferMatching::setupExhaustiveMatching( int max_template_pts )
{
  setupExhaustiveMatching_( *model_ptr_, num_directions_, eta_direction_,
                            templates_pts_, templates_i_dir_, templates_bb_, 1, max_template_pts );
}

void DirectionalChamferMatching::match(int num_best_matches, vector< TemplateMatch >& matches, int image_step)
{
  const int n_threads = ( parallelism_enabled_ ? omp_get_max_threads() : 1 );

  vector< multimap< double, TemplateMatch > > best_matches(n_threads);
  vector< float > min_dists( n_threads, std::numeric_limits< float >::max());
  const int w = cam_model_.imgWidth(), h = cam_model_.imgHeight();

#if CHAMFER_MATCHING_FAST_MAP && defined(D2CO_USE_SSE)
  
  // TODO Move from here to top
  
//   cv_ext::BasicTimer timer;
  
  int step_tensor_w = cvCeil(float(w)/image_step), step_tensor_h = h;
  vector<ImageTensor> step_tensor(image_step);
  for( auto &tensor : step_tensor )
    tensor.create (step_tensor_h, step_tensor_w, num_directions_, cv::DataType<ushort>::type, MEM_ALIGN_16 );

  // TODO Use vectorization
  for( int i = 0; i < num_directions_; i++ )
  {
    Mat &orig_mat_tensor = fast_dist_map_tensor_.at(i);

    for( int y = 0; y < h; y++)
    {
      ushort *orig_p = orig_mat_tensor.ptr<ushort>(y);
      vector <ushort *> step_p(image_step);
      for( int j = 0; j < image_step; j++ )
        step_p[j] = step_tensor[j][i].ptr<ushort>(y);
      
      for( int x = 0; x < w; x++, orig_p++)
        *step_p[x%image_step]++ = *orig_p;
    }
  }
  
//   cout<<"OBJ_REC_FAST_SSE data creation Elapsed time ms : "<<timer.elapsedTimeMs()<<endl;  

#endif

  #pragma omp parallel for schedule(dynamic) if( parallelism_enabled_ )
  for( int templ_i = 0; templ_i < int(templates_pts_.size()); templ_i++ )
  {
    int th_id = omp_get_thread_num();
    auto &th_best_matches = best_matches[th_id];
    float &min_dist = min_dists[th_id];
    
    vector<Point> proj_pts = templates_pts_[templ_i];
    const vector<int> &i_dirs = templates_i_dir_[templ_i];

    // TODO Remove this??
    if( image_step < 1 )
    {
      float dist = templateDist(proj_pts, i_dirs);

      if( int(th_best_matches.size()) < num_best_matches )
      {
        Eigen::Quaterniond r_quat;
        Eigen::Vector3d t_vec;
        model_ptr_->modelView(templ_i,r_quat,t_vec);
        th_best_matches.insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist,r_quat,t_vec, Point(0,0))) );
        min_dist = th_best_matches.rbegin()->first;
      }
      else if( dist < min_dist )
      {
        Eigen::Quaterniond r_quat;
        Eigen::Vector3d t_vec;
        model_ptr_->modelView(templ_i,r_quat,t_vec);
        th_best_matches.insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist,r_quat,t_vec, Point(0,0))) );
        th_best_matches.erase(--th_best_matches.rbegin().base());
        min_dist = th_best_matches.rbegin()->first;
      }
    }
    else
    {
      Rect &bb = templates_bb_[templ_i];
      Point tl = bb.tl();
      for( int i_p = 0; i_p < int(proj_pts.size()); i_p++ )
        proj_pts[i_p] -= tl;

      const int dists_w = (w - bb.width)/image_step + 1, dists_h = (h - bb.height)/image_step + 1;      
      Mat_<float> avg_dists(dists_h, dists_w);
      
#if CHAMFER_MATCHING_FAST_MAP
      
#if defined(D2CO_USE_AVX)
      
      cv_ext::AlignedMat c_dists(dists_h, dists_w, DataType<ushort>::type, cv::Scalar(0), cv_ext::MEM_ALIGN_32);
      
      __m256i cur_dist, acc_dist, *c_dists_p, *dist_map_p;
      int vec_c_dists_w = c_dists.step1()/16;
      for( int i_p = 0; i_p < int(proj_pts.size()); i_p++ )
      {
        int x = proj_pts[i_p].x;
        Mat &dist_map = step_tensor[x%image_step][i_dirs[i_p]];

        for( int y = proj_pts[i_p].y, c_dists_y = 0; c_dists_y < dists_h; y += image_step, c_dists_y++ )
        {
          c_dists_p = (__m256i *)c_dists.ptr<ushort>(c_dists_y);
          dist_map_p = (__m256i *)(dist_map.ptr<ushort>(y) + x/image_step);  

          for(int vec_c_dists_x = 0; vec_c_dists_x < vec_c_dists_w; dist_map_p++, c_dists_p++, vec_c_dists_x++ )
          {
            cur_dist = _mm256_loadu_si256(dist_map_p);
            acc_dist = _mm256_load_si256(c_dists_p);
            acc_dist = _mm256_adds_epu16(acc_dist, cur_dist);
            _mm256_store_si256(c_dists_p, acc_dist );
          }
        }
      }
      
#elif defined(D2CO_USE_SSE)


      cv_ext::AlignedMat c_dists(dists_h, dists_w, DataType<ushort>::type, cv::Scalar(0), cv_ext::MEM_ALIGN_16);
      
      __m128i cur_dist, acc_dist, *c_dists_p, *dist_map_p;
      int vec_c_dists_w = c_dists.step1()/8;
      for( int i_p = 0; i_p < int(proj_pts.size()); i_p++ )
      {
        int x = proj_pts[i_p].x;
        Mat &dist_map = step_tensor[x%image_step][i_dirs[i_p]];

        for( int y = proj_pts[i_p].y, c_dists_y = 0; c_dists_y < dists_h; y += image_step, c_dists_y++ )
        {
          c_dists_p = (__m128i *)c_dists.ptr<ushort>(c_dists_y);
          dist_map_p = (__m128i *)(dist_map.ptr<ushort>(y) + x/image_step);  

          for(int vec_c_dists_x = 0; vec_c_dists_x < vec_c_dists_w; dist_map_p++, c_dists_p++, vec_c_dists_x++ )
          {
            cur_dist = _mm_loadu_si128(dist_map_p);
            acc_dist = _mm_load_si128(c_dists_p);
            acc_dist = _mm_adds_epu16(acc_dist, cur_dist);
            _mm_store_si128(c_dists_p, acc_dist );
          }
        }
      }

#elif defined(D2CO_USE_NEON)

      cv_ext::AlignedMat c_dists(dists_h, dists_w, DataType<ushort>::type, cv::Scalar(0), cv_ext::MEM_ALIGN_16);
      
      uint16x8_t cur_dist, acc_dist;
      uint16_t* c_dists_p, *dist_map_p;
      int vec_c_dists_w = c_dists.step1()/8;
      for( int i_p = 0; i_p < int(proj_pts.size()); i_p++ )
      {
        int x = proj_pts[i_p].x;
        Mat &dist_map = step_tensor[x%image_step][i_dirs[i_p]];

        for( int y = proj_pts[i_p].y, c_dists_y = 0; c_dists_y < dists_h; y += image_step, c_dists_y++ )
        {
          c_dists_p = (uint16_t *)c_dists.ptr<ushort>(c_dists_y);
          dist_map_p = (uint16_t *)(dist_map.ptr<ushort>(y) + x/image_step);  

          for(int vec_c_dists_x = 0; vec_c_dists_x < vec_c_dists_w; dist_map_p++, c_dists_p++, vec_c_dists_x++ )
          {
//             vld1q_u16(__transfersize(8) uint16_t const * ptr);
//             cur_dist = _mm_loadu_si128(dist_map_p);
//             acc_dist = _mm_load_si128(c_dists_p);
            cur_dist = vld1q_u16(dist_map_p); // Unaligned!!
            acc_dist = vld1q_u16(c_dists_p);
            acc_dist = vaddq_s16(acc_dist, cur_dist);
            vst1q_u16(c_dists_p, acc_dist );
          }
        }
      }
      
#else // defined(D2CO_USE_SSE)
      // TODO Check here
      Mat c_dists( dists_h, dists_w, DataType<ushort>::type, Scalar(0));
      for( int i_p = 0; i_p < proj_pts.size(); i_p++ )
      {
        Mat &dist_map = fast_dist_map_tensor_[i_dirs[i_p]];

        for( int y = proj_pts[i_p].y, c_dists_y = 0; c_dists_y < dists_h; y += image_step, c_dists_y++ )
        {
          ushort *dists_p = c_dists.ptr<ushort>(c_dists_y);
          ushort *dist_map_p = dist_map.ptr<ushort>(y) + proj_pts[i_p].x;

          for(int c_dists_x = 0; c_dists_x < dists_w; dist_map_p += image_step, dists_p++, c_dists_x++ )
            *dists_p += *dist_map_p;
        }
      }
#endif

      float norm_factor = 1.0f/proj_pts.size();
      c_dists.convertTo(avg_dists, cv::DataType< float >::type, norm_factor);
      
#else // CHAMFER_MATCHING_FAST_MAP

      // TODO Check here
      avg_dists.setTo(Scalar(0));
      for( int i_p = 0; i_p < proj_pts.size(); i_p++ )
      {
        Mat &dist_map = (*dist_map_tensor_ptr_)[i_dirs[i_p]];

        // TODO center the searched area
        for( int y = proj_pts[i_p].y, avg_dists_y = 0; avg_dists_y < dists_h; y += image_step, avg_dists_y++ )
        {
          float *dists_p = avg_dists.ptr<float>(avg_dists_y);
          float *dist_map_p = dist_map.ptr<float>(y) + proj_pts[i_p].x;

          for(int avg_dists_x = 0; avg_dists_x < dists_w; dist_map_p += image_step, dists_p++, avg_dists_x++ )
            *dists_p += *dist_map_p;
        }
      }

      avg_dists *= 1.0f/proj_pts.size();
      
#endif
      
      for(int avg_dists_y = 0; avg_dists_y < dists_h; avg_dists_y++ )
      {
        float *dists_p = avg_dists.ptr<float>(avg_dists_y);
        for(int avg_dists_x = 0; avg_dists_x < dists_w; avg_dists_x++, dists_p++ )
        {
          float dist = *dists_p;
          if( int(th_best_matches.size()) < num_best_matches )
          {
            Eigen::Quaterniond r_quat;
            Eigen::Vector3d t_vec;
            model_ptr_->modelView(templ_i,r_quat,t_vec);
            Point offset( avg_dists_x*image_step, avg_dists_y*image_step);
            offset -= bb.tl();

            th_best_matches.insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist,r_quat,t_vec, offset)) );
            min_dist = th_best_matches.rbegin()->first;
          }
          else if( dist < min_dist )
          {
            Eigen::Quaterniond r_quat;
            Eigen::Vector3d t_vec;
            model_ptr_->modelView(templ_i,r_quat,t_vec);
            Point offset( avg_dists_x*image_step, avg_dists_y*image_step);
            offset -= bb.tl();
            th_best_matches.insert ( std::pair<double,TemplateMatch>(dist,TemplateMatch(templ_i, dist,r_quat,t_vec, offset)) );
            th_best_matches.erase(--th_best_matches.rbegin().base());
            min_dist = th_best_matches.rbegin()->first;
          }
        }
      }      
    }
  }
  
  for( int i = 1; i < n_threads; i++ )
    best_matches[0].insert(best_matches[i].begin(), best_matches[i].end());

  matches.clear();
  matches.reserve(num_best_matches);
  int num_matches = 0;
  for( auto iter = best_matches[0].begin();
       iter != best_matches[0].end() && num_matches < num_best_matches;
       iter++, num_matches++ )
  {
    normalizeMatch(iter->second);
    matches.push_back(iter->second);
  }
  
  
}

inline float DirectionalChamferMatching::templateDist( const std::vector<cv::Point> &proj_pts,
                                                       const std::vector<int> &i_dirs )
{
  double avg_dist = 0;
#if CHAMFER_MATCHING_FAST_MAP
  for( int i = 0; i < int(proj_pts.size()); i++ )  
    avg_dist += fast_dist_map_tensor_[i_dirs[i]].at<ushort>( proj_pts[i].y, proj_pts[i].x );
#else
  ImageTensor &dist_map_tensor = *dist_map_tensor_ptr_;
  for( int i = 0; i < proj_pts.size(); i++ )  
    avg_dist += dist_map_tensor[i_dirs[i]].at<float>( proj_pts[i].y, proj_pts[i].x );
#endif

  if( proj_pts.size() )
    avg_dist /= proj_pts.size();
  else
    avg_dist = std::numeric_limits< float >::max();

  return float(avg_dist);
}


// TODO Remove this function
double DirectionalChamferMatching::getDistance(const std::vector< Point2f > &proj_pts,
                                               const std::vector< float > &normal_directions) const
{
  int n_pts = 0;
  double avg_dist = 0;

  ImageTensor &dist_map_tensor = *dist_map_tensor_ptr_;
  const int img_w = dist_map_tensor[0].cols, img_h = dist_map_tensor[0].rows;

  for( int i = 0; i < int(proj_pts.size()); i++ )
  {
    const Point2f &coord = proj_pts.at(i);

    // PinholeSceneProjector marca los puntos OCLUIDOS exactamente como
    // (-1,-1). La autooclusion es legitima (la cara trasera del objeto no se
    // ve), asi que esos puntos no entran en el promedio.
    if( coord.x == -1.0f && coord.y == -1.0f )
      continue;

    // Todo lo demas es una proyeccion real y cuenta en el denominador,
    // este o no dentro del encuadre.
    n_pts++;

    const int x = cvRound(coord.x), y = cvRound(coord.y);

    // FIX (dos defectos en el codigo original, que hacia solo `if(coord.x>=0)`):
    //
    //  1. LECTURA FUERA DE LIMITES. No habia cota superior para x ni ninguna
    //     comprobacion de y, y se leia dist_map_tensor[z].at<float>(y,x)
    //     directamente. cv::Mat::at no valida indices en release, asi que en
    //     cuanto la pose proyectaba el modelo fuera de la imagen se leia
    //     memoria ajena: comportamiento indefinido, y el valor corrupto es
    //     justo el que se usa como puntuacion de aceptacion.
    //
    //  2. LOS PUNTOS FUERA DE IMAGEN NO PENALIZABAN. Quedaban fuera del
    //     numerador Y del denominador, de modo que una pose que sacase casi
    //     todo el modelo del encuadre (o lo encogiese alejandolo) puntuaba
    //     como perfecta con los pocos puntos que quedaban sobre algun borde.
    //
    // Ahora esos puntos cuestan lo peor que hay en el tensor.
    if( x < 0 || y < 0 || x >= img_w || y >= img_h )
    {
      avg_dist += out_of_image_penalty_;
      continue;
    }

    float direction = normal_directions[i] + M_PI/2;
    if( direction >= M_PI/2 )
      direction -= M_PI;
    direction += M_PI/2;

    int z = cvRound(eta_direction_*direction);
    z %= num_directions_;

    avg_dist += dist_map_tensor[z].at<float>( y, x );
  }

  if( n_pts )
    avg_dist /= n_pts;
  else
    avg_dist = std::numeric_limits< float >::max();

  return avg_dist;
}

double DirectionalChamferMatching::avgDistance( int idx )
{
  std::vector<Point2f> proj_pts;
  std::vector<float> normal_directions;

  if( idx < 0 )
    model_ptr_->projectRasterPoints( proj_pts, normal_directions );
  else
    model_ptr_->projectRasterPoints( idx, proj_pts, normal_directions );

  return getDistance( proj_pts, normal_directions );
}


double DirectionalChamferMatching::optimize()
{
  if( !optimizer_ptr_ )
    return -1;

  ceres::Solver::Options options;
  //options.linear_solver_type = ceres::DENSE_SCHUR;
  options.max_num_iterations = optimizer_num_iterations_;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = verbouse_mode_;

  ceres::Solver::Summary summary;
  ceres::Solve ( options, &optimizer_ptr_->problem, &summary );

  return summary.final_cost;
}

void DirectionalChamferMatching::updateOptimizer( int idx )
{
  const std::vector<Point3f> &model_pts = ( idx < 0 )?model_ptr_->getPoints():model_ptr_->getPrecomputedPoints(idx);
  const std::vector<Point3f> &model_dpts = ( idx < 0 )?model_ptr_->getDPoints():model_ptr_->getPrecomputedDPoints(idx);

  optimizer_ptr_ = std::shared_ptr< Optimizer > ( new Optimizer () );

  for ( int i = 0; i < int(model_pts.size()); ++i )
  {
    ceres::CostFunction* cost_function =
      DirectionalChamferResidual::Create ( cam_model_, dist_map_tensor_ptr_, model_pts[i], model_dpts[i] );

    optimizer_ptr_->problem.AddResidualBlock ( cost_function, new ceres::HuberLoss(1.0), transf_.data() );
  }
}

class HybridDirectionalChamferMatching::Optimizer
{
public:
  Optimizer(){};
  ~Optimizer(){};
  ceres::Problem problem;
};

HybridDirectionalChamferMatching::HybridDirectionalChamferMatching () :
  max_icp_iterations_(5),
  selected_idx_(-1)
{
  setNumDirections(60);
}

void HybridDirectionalChamferMatching::setNumDirections(int n )
{
  num_directions_ = n;
  eta_direction_ = double(num_directions_)/M_PI;
}

void HybridDirectionalChamferMatching::setInput ( const ImageTensorPtr& dist_map_tensor_ptr,
                                                  const ImageTensorPtr& edgels_map_tensor_ptr )
{
  if( dist_map_tensor_ptr->depth() != num_directions_ ||
      dist_map_tensor_ptr->at(0).type() != DataType<float>::type || 
      dist_map_tensor_ptr->at(0).rows != cam_model_.imgHeight()|| 
      dist_map_tensor_ptr->at(0).cols  != cam_model_.imgWidth() ||
      edgels_map_tensor_ptr->depth() != num_directions_ ||
      edgels_map_tensor_ptr->at(0).type() != DataType<Point2f>::type || 
      edgels_map_tensor_ptr->at(0).rows != cam_model_.imgHeight()|| 
      edgels_map_tensor_ptr->at(0).cols  != cam_model_.imgWidth() )
    throw invalid_argument("Invalid input data");  
  
  dist_map_tensor_ptr_ = dist_map_tensor_ptr;
  edgels_map_tensor_ptr_ = edgels_map_tensor_ptr;
}

void HybridDirectionalChamferMatching::setupExhaustiveMatching( int max_template_pts )
{
  std::cerr<<"IMPLEMENT ME!!"<<std::endl;
//   exit(-1);
}

void HybridDirectionalChamferMatching::match(int num_best_matches, vector< TemplateMatch >& matches, int image_step)
{
  std::cerr<<"IMPLEMENT ME!!"<<std::endl;
}

double HybridDirectionalChamferMatching::avgDistance ( int idx )
{
  int n_pts = 0;
  double avg_dist = 0;

  std::vector<Point2f> proj_pts;
  std::vector<float> normal_directions;

  if( idx < 0 )
    model_ptr_->projectRasterPoints( proj_pts, normal_directions );
  else
    model_ptr_->projectRasterPoints( idx, proj_pts, normal_directions );

  ImageTensor &dist_map_tensor = *dist_map_tensor_ptr_;
  for( int i = 0; i < int(proj_pts.size()); i++ )
  {
    const Point2f &coord = proj_pts.at(i);
    if( coord.x >= 0)
    {
      n_pts++;

      float direction = normal_directions[i] + M_PI/2;
      if( direction >= M_PI/2 )
        direction -= M_PI;
      direction += M_PI/2;

      int x = cvRound(coord.x), y = cvRound(coord.y);
      int z = cvRound(eta_direction_*direction);
      z %= num_directions_;

      avg_dist += dist_map_tensor[z].at<float>( y, x );
    }
  }

  if( n_pts )
    avg_dist /= n_pts;
  else
    avg_dist = std::numeric_limits< float >::max();

  return avg_dist;
}

double HybridDirectionalChamferMatching::optimize()
{
  double final_cost = -1.0;
  ImageTensor &edgels_map_tensor = *edgels_map_tensor_ptr_;

  for( int icp_iter = 0; icp_iter < max_icp_iterations_; icp_iter++)
  {

    d2co_optimizer_ptr_ = std::shared_ptr< Optimizer > ( new Optimizer () );

    for ( int i = 0; i < int(model_pts_.size()); ++i )
    {
      ceres::CostFunction* cost_function =
        DirectionalChamferResidual::Create ( cam_model_, dist_map_tensor_ptr_, model_pts_[i], model_dpts_[i] );

      d2co_optimizer_ptr_->problem.AddResidualBlock ( cost_function, new ceres::HuberLoss(1.0), transf_.data() );
    }

    ceres::Solver::Options options;
    //options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = optimizer_num_iterations_;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = verbouse_mode_;

    ceres::Solver::Summary summary;
    ceres::Solve ( options, &d2co_optimizer_ptr_->problem, &summary );

    std::vector<Point2f> img_pts;
    std::vector<float> normal_directions;
    if( selected_idx_ < 0 )
      model_ptr_->projectRasterPoints( transf_.data(), transf_.block<3,1>(4,0).data(),
                                       img_pts, normal_directions );
    else
      model_ptr_->projectRasterPoints( selected_idx_, transf_.data(), transf_.block<3,1>(4,0).data(),
                                       img_pts, normal_directions );

    icp_optimizer_ptr_ = std::shared_ptr< Optimizer > ( new Optimizer () );

    int res_size = 0;
    for ( int i = 0; i < int(model_pts_.size()); i++  )
    {
      if( img_pts[i].x >= 0 )
      {

        float direction = normal_directions[i] + M_PI/2;
        if( direction >= M_PI/2 )
          direction -= M_PI;
        direction += M_PI/2;

        int x = cvRound(img_pts[i].x), y = cvRound(img_pts[i].y);
        int z = cvRound(eta_direction_*direction);
        z %= num_directions_;

        const Point2f &closest_edgel = edgels_map_tensor[z].at<Point2f>( y, x );

        ceres::CostFunction* cost_function =
          ICPChamferResidual::Create ( cam_model_, model_pts_[i], closest_edgel );

        icp_optimizer_ptr_->problem.AddResidualBlock ( cost_function, new ceres::HuberLoss(1.0), transf_.data() );
        res_size++;
      }
    }

    if( res_size >= 3 )
    {
      ceres::Solver::Options options;
      //options.linear_solver_type = ceres::DENSE_SCHUR;
      options.max_num_iterations = optimizer_num_iterations_;
      options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
      options.minimizer_progress_to_stdout = verbouse_mode_;

      ceres::Solver::Summary summary;
      ceres::Solve ( options, &icp_optimizer_ptr_->problem, &summary );
      final_cost = summary.final_cost;
    }
    else
      break;
  }
  return final_cost;
}

void HybridDirectionalChamferMatching::updateOptimizer ( int idx )
{
  selected_idx_ = idx;

  model_pts_ = ( idx < 0 )?model_ptr_->getPoints():model_ptr_->getPrecomputedPoints(idx);
  model_dpts_ = ( idx < 0 )?model_ptr_->getDPoints():model_ptr_->getPrecomputedDPoints(idx);

//   d2co_optimizer_ptr_ = std::shared_ptr< Optimizer > ( new Optimizer () );
//
//   for ( int i = 0; i < model_pts_.size(); ++i )
//   {
//     ceres::CostFunction* cost_function =
//       DirectionalChamferResidual::Create ( cam_model_, dist_map_tensor_ptr_, model_pts_[i], model_dpts_[i] );
//
//     d2co_optimizer_ptr_->problem.AddResidualBlock ( cost_function, new ceres::HuberLoss(1.0), transf_.data() );
//   }
}


class BidirectionalChamferMatching::Optimizer
{
public:
  Optimizer(){};
  ~Optimizer(){};
  ceres::Problem problem;
};

struct BidirectionalChamferResidual
{
  BidirectionalChamferResidual ( const cv_ext::PinholeCameraModel &cam_model,
                                 const ImageTensorPtr &x_dist_map_tensor_ptr,
                                 const ImageTensorPtr &y_dist_map_tensor_ptr,
                                 const Point3f &model_pt, const Point3f &model_dpt ) :
    cam_model_ ( cam_model ),
    x_dist_map_tensor_ptr_ ( x_dist_map_tensor_ptr ),
    y_dist_map_tensor_ptr_ ( y_dist_map_tensor_ptr ),
    x_dist_map_tensor_( *x_dist_map_tensor_ptr_ ),
    y_dist_map_tensor_( *y_dist_map_tensor_ptr_ )
  {
    model_pt_[0] = model_pt.x;
    model_pt_[1] = model_pt.y;
    model_pt_[2] = model_pt.z;

    model_dpt_[0] = model_dpt.x;
    model_dpt_[1] = model_dpt.y;
    model_dpt_[2] = model_dpt.z;

    eta_direction_ = double(x_dist_map_tensor_.depth())/M_PI;
  }

  template <typename _T>
  bool operator() ( const _T* const pos, _T* residuals ) const
  {

    _T model_pt[3] = { _T ( model_pt_[0] ), _T ( model_pt_[1] ), _T ( model_pt_[2] ) };
    _T model_dpt[3] = { _T ( model_dpt_[0] ), _T ( model_dpt_[1] ), _T ( model_dpt_[2] ) };
    _T proj_pt[2], proj_dpt[2];

    cam_model_.quatRTProject ( pos, pos + 4, model_pt, proj_pt );
    cam_model_.quatRTProject ( pos, pos + 4, model_dpt, proj_dpt );

   if ( proj_pt[0] < _T ( 1 ) ||  proj_pt[1] < _T ( 1 ) ||
        proj_pt[0] > _T ( cam_model_.imgWidth() - 2 ) ||
        proj_pt[1] > _T ( cam_model_.imgHeight() - 2 ) ||
        proj_dpt[0] < _T ( 1 ) ||  proj_dpt[1] < _T ( 1 ) ||
        proj_dpt[0] > _T ( cam_model_.imgWidth() - 2 ) ||
        proj_dpt[1] > _T ( cam_model_.imgHeight() - 2 ) )
    {
      residuals[0] = _T ( 0 );
      residuals[1] = _T ( 0 );
      return true;
    }

    _T diff[2] = { proj_dpt[0] - proj_pt[0], proj_dpt[1] - proj_pt[1] };
    _T direction;

    if( diff[0] != _T(0) )
      direction = atan( diff[1]/diff[0] );
    else
      direction = _T(-M_PI/2);

    _T z = _T(eta_direction_) * ( direction + _T(M_PI/2) );
    //residuals[0] = cv_ext::tensorGetPix<float, _T> ( dist_map_tensor_, proj_pt[0], proj_pt[1], z );
    residuals[0] = cv_ext::tensorbilinearInterp<float, _T> ( x_dist_map_tensor_, proj_pt[0], proj_pt[1], z );
    residuals[1] = cv_ext::tensorbilinearInterp<float, _T> ( y_dist_map_tensor_, proj_pt[0], proj_pt[1], z );

    return true;
  }

  static ceres::CostFunction* Create ( const cv_ext::PinholeCameraModel &cam_model,
                                       const ImageTensorPtr &x_dist_map_tensor_ptr,
                                       const ImageTensorPtr &y_dist_map_tensor_ptr,
                                       const Point3f &model_pt, const Point3f &model_dpt)
  {
    return ( new ceres::NumericDiffCostFunction<BidirectionalChamferResidual, ceres::CENTRAL, 2, 7 > (
               new BidirectionalChamferResidual( cam_model, x_dist_map_tensor_ptr, y_dist_map_tensor_ptr,
                                                 model_pt, model_dpt ) ) );
  }

  const cv_ext::PinholeCameraModel &cam_model_;
  const ImageTensorPtr x_dist_map_tensor_ptr_, y_dist_map_tensor_ptr_;
  const ImageTensor &x_dist_map_tensor_, y_dist_map_tensor_;
  double model_pt_[3], model_dpt_[3];
  double eta_direction_;
};

BidirectionalChamferMatching::BidirectionalChamferMatching()
{
  setNumDirections(60);
}

void BidirectionalChamferMatching::setNumDirections(int n )
{
  num_directions_ = n;
  eta_direction_ = double(num_directions_)/M_PI;
}

void BidirectionalChamferMatching ::setInput ( const ImageTensorPtr& x_dist_map_tensor_ptr,
                                               const ImageTensorPtr& y_dist_map_tensor_ptr )
{
  if( x_dist_map_tensor_ptr->depth() != num_directions_ ||
      x_dist_map_tensor_ptr->at(0).type() != DataType<float>::type || 
      x_dist_map_tensor_ptr->at(0).rows != cam_model_.imgHeight()|| 
      x_dist_map_tensor_ptr->at(0).cols  != cam_model_.imgWidth() ||
      y_dist_map_tensor_ptr->depth() != num_directions_ ||
      y_dist_map_tensor_ptr->at(0).type() != DataType<float>::type || 
      y_dist_map_tensor_ptr->at(0).rows != cam_model_.imgHeight()|| 
      y_dist_map_tensor_ptr->at(0).cols  != cam_model_.imgWidth() )
    throw invalid_argument("Invalid input data");  
  
  x_dist_map_tensor_ptr_ = x_dist_map_tensor_ptr;
  y_dist_map_tensor_ptr_ = y_dist_map_tensor_ptr;
}

void BidirectionalChamferMatching::setupExhaustiveMatching( int max_template_pts )
{
  std::cerr<<"IMPLEMENT ME!!"<<std::endl;
//   exit(-1);
}

void BidirectionalChamferMatching::match(int num_best_matches, vector< TemplateMatch >& matches, int image_step)
{
  std::cerr<<"IMPLEMENT ME!!"<<std::endl;
}

double BidirectionalChamferMatching ::avgDistance( int idx )
{
  std::cerr<<"IMPLEMENT ME!!"<<std::endl;
  return -1;
//   int n_pts = 0;
//   double avg_dist = 0;
//
//   std::vector<Point2f> proj_pts;
//   std::vector<float> normal_directions;
//
//   if( idx < 0 )
//     model_ptr_->projectRasterPoints( proj_pts, normal_directions );
//   else
//     model_ptr_->projectRasterPoints( idx, proj_pts, normal_directions );
//
//   std::vector<Mat > &dist_map_tensor = *dist_map_tensor_ptr_;
//   for( int i = 0; i < proj_pts.size(); i++ )
//   {
//     const Point2f &coord = proj_pts.at(i);
//     if( coord.x >= 0)
//     {
//       n_pts++;
//
//       float direction = normal_directions[i] + M_PI/2;
//       if( direction >= M_PI/2 )
//         direction -= M_PI;
//       direction += M_PI/2;
//
//       int x = cvRound(coord.x), y = cvRound(coord.y);
//       int z = cvRound(eta_direction_*direction);
//       z %= num_directions_;
//
//       avg_dist += dist_map_tensor[z].at<float>( y, x );
//     }
//   }
//
//   if( n_pts )
//     avg_dist /= n_pts;
//   else
//     avg_dist = std::numeric_limits< float >::max();
//
//   return avg_dist;
}


double BidirectionalChamferMatching ::optimize()
{
  if( !optimizer_ptr_ )
    return -1;

  ceres::Solver::Options options;
  //options.linear_solver_type = ceres::DENSE_SCHUR;
  options.max_num_iterations = optimizer_num_iterations_;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = verbouse_mode_;

  ceres::Solver::Summary summary;
  ceres::Solve ( options, &optimizer_ptr_->problem, &summary );

  return summary.final_cost;
}

void BidirectionalChamferMatching ::updateOptimizer( int idx )
{
  const std::vector<Point3f> &model_pts = ( idx < 0 )?model_ptr_->getPoints():model_ptr_->getPrecomputedPoints(idx);
  const std::vector<Point3f> &model_dpts = ( idx < 0 )?model_ptr_->getDPoints():model_ptr_->getPrecomputedDPoints(idx);

  optimizer_ptr_ = std::shared_ptr< Optimizer > ( new Optimizer () );

  for ( int i = 0; i < int(model_pts.size()); ++i )
  {
    ceres::CostFunction* cost_function =
      BidirectionalChamferResidual::Create ( cam_model_, x_dist_map_tensor_ptr_, y_dist_map_tensor_ptr_,
                                             model_pts[i], model_dpts[i] );

    optimizer_ptr_->problem.AddResidualBlock ( cost_function, new ceres::HuberLoss(1.0), transf_.data() );
  }
}

class ICPDirectionalChamferMatching::Optimizer
{
public:
  Optimizer(){};
  ~Optimizer(){};
  ceres::Problem problem;
};

ICPDirectionalChamferMatching::ICPDirectionalChamferMatching () :
  num_icp_iterations_(50),
  selected_idx_(-1)
{
  setNumDirections( 60 );
}

void ICPDirectionalChamferMatching::setNumDirections(int n )
{
  num_directions_ = n;
  eta_direction_ = double(num_directions_)/M_PI;
}

void ICPDirectionalChamferMatching::setInput ( const ImageTensorPtr& edgels_map_tensor_ptr )
{
  if( edgels_map_tensor_ptr->depth() != num_directions_ ||
      edgels_map_tensor_ptr->at(0).type() != DataType<Point2f>::type || 
      edgels_map_tensor_ptr->at(0).rows != cam_model_.imgHeight()|| 
      edgels_map_tensor_ptr->at(0).cols  != cam_model_.imgWidth() )
    throw invalid_argument("Invalid input data");
  edgels_map_tensor_ptr_ = edgels_map_tensor_ptr;
}

void ICPDirectionalChamferMatching::setupExhaustiveMatching( int max_template_pts )
{
  std::cerr<<"IMPLEMENT ME!!"<<std::endl;
//   exit(-1);
}

void ICPDirectionalChamferMatching::match(int num_best_matches, vector< TemplateMatch >& matches, int image_step)
{
  std::cerr<<"IMPLEMENT ME!!"<<std::endl;
}

double ICPDirectionalChamferMatching::avgDistance ( int idx )
{
  int n_pts = 0;
  double avg_dist = 0;

  std::vector<Point2f> proj_pts;
  std::vector<float> normal_directions;

  if( idx < 0 )
    model_ptr_->projectRasterPoints( proj_pts, normal_directions );
  else
    model_ptr_->projectRasterPoints( idx, proj_pts, normal_directions );

  ImageTensor &edgels_map_tensor = *edgels_map_tensor_ptr_;
  for( int i = 0; i < int(proj_pts.size()); i++ )
  {
    const Point2f &coord = proj_pts.at(i);
    if( coord.x >= 0)
    {
      n_pts++;

      float direction = normal_directions[i] + M_PI/2;
      if( direction >= M_PI/2 )
        direction -= M_PI;
      direction += M_PI/2;

      int x = cvRound(coord.x), y = cvRound(coord.y);
      int z = cvRound(eta_direction_*direction);
      z %= num_directions_;

      const Point2f &closest_edgel = edgels_map_tensor[z].at<Point2f>( y, x );
      avg_dist += sqrt( (closest_edgel.x - coord.x)*(closest_edgel.x - coord.x) +
                        (closest_edgel.y - coord.y)*(closest_edgel.y - coord.y) );

    }
  }

  if( n_pts )
    avg_dist /= n_pts;
  else
    avg_dist = std::numeric_limits< float >::max();

  return avg_dist;
}

void ICPDirectionalChamferMatching::updateOptimizer ( int idx )
{
  selected_idx_ = idx;
  model_pts_ = ( idx < 0 )?model_ptr_->getPoints():model_ptr_->getPrecomputedPoints(idx);
}

double ICPDirectionalChamferMatching::optimize()
{
  double final_cost = -1.0;
  ImageTensor &edgels_map_tensor = *edgels_map_tensor_ptr_;

  for( int icp_iter = 0; icp_iter < num_icp_iterations_; icp_iter++)
  {

    std::vector<Point2f> img_pts;
    std::vector<float> normal_directions;
    if( selected_idx_ < 0 )
      model_ptr_->projectRasterPoints( transf_.data(), transf_.block<3,1>(4,0).data(),
                                       img_pts, normal_directions );
    else
      model_ptr_->projectRasterPoints( selected_idx_, transf_.data(), transf_.block<3,1>(4,0).data(),
                                       img_pts, normal_directions );

    optimizer_ptr_ = std::shared_ptr< Optimizer > ( new Optimizer () );

    int res_size = 0;
    for ( int i = 0; i < int(model_pts_.size()); ++i )
    {
      if( img_pts[i].x >= 0 )
      {

        float direction = normal_directions[i] + M_PI/2;
        if( direction >= M_PI/2 )
          direction -= M_PI;
        direction += M_PI/2;

        int x = cvRound(img_pts[i].x), y = cvRound(img_pts[i].y);
        int z = cvRound(eta_direction_*direction);
        z %= num_directions_;

        const Point2f &closest_edgel = edgels_map_tensor[z].at<Point2f>( y, x );

        ceres::CostFunction* cost_function =
          ICPChamferResidual::Create ( cam_model_, model_pts_[i], closest_edgel );

        optimizer_ptr_->problem.AddResidualBlock ( cost_function, new ceres::HuberLoss(1.0), transf_.data() );
        res_size++;
      }
    }
    if( res_size >= 3 )
    {
      ceres::Solver::Options options;
      //options.linear_solver_type = ceres::DENSE_SCHUR;
      options.max_num_iterations = optimizer_num_iterations_;
      options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
      options.minimizer_progress_to_stdout = verbouse_mode_;

      ceres::Solver::Summary summary;
      ceres::Solve ( options, &optimizer_ptr_->problem, &summary );
      final_cost = summary.final_cost;
    }
  }
  return final_cost;
}

class MultiViewsDirectionalChamferMatching::Optimizer
{
public:
  Optimizer(){};
  ~Optimizer(){};
  ceres::Problem problem;
};

struct MultiViewsDirectionalChamferResidual
{
  MultiViewsDirectionalChamferResidual ( const cv_ext::PinholeCameraModel &cam_model,
                                const ImageTensorPtr &dist_map_tensor_ptr,
                                const Point3f &model_pt, const Point3f &model_dpt,
                                const Eigen::Quaterniond &view_q, const Eigen::Vector3d &view_t ) :
    cam_model_ ( cam_model ),
    dist_map_tensor_ptr_ ( dist_map_tensor_ptr ),
    dist_map_tensor_( *dist_map_tensor_ptr )
  {
    model_pt_[0] = model_pt.x;
    model_pt_[1] = model_pt.y;
    model_pt_[2] = model_pt.z;

    model_dpt_[0] = model_dpt.x;
    model_dpt_[1] = model_dpt.y;
    model_dpt_[2] = model_dpt.z;

    eta_direction_ = double(dist_map_tensor_.depth())/M_PI;

    view_[0]=view_q.w(); view_[1]=view_q.x(); view_[2]=view_q.y(); view_[3]=view_q.z();
    view_[4]=view_t(0); view_[5]=view_t(1); view_[6]=view_t(2);
  }

  template <typename _T>
  bool operator() ( const _T* const pos, _T* residuals ) const
  {

    _T model_pt[3] = { _T ( model_pt_[0] ), _T ( model_pt_[1] ), _T ( model_pt_[2] ) };
    _T model_dpt[3] = { _T ( model_dpt_[0] ), _T ( model_dpt_[1] ), _T ( model_dpt_[2] ) };
    _T proj_pt[2], proj_dpt[2];

    _T view[7]={_T(view_[0]), _T(view_[1]), _T(view_[2]), _T(view_[3]), _T(view_[4]), _T(view_[5]), _T(view_[6])};


    Eigen::Quaternion<_T> pos_q; cv_ext::quat2EigenQuat(pos, pos_q); pos_q.normalize();
    Eigen::Matrix<_T,3,1> pos_t; pos_t<<pos[4], pos[5], pos[6];
    Eigen::Quaternion<_T> view_q; cv_ext::quat2EigenQuat(view, view_q); //view_q.normalize();
    Eigen::Matrix<_T,3,1> view_t; view_t<<view[4], view[5], view[6];

    Eigen::Transform< _T,3, Eigen::Affine > pos_T, T, view_T;
    pos_T.linear()=Eigen::Matrix<_T, 3,3>(pos_q); pos_T.translation()=pos_t;
    view_T.linear()=Eigen::Matrix<_T, 3,3>(view_q); view_T.translation()=view_t;

    T=(view_T*pos_T);
    Eigen::Matrix<_T,3,1> transf_p; transf_p<<model_pt[0], model_pt[1], model_pt[2]; transf_p=T*transf_p;
    Eigen::Matrix<_T,3,1> transf_dp; transf_dp<<model_dpt[0], model_dpt[1], model_dpt[2]; transf_dp=T*transf_dp;


    _T transf_pt[3] = { transf_p(0) ,transf_p(1) ,transf_p(2)  };
    _T transf_dpt[3] = { transf_dp(0) , transf_dp(1) , transf_dp(2)  };
    cam_model_.project(transf_pt, proj_pt);
    cam_model_.project(transf_dpt, proj_dpt);

    if (proj_pt[0] < _T ( 1 ) ||  proj_pt[1] < _T ( 1 ) ||
        proj_pt[0] > _T ( cam_model_.imgWidth() - 2 ) ||
        proj_pt[1] > _T ( cam_model_.imgHeight() - 2 ) ||
        proj_dpt[0] < _T ( 1 ) ||  proj_dpt[1] < _T ( 1 ) ||
        proj_dpt[0] > _T ( cam_model_.imgWidth() - 2 ) ||
        proj_dpt[1] > _T ( cam_model_.imgHeight() - 2 ) )
    {
      residuals[0] = _T ( 0 );
      return true;
    }
    
    _T diff[2] = { proj_dpt[0] - proj_pt[0], proj_dpt[1] - proj_pt[1] };
    _T direction;

    if( diff[0] != _T(0) )
      direction = atan( diff[1]/diff[0] );
    else
      direction = _T(-M_PI/2);

    _T z = _T(eta_direction_) * ( direction + _T(M_PI/2) );
    //residuals[0] = cv_ext::tensorGetPix<float, _T> ( dist_map_tensor_, proj_pt[0], proj_pt[1], z );
    residuals[0] = cv_ext::tensorbilinearInterp<float, _T> ( dist_map_tensor_, proj_pt[0], proj_pt[1], z );


    return true;
  }

  static ceres::CostFunction* Create ( const cv_ext::PinholeCameraModel &cam_model,
                                       const ImageTensorPtr &dist_map_tensor_ptr,
                                       const Point3f &model_pt, const Point3f &model_dpt,
                                       const Eigen::Quaterniond &view_q, const Eigen::Vector3d &view_t)
  {
    return ( new ceres::NumericDiffCostFunction<MultiViewsDirectionalChamferResidual, ceres::CENTRAL, 1, 7 > (
               new MultiViewsDirectionalChamferResidual( cam_model, dist_map_tensor_ptr, model_pt, model_dpt, view_q, view_t ) ) );
  }

  const cv_ext::PinholeCameraModel &cam_model_;
  const ImageTensorPtr dist_map_tensor_ptr_;
  const ImageTensor & dist_map_tensor_;
  double model_pt_[3], model_dpt_[3];
  double eta_direction_;
  double view_[7];
};


MultiViewsDirectionalChamferMatching::MultiViewsDirectionalChamferMatching()
{
  setNumDirections(60);
}

void MultiViewsDirectionalChamferMatching::setNumDirections(int n )
{
  num_directions_ = n;
  eta_direction_ = double(num_directions_)/M_PI;
}

void MultiViewsDirectionalChamferMatching::setInput(const MultiViewsInputVec& input)
{
  input_=input;
}


double MultiViewsDirectionalChamferMatching::avgDistance( int idx )
{
  return -1;
//   int n_pts = 0;
//   double avg_dist = 0;
//
//   std::vector<Point2f> proj_pts;
//   std::vector<float> normal_directions;
//
//   if( idx < 0 )
//     model_ptr_->projectRasterPoints( proj_pts, normal_directions );
//   else
//     model_ptr_->projectRasterPoints( idx, proj_pts, normal_directions );
//
//   std::vector<Mat > &dist_map_tensor = *dist_map_tensor_ptr_;
//   for( int i = 0; i < proj_pts.size(); i++ )
//   {
//     const Point2f &coord = proj_pts.at(i);
//     if( coord.x >= 0)
//     {
//       n_pts++;
//
//       float direction = normal_directions[i] + M_PI/2;
//       if( direction >= M_PI/2 )
//         direction -= M_PI;
//       direction += M_PI/2;
//
//       int x = cvRound(coord.x), y = cvRound(coord.y);
//       int z = cvRound(eta_direction_*direction);
//       z %= num_directions_;
//
//       avg_dist += dist_map_tensor[z].at<float>( y, x );
//     }
//   }
//
//   if( n_pts )
//     avg_dist /= n_pts;
//   else
//     avg_dist = std::numeric_limits< float >::max();
//
//   return avg_dist;
}


double MultiViewsDirectionalChamferMatching::optimize()
{
  if( !optimizer_ptr_ )
    return -1;

  ceres::Solver::Options options;
  //options.linear_solver_type = ceres::DENSE_SCHUR;
  options.max_num_iterations = optimizer_num_iterations_;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = verbouse_mode_;

  ceres::Solver::Summary summary;
  ceres::Solve ( options, &optimizer_ptr_->problem, &summary );

  return summary.final_cost;
}
  
void MultiViewsDirectionalChamferMatching::updateOptimizer( int idx )
{

  optimizer_ptr_ = std::shared_ptr< Optimizer > ( new Optimizer () );
  
  const Eigen::Quaterniond r_quat = Eigen::Quaternion< double, Eigen::DontAlign >(transf_.data());
  const Eigen::Vector3d t_vec = Eigen::Map<const Eigen::Vector3d>(transf_.block<3,1>(4,0).data());
  Eigen::Matrix3d R(r_quat);
  Eigen::Affine3d RT; RT.linear()=R; RT.translation()=t_vec;
  for(size_t i=0; i<input_.size(); i++)
  {
    assert(input_[i].views.size()==input_[i].dist_map_tensor_ptr_vec.size());
    for(size_t j=0; j<input_[i].views.size(); j++)
    {     
      Eigen::Affine3d RT_transf=input_[i].views[j]*RT;
      Eigen::Quaterniond r(RT_transf.linear()); r.normalize();
      Eigen::Vector3d t=RT_transf.translation();
      input_[i].model_ptr->setModelView(r, t);
      
      const std::vector<Point3f> &model_pts = ( idx < 0 )?input_[i].model_ptr->getPoints()
                                                         :input_[i].model_ptr->getPrecomputedPoints(idx);
      const std::vector<Point3f> &model_dpts = ( idx < 0 )?input_[i].model_ptr->getDPoints()
                                                          :input_[i].model_ptr->getPrecomputedDPoints(idx);
        
      Eigen::Affine3d view_inv=input_[i].views[j];
      Eigen::Quaterniond q(view_inv.linear()); q.normalize();

      for ( size_t k = 0; k < model_pts.size(); ++k )
      {
        ceres::CostFunction* cost_function =
          MultiViewsDirectionalChamferResidual::Create ( input_[i].camera_model, input_[i].dist_map_tensor_ptr_vec[j],
                                                model_pts[k], model_dpts[k], q, view_inv.translation() );
        optimizer_ptr_->problem.AddResidualBlock ( cost_function, new ceres::HuberLoss(1.0), transf_.data() );
      }
    }
  }
}
