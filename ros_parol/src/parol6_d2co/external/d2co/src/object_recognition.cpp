#include <sstream>
#include <omp.h>
#include "boost/shared_ptr.hpp"

#include "object_recognition.h"

// cv::Mat dbg_input_img;
// 
// static const double OUTER_POINT_SCORE = 0.6;


// ObjectRecognition::ObjectRecognition() :
//  plane_normal_(0,0,1),
//  plane_dist_(0.5)
// {}
// 
// void ObjectRecognition::setCamModel ( cv_ext::PinholeCameraModel& cam_model )
// {
//   cam_model_ = cam_model;
// }
// 
// void ObjectRecognition::setModel( std::string model_name, bool precompute_views, double step_model, 
//                                   double step_xy )
// {
//   Model new_model;
//   new_model.obj_model_ptr = RasterObjectModel3DPtr( new RasterObjectModel3D() );
//   new_model.obj_model_ptr->setCamModel( cam_model_ );
//   new_model.obj_model_ptr->setStepMeters ( step_model ); //meters
//   //obj_model_ptr->setCentroidOrigOffset();
//   new_model.obj_model_ptr->setBBCenterOrigOffset();
//   
//   std::stringstream stl_file;
//   stl_file <<model_name<<".stl";
//   if ( !new_model.obj_model_ptr->setModelFile ( stl_file.str() ) )
//     return;
//   
//   //obj_model_ptr->setMinSegmentsLen(0.01);
//   new_model.obj_model_ptr->computeRaster();
//   models_[model_name] = new_model;
//   
//   if( precompute_views )
//   {
//     precomputeModelViews( new_model.obj_model_ptr, quat_rotations_, plane_q_rot_, plane_center_, step_xy ); 
//     new_model.obj_model_ptr->savePrecomputedModelsViews(model_name);
//   }
// }
// 
// void ObjectRecognition::clearModels()
// {
//   models_.clear();
//   current_model_.empty();
// }
// 
// void ObjectRecognition::setSearchingPlane ( Eigen::Vector3d plane_normal, double plane_dist )
// {
//   quat_rotations_.clear();
//   plane_normal /= plane_normal.norm();
//   
//   double normalized_pt[] = {0.0, 0.0};
//   Eigen::Vector3d plane_x, plane_y;
// 
//   plane_center_(0) = 0;
//   plane_center_(1) = 0;
//   plane_center_(2) = plane_dist/ plane_normal(2);
//   
//   normalized_pt[0] = 1.0;
//   
//   plane_x(2) = plane_dist/( plane_normal(0)*normalized_pt[0] + plane_normal(1)*normalized_pt[1] + plane_normal(2) );
//   plane_x(0) = normalized_pt[0] * plane_x(2);
//   plane_x(1) = normalized_pt[1] * plane_x(2);
//   
//   plane_x -= plane_center_;
//   plane_x /= plane_x.norm();
//   plane_y = plane_normal.cross(plane_x);
//   
//   Eigen::Matrix3d plane_rotation;
//   plane_rotation << plane_x(0), plane_y(0), plane_normal(0),
//                     plane_x(1), plane_y(1), plane_normal(1),
//                     plane_x(2), plane_y(2), plane_normal(2);
//   
//   plane_q_rot_ = plane_rotation;
//   
//   std::vector <cv::Point3f> orientations_pts, orientations_pts0, orientations_pts1, orientations_pts2;
//   std::vector< Eigen::Quaterniond > quat_rotations0, quat_rotations1, quat_rotations2;
//  
//   // WARNING HACK
//   orientations_pts0.push_back(cv::Point3f(plane_normal(0),plane_normal(1),plane_normal(2)));
//   orientations_pts0.push_back(cv::Point3f(-plane_normal(0),-plane_normal(1),-plane_normal(2)));
//   orientations_pts1.push_back(cv::Point3f(-plane_normal(0),plane_normal(2),plane_normal(1)));
//   orientations_pts1.push_back(cv::Point3f(plane_normal(0),-plane_normal(2),-plane_normal(1)));
//   // WARNING BUG CHECK HERE!!!
//   orientations_pts2.push_back(cv::Point3f(plane_normal(0),-plane_normal(1),-plane_normal(2)));
//   orientations_pts2.push_back(cv::Point3f(-plane_normal(0),plane_normal(1),plane_normal(2)));
//   
//   cv_ext::sampleRotationsAroundVectors( orientations_pts0, quat_rotations0, 18, cv_ext::COORDINATE_Z_AXIS );
//   cv_ext::sampleRotationsAroundVectors( orientations_pts1, quat_rotations1, 18, cv_ext::COORDINATE_Y_AXIS );
//   cv_ext::sampleRotationsAroundVectors( orientations_pts2, quat_rotations2, 18, cv_ext::COORDINATE_X_AXIS );
//   
//   quat_rotations_.insert(quat_rotations_.end(), quat_rotations0.begin(), quat_rotations0.end());
//   quat_rotations_.insert(quat_rotations_.end(), quat_rotations1.begin(), quat_rotations1.end());
//   quat_rotations_.insert(quat_rotations_.end(), quat_rotations2.begin(), quat_rotations2.end());
// }
// 
// 
// void ObjectRecognition::getObjectMask( const cv::Mat &src_img, cv::Mat &obj_mask )
// {
//   cv::Mat tmp_img, img;
//   cv::pyrDown(src_img, img);
//   //cv::pyrDown(tmp_img, img);
//   cv::pyrDown(img, tmp_img);
//   img = tmp_img;
//   
//   cv::Mat obj_mask_h, obj_mask_l, res_obj_mask;
//   cv::threshold(img, obj_mask_h, 180, 255, cv::THRESH_BINARY );
//   cv::threshold(img, obj_mask_l, 10, 255, cv::THRESH_BINARY_INV );
//   
//   //cv::adaptiveThreshold(img, obj_mask, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 3, 3);  
//   
//   res_obj_mask = obj_mask_h | obj_mask_l;
//   int dilation_size = 4;
//   cv::Mat kernel = cv::getStructuringElement( cv::MORPH_RECT,
//                                               cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
//                                               cv::Point( dilation_size, dilation_size ) );
//   cv::dilate(res_obj_mask, res_obj_mask, kernel );
//   cv::resize( res_obj_mask, obj_mask, src_img.size(), 0, 0, cv::INTER_LINEAR );
// }
// 
// void ObjectRecognition::checkInMask(cv::Mat &quant, int r, int c, int T, int q, cv::Mat &bin)
// {
//   bool isPresent=false;
//   for (int i=-T/2; i<=T/2; i++)
//   {
//     for (int j=-T/2; j<=T/2; j++)
//     {
//       if(quant.at<float>(r+i,c+j)==(float)q)
//       {
//         isPresent=true;
//         break;
//       } 
//     }
//   }
//   if(isPresent)
//   {
//     for (int i=-T/2; i<=T/2; i++)
//     {
//       for (int j=-T/2; j<=T/2; j++)
//       {
//         bin.at<char>(r+i,c+j)=bin.at<char>(r+i,c+j)|(1<<q);
//       }
//     }
//   }
// }
// 
// int ObjectRecognition::getQuantizedOrientation(float dir, int n0)
// {
//   if (dir<0) dir+=M_PI;
//   //else if(dir>M_PI) dir -= M_PI;
//   float alfa=M_PI/n0;
//   for (int i=0; i<n0; i++){
//     if (dir<=alfa*(i+1)) 
//     {
//       return i;
//     }
//   }
// }
// 
// void ObjectRecognition::computeBinaryImage( const cv::Mat gr_dir, const cv::Mat gr_mag, cv::Mat& binary)
// {
//   int n0=8;
//   int T=8;
//   cv::Mat quant_gr(gr_dir.size(), CV_32F);
//   
//   cv::Mat gr_mag_thresh;
//   gr_mag.convertTo(gr_mag_thresh, CV_8UC1,255);
//   
//   cv::adaptiveThreshold(gr_mag_thresh, gr_mag_thresh, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 11,-8);
//   
//   for (int r=0; r<gr_dir.rows; r++)
//   {
//     for (int c=0; c<gr_dir.cols; c++)
//     { 
//       if (gr_mag_thresh.at<unsigned char>(r,c)==255) 
//         quant_gr.at<float>(r,c)=(float)getQuantizedOrientation(gr_dir.at<float>(r,c)+M_PI/2, n0);
//       else
//         quant_gr.at<float>(r,c)=-1;
//     }
//   }
//   for (int r=0+T/2; r<quant_gr.rows-T/2; r++)
//   {
//     for (int c=0+T/2; c<quant_gr.cols-T/2; c++)
//     {
//       if (quant_gr.at<float>(r,c)>=0)
//       {
//         for(int i=0; i<n0; i++)
//         {
//           checkInMask(quant_gr,r,c,T, i,binary);
//         }
//       }
//     }
//   }
//   //cv_ext::showImage(gr_mag_thresh, "gr_mag_thresh" ,true ,20 );
//   //cv_ext::showImage(quant_gr, "quantized" ,true ,20 );
//   //cv_ext::showImage(binary, "binary");
//   
// }
// 
// void ObjectRecognition::computeLookupTable( std::vector<float*> &lookup_tables )
// {
//   lookup_tables.clear();
//   int n0=8;
//   float alfa=M_PI/n0;
//   for(int q=0; q<n0; q++)
//   {
//     float* table = (float*) malloc (256*sizeof(float));
//     table[0]=0;
//     for(int index=1;index<256;index++)
//     {
//       float max_cos=0;
//       for (int i=0;i<n0;i++)
//       {
//         unsigned char qi=1<<i;
//         if(((unsigned char)index&qi)==qi)
//         {
//           float current_cos=fabs(cos(quantumToAngle(q,n0)-quantumToAngle(i,n0)));
//           if(current_cos>max_cos) max_cos=current_cos;
//         }
//       }
//       table[index]=max_cos;
//     }
//     lookup_tables.push_back(table);
//   }
// }
// 
// void ObjectRecognition::computeResponseMaps(cv::Mat &binary, std::vector<float*> &lookup_tables, std::vector<cv::Mat> &responseMaps)
// {
//   responseMaps.clear();
//   for (int i=0;i<lookup_tables.size();i++)
//   {
//     cv::Mat r_m = cv::Mat(binary.size(), CV_32F, cvScalar(0));
//     for(int r=0; r<binary.rows; r++){
//       for(int c=0; c<binary.cols; c++){
//         unsigned char index=binary.at<unsigned char>(r,c);
//         if ((int)index>0)
//           r_m.at<float>(r,c)=lookup_tables[i][(int)index];
//       }
//     }
//     responseMaps.push_back(r_m);
//    // cv_ext::showImage(responseMaps[i], "response map");
//   }
// }
// 
// float ObjectRecognition::quantumToAngle(int q, int n0)
// {
//   float alfa=M_PI/n0;
//   return alfa*q;
// }
// 
// void ObjectRecognition::extractIndicesFromLookupTables(RasterObjectModel3DPtr &obj_model_ptr,
//                                                        const cv::Mat &binary,
//                                                        const std::vector<float*> &lookup_tables,
//                                                        std::vector<int> &idxs)
// {
//   idxs.clear();
//   std::map<float,int> map_score_idx;
//   
//   cv_ext::BasicTimer timer;
//   
//   const int n_threads = omp_get_max_threads();
//   std::cout<<"Max num threads :"<<n_threads<<std::endl;
//   std::vector<std::pair<float,int> > idxs_v[n_threads];
//   #pragma omp parallel
//   {
//     int th_id = omp_get_thread_num();
//     std::vector<std::pair<float,int> > &local_idxs = idxs_v[th_id];
//     int precomputed_views = obj_model_ptr->numPrecomputedModelsViews();
// 
//     for( int idx = th_id; idx < precomputed_views; idx+=n_threads)
//     {
//       int view_idx = idx;
//     
//       Eigen::Quaterniond quat;
//       Eigen::Vector3d t;
//       obj_model_ptr->modelView(view_idx, quat, t);
//       
//       std::vector<cv::Point2f> proj_pts;
//       std::vector<float> normal_directions;
//       obj_model_ptr->projectRasterPoints(view_idx, quat, t, proj_pts, normal_directions );
//       float score=0;
//       for(int i=0; i<proj_pts.size(); i++)
//       {
//         int q=getQuantizedOrientation(normal_directions[i]+M_PI/2,8);
//         score+=lookup_tables[q][(int)binary.at<unsigned char>(proj_pts[i])];
//       }
//       score/=(float)proj_pts.size();
//       local_idxs.push_back(std::pair<float,int>(-score, view_idx));
//     }
//   }
//     
//   std::cout<<"extract indices from lookup tables : "<<timer.elapsedTimeMs() <<std::endl;
//     
//   for( int th_id = 0; th_id < n_threads; th_id++)
//   {
//     std::vector<std::pair<float,int> > &local_idxs = idxs_v[th_id];
//     for(int i = 0; i < local_idxs.size(); i++)
//     {
//       map_score_idx.insert ( local_idxs[i] );
//     }
//   }
//     
//   int i=0;
//   int last=100;
//   for( std::map<float,int>::iterator it = map_score_idx.begin(); it != map_score_idx.end()&&i<last; ++it, i++ ) {
//   	idxs.push_back( it->second );
//   }
// }
// 
// void ObjectRecognition::extractIndicesFromResponseMaps(RasterObjectModel3DPtr &obj_model_ptr,
//                                                        const std::vector<cv::Mat> &responseMaps,
//                                                        std::vector<int> &idxs)
// {
//   idxs.clear();
//   std::map<float,int> map_score_idx;
//   
//   cv_ext::BasicTimer timer;
//   
//   const int n_threads = omp_get_max_threads();
//   std::cout<<"Max num threads :"<<n_threads<<std::endl;
//   std::vector<std::pair<float,int> > idxs_v[n_threads];
//   #pragma omp parallel
//   {
//     int th_id = omp_get_thread_num();
//     std::vector<std::pair<float,int> > &local_idxs = idxs_v[th_id];
//     int precomputed_views = obj_model_ptr->numPrecomputedModelsViews();
// 
//     for( int idx = th_id; idx < precomputed_views; idx+=n_threads)
//     {
//       int view_idx = idx;
//     
//       Eigen::Quaterniond quat;
//       Eigen::Vector3d t;
//       obj_model_ptr->modelView(view_idx, quat, t);
//       
//       std::vector<cv::Point2f> proj_pts;
//       std::vector<float> normal_directions;
//       obj_model_ptr->projectRasterPoints(view_idx, quat, t, proj_pts, normal_directions );
//       float score=0;
//       for(int i=0; i<proj_pts.size(); i++)
//       {
//         int q=getQuantizedOrientation(normal_directions[i]+M_PI/2,8);
//         score+=responseMaps[q].at<float>(proj_pts[i]);
//       }
//       score/=(float)proj_pts.size();
//       local_idxs.push_back(std::pair<float,int>(-score, view_idx));
//     }
//   }
// 
//   std::cout<<"extract indices from responce maps : "<<timer.elapsedTimeMs() <<std::endl;
//     
//   for( int th_id = 0; th_id < n_threads; th_id++)
//   {
//     std::vector<std::pair<float,int> > &local_idxs = idxs_v[th_id];
//     for(int i = 0; i < local_idxs.size(); i++)
//     {
//       map_score_idx.insert ( local_idxs[i] );
//     }
//   }
//     
//   int i=0;
//   int last=100;
//   for( std::map<float,int>::iterator it = map_score_idx.begin(); it != map_score_idx.end()&&i<last; ++it, i++ ) {
//   	idxs.push_back( it->second );
//   }
// }
// 
// void ObjectRecognition::extractIndicesFromResponseMaps(RasterModelImagePtr &model_image_ptr,
//                                                        const std::vector<cv::Mat> &responseMaps,
//                                                        std::vector<int> &idxs)
// {
//   idxs.clear();
//   std::map<float,int> map_score_idx;
//   
//   cv_ext::BasicTimer timer;
//   
//   const int n_threads = omp_get_max_threads();
//   std::cout<<"Max num threads :"<<n_threads<<std::endl;
//   std::vector<std::pair<float,int> > idxs_v[n_threads];
//   #pragma omp parallel
//   {
//     int th_id = omp_get_thread_num();
//     std::vector<std::pair<float,int> > &local_idxs = idxs_v[th_id];
//     int precomputed_views = model_image_ptr->numPrecomputedModelsViews();
// 
//     for( int idx = th_id; idx < precomputed_views; idx+=n_threads)
//     {
//       int view_idx = idx;
//     
//       Eigen::Quaterniond quat;
//       Eigen::Vector3d t;
//       std::vector<cv::Point2f> proj_pts;
//       std::vector<unsigned char> q_normal_directions;
//       model_image_ptr->getModel(view_idx, quat, t, proj_pts, q_normal_directions );
//       //std::cout<<"sizes: "<<proj_pts.size()<<" "<<q_normal_directions.size()<<std::endl;
//       float score=0;
//       for(int i=0; i<proj_pts.size(); i++)
//       {
//         //std::cout<<"q_norm[i]: "<<(int)q_normal_directions[i]<<std::endl;
//         score+=responseMaps[(int)q_normal_directions[i]].at<float>(proj_pts[i]);
//       }
//       score/=(float)proj_pts.size();
//       local_idxs.push_back(std::pair<float,int>(-score, view_idx));
//     }
//   }
// 
//   std::cout<<"extract indices from responce maps : "<<timer.elapsedTimeMs() <<std::endl;
//     
//   for( int th_id = 0; th_id < n_threads; th_id++)
//   {
//     std::vector<std::pair<float,int> > &local_idxs = idxs_v[th_id];
//     for(int i = 0; i < local_idxs.size(); i++)
//     {
//       map_score_idx.insert ( local_idxs[i] );
//     }
//   }
//     
//   int i=0;
//   int last=100;
//   for( std::map<float,int>::iterator it = map_score_idx.begin(); it != map_score_idx.end()&&i<last; ++it, i++ ) {
//   	idxs.push_back( it->second );
//   }
// }
// 
// void ObjectRecognition::extractIndicesFromLookupTables(RasterModelImagePtr &model_image_ptr,
//                                                        const cv::Mat &binary,
//                                                        const std::vector<float*> &lookup_tables,
//                                                        std::vector<int> &idxs)
// {
//   idxs.clear();
//   std::map<float,int> map_score_idx;
//   
//   cv_ext::BasicTimer timer;
//   
//   const int n_threads = omp_get_max_threads();
//   std::cout<<"Max num threads :"<<n_threads<<std::endl;
//   std::vector<std::pair<float,int> > idxs_v[n_threads];
//   
//   std::cout<<"model_image_ptr->numPrecomputedModelsViews() "<<model_image_ptr->numPrecomputedModelsViews()<<std::endl;
//   #pragma omp parallel
//   {
//     int th_id = omp_get_thread_num();
//     std::vector<std::pair<float,int> > &local_idxs = idxs_v[th_id];
//     int precomputed_views = model_image_ptr->numPrecomputedModelsViews();
// 
//     for( int idx = th_id; idx < precomputed_views; idx+=n_threads)
//     {
//       int view_idx = idx;
//     
//       Eigen::Quaterniond quat;
//       Eigen::Vector3d t;
//       std::vector<cv::Point2f> proj_pts;
//       std::vector<unsigned char> q_normal_directions;
//       model_image_ptr->getModel(view_idx, quat, t, proj_pts, q_normal_directions );
//       float score=0;
//       for(int i=0; i<proj_pts.size(); i++)
//       {
//         score+=lookup_tables[(int)q_normal_directions[i]][(int)binary.at<unsigned char>(proj_pts[i])];
//       }
//       score/=(float)proj_pts.size();
//       local_idxs.push_back(std::pair<float,int>(-score, view_idx));
//     }
//   }
// 
//   std::cout<<"extract indices from lookup tables : "<<timer.elapsedTimeMs() <<std::endl;
//     
//   for( int th_id = 0; th_id < n_threads; th_id++)
//   {
//     std::vector<std::pair<float,int> > &local_idxs = idxs_v[th_id];
//     for(int i = 0; i < local_idxs.size(); i++)
//     {
//       map_score_idx.insert ( local_idxs[i] );
//     }
//   }
//     
//   int i=0;
//   int last=100;
//   for( std::map<float,int>::iterator it = map_score_idx.begin(); it != map_score_idx.end()&&i<last; ++it, i++ ) {
//   	idxs.push_back( it->second );
//   }
// }
// 
// void ObjectRecognition::extractIndicesFromTensor(RasterModelImagePtr &model_image_ptr,
//                                                    ImageTensorPtr &dist_map_tensor_ptr,
//                                                    std::vector<int> &idxs)
// {
//   idxs.clear();
//   std::map<float,int> map_score_idx;
//   
//   cv_ext::BasicTimer timer;
//   
//   const int n_threads = omp_get_max_threads();
//   std::cout<<"Max num threads :"<<n_threads<<std::endl;
//   std::vector<std::pair<float,int> > idxs_v[n_threads];
//   #pragma omp parallel
//   {
//     int th_id = omp_get_thread_num();
//     std::vector<std::pair<float,int> > &local_idxs = idxs_v[th_id];
//     int precomputed_views = model_image_ptr->numPrecomputedModelsViews();
//     
//     int num_directions = dist_map_tensor_ptr->size();
//     float eta_direction = (float)(num_directions)/M_PI;
// 
//     for( int idx = th_id; idx < precomputed_views; idx+=n_threads)
//     {
//       int view_idx = idx;
//     
//       Eigen::Quaterniond quat;
//       Eigen::Vector3d t;
//       std::vector<cv::Point2f> proj_pts;
//       std::vector<float> normal_directions;
//       model_image_ptr->getModel(view_idx, quat, t, proj_pts, normal_directions );
//       int n_pts = 0;
//       float avg_dist = 0;
//       std::vector<cv::Mat > &dist_map_tensor = *dist_map_tensor_ptr;
//       for(int i=0; i<proj_pts.size(); i++)
//       {
//         const cv::Point2f &coord = proj_pts.at(i);
//         if( coord.x >= 0)
//         {
//           n_pts++;
//           
//           float direction = normal_directions[i] + M_PI/2;
//           if( direction >= M_PI/2 )
//             direction -= M_PI;
//           direction += M_PI/2;
//                   
//           int x = round(coord.x), y = round(coord.y);
//           int z = round(eta_direction*direction);
//           z %= num_directions;
//           
//           avg_dist += dist_map_tensor[z].at<float>( y, x );
//         }
//       }
//       if( n_pts )
//         avg_dist /= n_pts;
//       else
//         avg_dist = std::numeric_limits< float >::max();
//         
//       local_idxs.push_back(std::pair<float,int>(avg_dist, view_idx));
//     }
//   }
// 
//   std::cout<<"extract indices from tensor : "<<timer.elapsedTimeMs() <<std::endl;
//   
//   for( int th_id = 0; th_id < n_threads; th_id++)
//   {
//     std::vector<std::pair<float,int> > &local_idxs = idxs_v[th_id];
//     for(int i = 0; i < local_idxs.size(); i++)
//     {
//       map_score_idx.insert ( local_idxs[i] );
//     }
//   }
//   int i=0;
//   int last=100;
//   for( std::map<float,int>::iterator it = map_score_idx.begin(); it != map_score_idx.end()&&i<last; ++it, i++ ) {
//   	idxs.push_back( it->second );
//   }
// }
// 
// void ObjectRecognition::extractIndicesFromTensor(RasterModelImagePtr &model_image_ptr,
//                                                    ImageTensorPtr &dist_map_tensor_ptr,
//                                                    std::vector<std::pair<float,int> > &idxs)
// {
// 
//   idxs.clear();
//   std::map<float,int> map_score_idx;
//   
//   cv_ext::BasicTimer timer;
//   
//   const int n_threads = omp_get_max_threads();
//   std::cout<<"Max num threads :"<<n_threads<<std::endl;
//   std::vector<std::pair<float,int> > idxs_v[n_threads];
//   #pragma omp parallel
//   {
//     int th_id = omp_get_thread_num();
//     std::vector<std::pair<float,int> > &local_idxs = idxs_v[th_id];
//     int precomputed_views = model_image_ptr->numPrecomputedModelsViews();
//     
//     int num_directions = dist_map_tensor_ptr->size();
//     float eta_direction = (float)(num_directions)/M_PI;
// 
//     for( int idx = th_id; idx < precomputed_views; idx+=n_threads)
//     {
//       int view_idx = idx;
//     
//       Eigen::Quaterniond quat;
//       Eigen::Vector3d t;
//       std::vector<cv::Point2f> proj_pts;
//       std::vector<float> normal_directions;
//       model_image_ptr->getModel(view_idx, quat, t, proj_pts, normal_directions );
//       int n_pts = 0;
//       float avg_dist = 0;
//       std::vector<cv::Mat > &dist_map_tensor = *dist_map_tensor_ptr;
//       for(int i=0; i<proj_pts.size(); i++)
//       {
//         const cv::Point2f &coord = proj_pts.at(i);
//         if( coord.x >= 0)
//         {
//           n_pts++;
//           
//           float direction = normal_directions[i] + M_PI/2;
//           if( direction >= M_PI/2 )
//             direction -= M_PI;
//           direction += M_PI/2;
//                   
//           int x = round(coord.x), y = round(coord.y);
//           int z = round(eta_direction*direction);
//           z %= num_directions;
//           
//           avg_dist += dist_map_tensor[z].at<float>( y, x );
//         }
//       }
//       if( n_pts )
//         avg_dist /= n_pts;
//       else
//         avg_dist = std::numeric_limits< float >::max();
//         
//       local_idxs.push_back(std::pair<float,int>(avg_dist, view_idx));
//     }
//   }
// 
//   std::cout<<"extract indices from tensor : "<<timer.elapsedTimeMs() <<std::endl;
//   
//   for( int th_id = 0; th_id < n_threads; th_id++)
//   {
//     std::vector<std::pair<float,int> > &local_idxs = idxs_v[th_id];
//     for(int i = 0; i < local_idxs.size(); i++)
//     {
//       map_score_idx.insert ( local_idxs[i] );
//     }
//   }
//   int i=0;
//   int last=100;
//   for( std::map<float,int>::iterator it = map_score_idx.begin(); it != map_score_idx.end()&&i<last; ++it, i++ ) {
//   	idxs.push_back( *it );
//   }
// }
// 
// 
// double ObjectRecognition::evaluateScore ( cv_ext::ImageStatisticsPtr &img_stats_p,
//                                           std::vector<cv::Point2f> &raster_pts,
//                                           const std::vector<float> &normal_directions )
// {
//   // TODO Use gradient magnitude
//   boost::shared_ptr< std::vector<float> > g_dir_p =  img_stats_p->getGradientDirections ( raster_pts );
//   boost::shared_ptr< std::vector<float> > g_mag_p =  img_stats_p->getGradientMagnitudes ( raster_pts );
//   
//   std::vector<float> &g_dir = *g_dir_p;
//   std::vector<float> &g_mag = *g_mag_p;
// 
//   if ( !g_dir.size() || !g_mag_p->size() )
//     return 0;
// 
//   double score = 0;
//   for ( int i = 0; i < g_dir.size(); i++ )
//   {
//     float &direction = g_dir[i], magnitude = g_mag[i];
//     if ( img_stats_p->outOfImage ( direction ) )
//       score += OUTER_POINT_SCORE;
//     else
//       score += ((magnitude > 0.01)?1.0:magnitude ) * std::abs ( cos ( double ( direction ) - normal_directions[i] ) );
//   }
// 
//   return score/g_dir.size();
// }
// 
// void ObjectRecognition::precomputeAndSaveModelViews( RasterObjectModel3DPtr &obj_model_ptr,
//                                                      const std::vector< Eigen::Quaterniond > &init_rotations,
//                                                      const std::vector<Eigen::Vector3d> &init_translations)
// {
//   for( int i_t=0; i_t< init_translations.size(); i_t++)
//   {
//     std::cout<<"init_guess_block "<<i_t<<std::endl;
//     for( int i_rot = 0; i_rot < init_rotations.size(); i_rot++ )
//     {
//       obj_model_ptr->setModelView(init_rotations[i_rot], init_translations[i_t]);
//       obj_model_ptr->storeModelView();
//     }  
//   }      
//   std::cout<<"saving views..."<<std::endl;
//   obj_model_ptr->savePrecomputedModelsViews ("/home/spqr/precomputed_1");
// }
// 
// void ObjectRecognition::precomputeModelViews( RasterObjectModel3DPtr &obj_model_ptr,
//                                                     RasterModelImagePtr &model_image_ptr,
//                                                     const std::vector< Eigen::Quaterniond > &init_rotations, 
//                                                     const std::vector<Eigen::Vector3d> &init_translations)
// {
//   for( int i_t=0; i_t< init_translations.size(); i_t++)
//   {
//     std::cout<<"init_guess_block "<<i_t<<std::endl;
//     for( int i_rot = 0; i_rot < init_rotations.size(); i_rot++ )
//     {
//       obj_model_ptr->setModelView(init_rotations[i_rot], init_translations[i_t]);
//       std::vector<cv::Point2f> proj_pts;
//       std::vector<float> normal_directions;
//       obj_model_ptr->projectRasterPoints(proj_pts, normal_directions);
//       model_image_ptr->storeModel(init_rotations[i_rot], init_translations[i_t], proj_pts, normal_directions);
//     }  
//   }      
//   std::cout<<"num precomputed models: "<<model_image_ptr->numPrecomputedModelsViews()<<std::endl;
// }
// 
// void ObjectRecognition::precomputeAndSaveModelViews( RasterObjectModel3DPtr &obj_model_ptr,
//                                                     RasterModelImagePtr &model_image_ptr,
//                                                     const std::vector< Eigen::Quaterniond > &init_rotations, 
//                                                     const std::vector<Eigen::Vector3d> &init_translations)
// {
//   for( int i_t=0; i_t< init_translations.size(); i_t++)
//   {
//     std::cout<<"init_guess_block "<<i_t<<std::endl;
//     for( int i_rot = 0; i_rot < init_rotations.size(); i_rot++ )
//     {
//       obj_model_ptr->setModelView(init_rotations[i_rot], init_translations[i_t]);
//       std::vector<cv::Point2f> proj_pts;
//       std::vector<float> normal_directions;
//       obj_model_ptr->projectRasterPoints(proj_pts, normal_directions);
//       model_image_ptr->storeModel(init_rotations[i_rot], init_translations[i_t], proj_pts, normal_directions);
//     }  
//   }      
//   std::cout<<"saving image views..."<<std::endl;
//   model_image_ptr->saveModels ("normal", 4);
//   std::cout<<"saving image views..."<<std::endl;
//   model_image_ptr->saveModelsQuantizedNormals ("quantized_normal", 4);
//   
//   std::cout<<"saving done"<<std::endl;
// }
// 
// void ObjectRecognition::extractBestModelViews( const int N, const cv::Mat& img, DirectionalChamferMatching dc_matching,
//                                                RasterObjectModel3DPtr &obj_model_ptr,
//                                               const std::vector< Eigen::Quaterniond > &init_rotations,
//                                               const std::vector<Eigen::Vector3d> &init_translations,
//                                               std::vector<std::pair<double,Eigen::Affine3d> > &model_views )
// {
//   cv_ext::ImageStatisticsPtr img_stats_p =
//   cv_ext::ImageStatistics::createImageStatistics ( img, true );
//   
//   std::map<double,Eigen::Affine3d> map_avgDist_idx;
//   int idx=0;
//   for( int i_t=0; i_t< init_translations.size(); i_t++)
//   {
//     for( int i_rot = 0; i_rot < init_rotations.size(); i_rot++, idx++ )
//     {
//       double avg_d=dc_matching.getAvgDistance( init_rotations[i_rot], init_translations[i_t] );
//       
//       //obj_model_ptr->storeModelView();
//       
//       Eigen::Affine3d T; T.linear()=Eigen::Matrix3d(init_rotations[i_rot]); T.translation()=init_translations[i_t];
//       map_avgDist_idx.insert ( std::pair<double,Eigen::Affine3d>(avg_d , T) );
//       std::cout<<idx<<std::endl;
//     }  
//   }      
//   std::cout<<"Num precomputed views : "<<map_avgDist_idx.size()<<std::endl;
//   int i=0;
//   int last=N;
//   std::cout<<"Num precomputed views considered : "<<last<<std::endl;
//   model_views.clear();
//   for( std::map<double,Eigen::Affine3d>::iterator it = map_avgDist_idx.begin(); it != map_avgDist_idx.end()&&i<last; ++it, i++ ) {
//   	model_views.push_back( *it );
//   }
//   
//   map_avgDist_idx.clear();
//   for(int y=0; y<model_views.size(); y++){
//     Eigen::Quaterniond q(model_views[y].second.linear());
//     Eigen::Vector3d t=model_views[y].second.translation();
//     dc_matching.refinePosition(q,t);
// //     double avg_d=dc_matching.getAvgDistance( q, t );
// //     Eigen::Affine3d T; T.linear()=Eigen::Matrix3d(q); T.translation()=t;
// //     map_avgDist_idx.insert ( std::pair<double,Eigen::Affine3d>(avg_d , T) );
//     
//     double avg_d=dc_matching.getAvgDistance( q, t );
//     Eigen::Affine3d T; T.linear()=Eigen::Matrix3d(q); T.translation()=t;
//     std::vector<cv::Point2f> proj_pts;
//     std::vector<float> normals;
//     obj_model_ptr->setModelView(q,t);
//     obj_model_ptr->projectRasterPoints ( proj_pts, normals );
//     double score=evaluateScore(img_stats_p,proj_pts,normals);
//     
//     map_avgDist_idx.insert ( std::pair<double,Eigen::Affine3d>(avg_d/score , T) );
//   }
//   i=0;
//   model_views.clear();
//   for( std::map<double,Eigen::Affine3d>::iterator it = map_avgDist_idx.begin(); it != map_avgDist_idx.end()&&i<last; ++it, i++ ) {
//     model_views.push_back( *it );
//   }
//   
// }
// 
// void ObjectRecognition::precomputeBestModelViews( DirectionalChamferMatching dc_matching, RasterObjectModel3DPtr &obj_model_ptr,
//                                               const std::vector< Eigen::Quaterniond > &init_rotations,
//                                               const std::vector<Eigen::Vector3d> &init_translations,
//                                               std::vector<int> &idxs )
// {
//   std::map<double,int> map_avgDist_idx;
//   int idx=0;
//   for( int i_t=0; i_t< init_translations.size(); i_t++)
//   {
//     for( int i_rot = 0; i_rot < init_rotations.size(); i_rot++, idx++ )
//     {
//       double avg_d=dc_matching.getAvgDistance( init_rotations[i_rot], init_translations[i_t] );
//       obj_model_ptr->storeModelView();
//       map_avgDist_idx.insert ( std::pair<double,int>(avg_d , idx) );
//     }  
//   }      
//   std::cout<<"Num precomputed views : "<<map_avgDist_idx.size()<<std::endl;
//   int i=0;
//   int last=50;
//   std::cout<<"Num precomputed views considered : "<<last<<std::endl;
//   for( std::map<double,int>::iterator it = map_avgDist_idx.begin(); it != map_avgDist_idx.end()&&i<last; ++it, i++ ) {
//   	idxs.push_back( it->second );
//   }
// }
// 
// 
// void ObjectRecognition::precomputeModelViews( RasterObjectModel3DPtr &obj_model_ptr,
//                                               const std::vector< Eigen::Quaterniond > &init_rotations, 
//                                               const std::vector<Eigen::Vector3d> &init_translations)
// {
// 
//   for( int i_t=0; i_t< init_translations.size(); i_t++)
//   {
//     for( int i_rot = 0; i_rot < init_rotations.size(); i_rot++)
//     {
//       obj_model_ptr->setModelView(init_rotations[i_rot], init_translations[i_t]);
//       obj_model_ptr->storeModelView();
//     }  
//   }      
// }
// 
// void ObjectRecognition::precomputeModelViews( RasterModelImagePtr &model_image_ptr, std::vector<int> &idxs,
//                                               RasterObjectModel3DPtr &obj_model_ptr)
// {
//   for(int i=0; i<idxs.size(); i++)
//   {
//     Eigen::Quaterniond quat;
//     Eigen::Vector3d t;
//     std::vector<cv::Point2f> proj_pts;
//     //std::vector<unsigned char> q_normal_directions;
//     std::vector<float> q_normal_directions;
//     model_image_ptr->getModel(idxs[i], quat, t, proj_pts, q_normal_directions );
//     obj_model_ptr->setModelView(quat, t);
//     obj_model_ptr->storeModelView();
//     idxs[i]=i;
//   }
//   model_image_ptr->clearModel();
// }
// 
// void ObjectRecognition::precomputeModelViews( RasterModelImagePtr &model_image_ptr, std::vector<int> &idxs, std::vector<int> &idxs_out,
//                                               RasterObjectModel3DPtr &obj_model_ptr)
// {
//   idxs_out.resize(idxs.size());
//   for(int i=0; i<idxs.size(); i++)
//   {
//     Eigen::Quaterniond quat;
//     Eigen::Vector3d t;
//     std::vector<cv::Point2f> proj_pts;
//     //std::vector<unsigned char> q_normal_directions;
//     std::vector<float> q_normal_directions;
//     model_image_ptr->getModel(idxs[i], quat, t, proj_pts, q_normal_directions );
//     obj_model_ptr->setModelView(quat, t);
//     obj_model_ptr->storeModelView();
//     idxs_out[i]=i;
//   }
//   //model_image_ptr->clearModel();
// }
// 
// void ObjectRecognition::extractBestModelViews( DirectionalChamferMatching dc_matching, RasterObjectModel3DPtr &obj_model_ptr,
//                                               std::vector<int> &idxs )
// {
//   std::map<double,int> map_avgDist_idx;
//   for(int i=0;i<obj_model_ptr->numPrecomputedModelsViews();i++)
//   {
//     double avg_d=dc_matching.getAvgDistance( i );
//     map_avgDist_idx.insert ( std::pair<double,int>(avg_d , i) );
//   }
//     
//   std::cout<<"Num precomputed views : "<<map_avgDist_idx.size()<<std::endl;
//   int i=0;
//   int last=50;
//   for( std::map<double,int>::iterator it = map_avgDist_idx.begin(); it != map_avgDist_idx.end()&&i<last; ++it, i++ ) {
//   	idxs.push_back( it->second );
//   }
// }
// 
// void ObjectRecognition::precomputeModelViews( RasterObjectModel3DPtr &obj_model_ptr, 
//                                               const std::vector< Eigen::Quaterniond > &init_rotations, 
//                                               Eigen::Quaterniond &plane_q_rot, Eigen::Vector3d &plane_center,
//                                               double step_xy, double area_max_side )
// {
//   int w = obj_model_ptr->cameraModel().imgWidth(),
//       h = obj_model_ptr->cameraModel().imgHeight();
// 
//   double h_area = area_max_side/2.0;
//   double scene_pt[3], img_pt[2], r_quat[4], transf_pt[3];
//   cv_ext::eigenQuat2Quat(plane_q_rot, r_quat);
// 
//   for( scene_pt[0] = -h_area; scene_pt[0] < h_area; scene_pt[0] += step_xy)
//   {
//     for( scene_pt[1] = -h_area; scene_pt[1] < h_area; scene_pt[1] += step_xy)
//     {
//       obj_model_ptr->cameraModel().quatRTProject(plane_q_rot, plane_center, scene_pt, img_pt );
//       
//       int x = round(img_pt[0]), y = round(img_pt[1]);
//       if ( x >= 100 && x < obj_model_ptr->cameraModel().imgWidth() - 100 &&
//            y >= 100 && y < obj_model_ptr->cameraModel().imgHeight() - 100  )
//       {
// 
//         std::cout<<x<<" "<<y<<std::endl;
// 
//         ceres::QuaternionRotatePoint( r_quat, scene_pt, transf_pt); 
//         transf_pt[0] += plane_center(0);
//         transf_pt[1] += plane_center(1);
//         transf_pt[2] += plane_center(2);
//         
//         Eigen::Vector3d t( transf_pt[0], transf_pt[1], transf_pt[2] );
// 
//         for( int i_rot = 0; i_rot < init_rotations.size(); i_rot++ )
//         {
//           Eigen::Quaterniond quat = init_rotations[i_rot];
//           obj_model_ptr->setModelView(quat, t);
//           obj_model_ptr->storeModelView();
//         }        
//       }
//     }
//   }
// }
// 
// bool ObjectRecognition::searchObject ( const std::string& model_name, const cv::Mat& img, 
//                                        Eigen::Quaterniond& q_rot, Eigen::Vector3d& t )
// {
//   if( model_name != current_model_ )
//   {
//     std::map<std::string, Model>::iterator it = models_.find(model_name);
//     if( it == models_.end() )
//       return false;
// 
//     current_model_ = model_name;
//     it->second.obj_model_ptr->loadPrecomputedModelsViews(model_name); 
//   }
//   
//   dbg_input_img = img;
//   
//   Model &model = models_[model_name];
//   cv::Mat obj_mask;
//   getObjectMask( img, obj_mask );
//   
//   //testPrecomputedViews(model.obj_model_ptr);
//   
//   cv_ext::ImageStatisticsPtr img_stats_p =
//     cv_ext::ImageStatistics::createImageStatistics ( img, true );
//  
//   cv_ext::BasicTimer timer;
//   
//   ImageTensorPtr dist_map_tensor_ptr;
//   DistanceTransform dc;
//   dc.computeDistanceMapTensor ( img, dist_map_tensor_ptr );
//   std::cout<<"computeDistanceMapTensor : "<<timer.elapsedTimeMs() <<std::endl;
//   
//   DirectionalChamferMatching dc_matching(cam_model_, dist_map_tensor_ptr );
//   dc_matching.setTemplateModel( model.obj_model_ptr );
//   dc_matching.enableVerbouseMode ( false);
// 
//   std::vector<PoseCandidate> out_poses;
//   
//   globalMatch(dc_matching, model.obj_model_ptr, img_stats_p, out_poses, obj_mask);
//   
//   PoseCandidate &best = out_poses.back();
//   q_rot = best.r;
//   t = best.t;
// }
// 
// bool ObjectRecognition::initialGuessMethods ( RasterObjectModel3DPtr &obj_model_ptr, RasterModelImagePtr &model_image_ptr, cv::Mat& img, 
//                                                        Eigen::Quaterniond& r_gr, Eigen::Vector3d& t_gr,
//                                                        bool algorithm, int &false_pos, int &time, int &time_with_images_computation,
//                                                        bool save_results, std::string save_dir )
// {
//   cv::Mat dbg_img;
//   dbg_input_img = img;
//   
//   if(save_results)
//   {
//     cv::Mat edge_map;
//     LSDEdgeDetector edge_detector;
//     edge_detector.setImage(img);
//     edge_detector.getEdgeMap(edge_map);
//     cv::imwrite(save_dir+"image.png", img);
//     cv::imwrite(save_dir+"edge_map.png", edge_map);
//   }
//   
//   obj_model_ptr->clearModel();
//   
//   cv_ext::BasicTimer timer;
//   
//   //testPrecomputedViews(model.obj_model_ptr);
//   
//   cv_ext::ImageStatisticsPtr img_stats_p =
//   cv_ext::ImageStatistics::createImageStatistics ( img, true );
//   
//   cv_ext::BasicTimer timer2;
//   std::vector<int> view_idxs, view_idxs2;
//   
//   if(!algorithm)
//   {
//     cv::Mat img_gr_dir, img_gr_mag;
//     img_gr_dir=img_stats_p->getGradientDirectionsImage();
//     img_gr_mag=img_stats_p->getGradientMagnitudesImage();
//     cv::Mat binImg = cv::Mat(img_gr_dir.size(), CV_8U, cvScalar(0));
//     computeBinaryImage(img_gr_dir, img_gr_mag, binImg);
//     std::vector<float*> lookup_tables;
//     computeLookupTable(lookup_tables);
//     std::vector<cv::Mat> responseMaps;
//     computeResponseMaps(binImg,lookup_tables,responseMaps);
//     std::cout<<"binary_image and lookup tables time : "<<timer.elapsedTimeMs() <<std::endl;
// 
//     //extractIndicesFromResponseMaps(model_image_ptr, responseMaps,view_idxs);
//     cv_ext::BasicTimer timer3;
//     extractIndicesFromLookupTables(model_image_ptr, binImg, lookup_tables,view_idxs);
//     time_with_images_computation=timer2.elapsedTimeMs();
//     time=timer3.elapsedTimeMs();
//     precomputeModelViews(model_image_ptr, view_idxs,  view_idxs2, obj_model_ptr);
//     std::cout<<"precomputation time elapsed (from lookup_tables): "<<timer2.elapsedTimeMs() <<std::endl; 
//     
//     std::cout<<"views: "<<view_idxs.size()<<std::endl;
//   }
//   else
//   {
//     ImageTensorPtr dist_map_tensor_ptr;
//     DistanceTransform dc;
//     dc.computeDistanceMapTensor ( img, dist_map_tensor_ptr );
//     std::cout<<"computeDistanceMapTensor : "<<timer.elapsedTimeMs() <<std::endl;
//     cv_ext::BasicTimer timer3;
//     extractIndicesFromTensor(model_image_ptr, dist_map_tensor_ptr, view_idxs);
//     time_with_images_computation=timer2.elapsedTimeMs();
//     time=timer3.elapsedTimeMs();
//     precomputeModelViews(model_image_ptr, view_idxs,  view_idxs2, obj_model_ptr);
//     std::cout<<"precomputation time elapsed (from tensor): "<<timer3.elapsedTimeMs() <<std::endl; 
//     std::cout<<"views: "<<view_idxs.size()<<std::endl;
//   }
// 
//   cv::cvtColor(dbg_input_img,dbg_img, cv::COLOR_GRAY2BGR );
//   double good_t=.03; double good_r=.35;
//   for (int i=0;i<view_idxs2.size()/2; i++)
//   {
//     std::vector<cv::Point2f> proj_pts2;
//     std::vector<float> normals2;
//     obj_model_ptr->projectRasterPoints(view_idxs2[i], proj_pts2, normals2);
//     
//     if (save_results&&i==0) cv::imwrite(save_dir+"best_init_guess.png", dbg_img); 
//     
//     cv::Mat_<double> t_vec, r_vec;
//     Eigen::Quaternion<double>  rq;
//     Eigen::Vector3d  t;
//     obj_model_ptr->modelView(view_idxs2[i], rq,t );
//     if((t_gr-t).norm()<good_t)
//     {
//       cv_ext::drawPoints ( dbg_img, proj_pts2,cv::Scalar (   0, 252, 124 ) );
//       cv_ext::showImage(dbg_img, "tmp", true, 2 );
//       break;
//     }
//     false_pos++;
//     cv_ext::drawPoints ( dbg_img, proj_pts2,cv::Scalar (   255, 77, 0 ) );
//     cv_ext::showImage(dbg_img, "tmp", true, 2 );
//   }
//     std::vector<cv::Point2f> proj_pts2;
//     std::vector<float> normals2;
//     obj_model_ptr->setModelView( r_gr,t_gr );
//     obj_model_ptr->projectRasterPoints( proj_pts2, normals2);
//     cv_ext::drawPoints ( dbg_img, proj_pts2,cv::Scalar (   20,40,255 ) );
//     cv_ext::showImage(dbg_img, "tmp", true, 2 );
//    cv::imwrite(save_dir+"good_init_guess.png", dbg_img); 
//   
//   cv::cvtColor(dbg_input_img,dbg_img, cv::COLOR_GRAY2BGR );
// 
// }
// 
// 
// bool ObjectRecognition::detectObject ( const int N, RasterObjectModel3DPtr &obj_model_ptr, const cv::Mat& img, 
//                                        const std::vector<Eigen::Quaterniond>& initial_q,
//                                        const std::vector<Eigen::Vector3d>& initial_t,
//                                        std::vector<std::pair<double,Eigen::Affine3d> >& model_views)
// {
//   dbg_input_img = img;
//   
//   
//   cv_ext::BasicTimer timer;
//   
//   ImageTensorPtr dist_map_tensor_ptr;
//   DistanceTransform dc;
//   dc.computeDistanceMapTensor ( img, dist_map_tensor_ptr );
//   std::cout<<"computeDistanceMapTensor : "<<timer.elapsedTimeMs() <<std::endl;
//   
//   DirectionalChamferMatching dc_matching(cam_model_, dist_map_tensor_ptr );
//   dc_matching.setTemplateModel( obj_model_ptr );
//   dc_matching.enableVerbouseMode ( false);
// 
//   extractBestModelViews(N, img, dc_matching, obj_model_ptr, initial_q, initial_t, model_views);
//   
// }
// 
// 
// bool ObjectRecognition::searchObjectWithInitialGuess ( RasterObjectModel3DPtr &obj_model_ptr, RasterModelImagePtr &model_image_ptr, const cv::Mat& img, 
//                                                        std::vector<Eigen::Quaterniond>& initial_q, std::vector<Eigen::Vector3d>& initial_t,
//                                                        Eigen::Quaterniond& q_rot_out, Eigen::Vector3d& t_out, double &score, 
//                                                        bool save_results, std::string save_dir )
// {
//   dbg_input_img = img;
//   
//   if(save_results)
//   {
//     cv::Mat edge_map;
//     LSDEdgeDetector edge_detector;
//     edge_detector.setImage(img);
//     edge_detector.getEdgeMap(edge_map);
//     cv::imwrite(save_dir+"image.png", img);
//     cv::imwrite(save_dir+"edge_map.png", edge_map);
//   }
//   
//   cv_ext::BasicTimer timer;
//   
//   //testPrecomputedViews(model.obj_model_ptr);
//   
//   cv_ext::ImageStatisticsPtr img_stats_p =
//   cv_ext::ImageStatistics::createImageStatistics ( img, true );
//   
//  /* cv::Mat img_gr_dir, img_gr_mag;
//   img_gr_dir=img_stats_p->getGradientDirectionsImage();
//   img_gr_mag=img_stats_p->getGradientMagnitudesImage();
//   cv::Mat binImg = cv::Mat(img_gr_dir.size(), CV_8U, cvScalar(0));
//   computeBinaryImage(img_gr_dir, img_gr_mag, binImg);
//   std::vector<float*> lookup_tables;
//   computeLookupTable(lookup_tables);
//   std::vector<cv::Mat> responseMaps;
//   computeResponseMaps(binImg,lookup_tables,responseMaps);
//   std::cout<<"binary_image and lookup tables time : "<<timer.elapsedTimeMs() <<std::endl;*/
//   
//   ImageTensorPtr dist_map_tensor_ptr;
//   DistanceTransform dc;
//   dc.computeDistanceMapTensor ( img, dist_map_tensor_ptr );
//   std::cout<<"computeDistanceMapTensor : "<<timer.elapsedTimeMs() <<std::endl;
//   
//   DirectionalChamferMatching dc_matching(cam_model_, dist_map_tensor_ptr );
//   dc_matching.setTemplateModel( obj_model_ptr );
//   dc_matching.enableVerbouseMode ( false);
//   
//   cv_ext::BasicTimer timer2;
//   std::vector<int> view_idxs;
//   //precomputeBestModelViews( dc_matching, obj_model_ptr, initial_q, initial_t, view_idxs);
//   //extractBestModelViews(dc_matching, obj_model_ptr, view_idxs);
//   //precomputeModelViews( obj_model_ptr, initial_q, initial_t);
//   //extractIndicesFromResponseMaps(obj_model_ptr, responseMaps,view_idxs);
//   //extractIndicesFromLookupTables(obj_model_ptr, binImg, lookup_tables,view_idxs);
//   
//   
//   timer2.reset();
//   //extractIndicesFromResponseMaps(model_image_ptr, responseMaps,view_idxs);
//   //extractIndicesFromLookupTables(model_image_ptr, binImg, lookup_tables,view_idxs);
//   extractIndicesFromTensor(model_image_ptr, dist_map_tensor_ptr, view_idxs);
//   precomputeModelViews(model_image_ptr, view_idxs, obj_model_ptr);
//   
//   
//   std::cout<<"precomputation time elapsed: "<<timer2.elapsedTimeMs() <<std::endl;
//                                               
//   std::vector<PoseCandidate> out_poses;
//   
//   matchWithInitialGuess(dc_matching, obj_model_ptr, img_stats_p, out_poses, view_idxs, save_results, save_dir);
//   
//   PoseCandidate &best = out_poses.back();
//   q_rot_out = best.r;
//   t_out = best.t;
//   score = best.score;
//   
//   float total_time_elapsed=(float)timer.elapsedTimeMs()/1000.0f;
//   std::cout<<"total time elapsed (s): "<<total_time_elapsed <<std::endl;
//   
//   if(save_results)
//   {
//     std::ofstream of((save_dir+"result.txt").c_str());
//     of<<"t: "<<t_out(0)<<" "<<t_out(1)<<" "<<t_out(2)<<std::endl;
//     of<<"q: "<<q_rot_out.x()<<" "<<q_rot_out.y()<<" "<<q_rot_out.z()<<" "<<q_rot_out.w()<<std::endl;
//     of<<"score: "<<score<<std::endl;
//     of<<"total time elapsed (s): "<<total_time_elapsed<<std::endl;
//     of.close();
//   }
// }
// 
// void ObjectRecognition::matchWithInitialGuess(DirectionalChamferMatching dc_matching,
//                                     RasterObjectModel3DPtr &obj_model_ptr,
//                                     cv_ext::ImageStatisticsPtr &img_stats_p,
//                                     std::vector<PoseCandidate> &out_poses,
//                                     std::vector<int>& view_idxs,
//                                     bool save_results, std::string save_dir)
// {
//   cv::Mat dbg_img;
//   cv::cvtColor(dbg_input_img,dbg_img, cv::COLOR_GRAY2BGR );
//   
//   std::cout<<"Num precomputed views considered : "<<view_idxs.size()<<std::endl;
//    
//   const int n_threads = omp_get_max_threads();
//   std::cout<<"Max num threads :"<<n_threads<<std::endl;
//   std::vector<PoseCandidate> pose_candidates[n_threads];
//   
//   //cvtColor(dbg_input_img,dbg_img, cv::COLOR_GRAY2BGR );
//   for (int i=0;i<view_idxs.size(); i++)
//   {
//     std::vector<cv::Point2f> proj_pts2;
//     std::vector<float> normals2;
//     obj_model_ptr->projectRasterPoints(view_idxs[i], proj_pts2, normals2);
//     cv_ext::drawPoints ( dbg_img, proj_pts2,cv::Scalar (  255, 0, 0 ) );
//     cv_ext::showImage(dbg_img, "tmp", true, 20 );
//     if (save_results&&i==0) cv::imwrite(save_dir+"best_init_guess.png", dbg_img); 
//   }
//   if (save_results) cv::imwrite(save_dir+"good_init_guess.png", dbg_img); 
//   
//   cv::cvtColor(dbg_input_img,dbg_img, cv::COLOR_GRAY2BGR );
//   
//   //cv::cvtColor(dbg_input_img,dbg_img, cv::COLOR_GRAY2BGR );
//  
//   cv_ext::BasicTimer timer;
//   
//   #pragma omp parallel
//   {
//     DirectionalChamferMatching matching ( dc_matching );
//     int th_id = omp_get_thread_num();
//     std::vector<PoseCandidate> &local_poses = pose_candidates[th_id];
//     int precomputed_views = view_idxs.size();
// 
//     for( int i = th_id; i < precomputed_views; i+=n_threads)
//     {
//       int view_idx = view_idxs[i];
//       
//       Eigen::Quaternion<double> init_r, r;
//       Eigen::Vector3d init_t, t;
//       obj_model_ptr->modelView(view_idx, init_r,init_t );
//       matching.refinePosition(view_idx, r, t);
//       //std::cout<<i<<" on "<<precomputed_views<<std::endl;
//       Eigen::Vector3d diff_t = init_t - t;
//       if ( diff_t.norm() < 0.2  )
//       {
// //         std::cout<<"init : "<<init_t.transpose()<<std::endl;
// //         std::cout<<"res : "<<t.transpose()<<std::endl;
//         std::vector<cv::Point2f> proj_pts;
//         std::vector<float> normals;
//         obj_model_ptr->projectRasterPoints(view_idx, r, t, proj_pts, normals);
//         double score = evaluateScore(img_stats_p, proj_pts, normals);
//                   
//         if (score > 0.74 )
//         {
//           PoseCandidate candiate;
//           candiate.score = score;
//           candiate.r = r;
//           candiate.t = t;
//           local_poses.push_back(candiate);
//         //  cv_ext::drawPoints ( dbg_img, proj_pts,cv::Scalar (  255, 0, 0 ) );
//          // cv_ext::showImage(dbg_img, "tmp", true, 10  );
//         }
//       }
//     }
//   }
//   
//   for( int th_id = 0; th_id < n_threads; th_id++)
//   {
//     std::vector<PoseCandidate> &local_poses = pose_candidates[th_id];
//     for(int i = 0; i < local_poses.size(); i++)
//     {
//       out_poses.push_back(local_poses[i]);
//     }
//   }
//   
//   std::cout <<"globalMatch() elapsed time : "<<timer.elapsedTimeMs() /1000.0<<std::endl;
//   
//   double max_score = 0; 
//   Eigen::Quaternion<double> best_r;
//   Eigen::Vector3d best_t;
//   for( int i = 0; i < out_poses.size(); i++)
//   {
//     Eigen::Quaternion<double> r = out_poses[i].r;
//     Eigen::Vector3d t = out_poses[i].t;
//     dc_matching.refinePosition(r, t);
//     std::vector<cv::Point2f> proj_pts;
//     std::vector<float> normals;
//     obj_model_ptr->projectRasterPoints( proj_pts, normals);
//     double score = evaluateScore(img_stats_p, proj_pts, normals);
//     if( score > max_score )
//     {
//       max_score = score;
//       best_r = r;
//       best_t = t;
//     }
//   }
//   obj_model_ptr->setModelView( best_r, best_t );
//   std::vector<cv::Point2f> proj_pts;
//   obj_model_ptr->projectRasterPoints(proj_pts);
//   cv_ext::drawPoints ( dbg_img, proj_pts,cv::Scalar (  0, 0, 255 ) );
//   cv_ext::showImage(dbg_img, "tmp", true, 20);
//   if (save_results) cv::imwrite(save_dir+"alignment_result.png", dbg_img); 
//   
//   PoseCandidate best;
//   best.score = max_score;
//   best.r = best_r;
//   best.t = best_t;
//   out_poses.push_back(best);
// }
// 
// void ObjectRecognition::globalMatch(DirectionalChamferMatching dc_matching, 
//                                     RasterObjectModel3DPtr &obj_model_ptr,
//                                     cv_ext::ImageStatisticsPtr &img_stats_p,
//                                     std::vector<PoseCandidate> &out_poses,
//                                     const cv::Mat &mask )
// {
//   cv::Mat dbg_img;
//   cv::cvtColor(dbg_input_img,dbg_img, cv::COLOR_GRAY2BGR );
//   std::vector<int> view_idxs;
//   if( !mask.empty())
//   {
//     double scene_pt[] = {0, 0, 0}, proj_pt[2];
//     std::vector<cv::Point> points;
//     for( int i = 0; i < obj_model_ptr->numPrecomputedModelsViews(); i++)
//     {
//       Eigen::Quaterniond quat;
//       Eigen::Vector3d t;
//       obj_model_ptr->setModelView(i);
//       obj_model_ptr->modelView(quat,t );
//       obj_model_ptr->cameraModel().quatRTProject(quat, t, scene_pt, proj_pt);
//       int x = round(proj_pt[0]), y = round(proj_pt[1]);
//       if( mask.at<uchar>(y,x) )
//       {
//         view_idxs.push_back(i);
//         //points.push_back(cv::Point(x,y));
//       }
//     }
//     //cv_ext::drawPoints ( dbg_img, points,cv::Vec3b (  0,0, 255 ) );
//     //cv_ext::showImage(dbg_img, "tmp");
//   }
//   else
//   {
//     for( int i = 0; i < obj_model_ptr->numPrecomputedModelsViews(); i++)
//       view_idxs.push_back(i);
//   }
//   
//   std::cout<<"Num precomputed views : "<<view_idxs.size()<<std::endl;
//    
//   const int n_threads = omp_get_max_threads();
//   std::cout<<"Max num threads :"<<n_threads<<std::endl;
//   std::vector<PoseCandidate> pose_candidates[n_threads];
//   
//   //cvtColor(dbg_input_img,dbg_img, cv::COLOR_GRAY2BGR );
//  
//   cv_ext::BasicTimer timer;
//   
//   #pragma omp parallel
//   {
//     DirectionalChamferMatching matching ( dc_matching );
//     int th_id = omp_get_thread_num();
//     std::vector<PoseCandidate> &local_poses = pose_candidates[th_id];
//     int precomputed_views = view_idxs.size();
// 
//     for( int i = th_id; i < precomputed_views; i+=n_threads)
//     {
//       int view_idx = view_idxs[i];
//       
//       Eigen::Quaternion<double> init_r, r;
//       Eigen::Vector3d init_t, t;
//       obj_model_ptr->modelView(view_idx, init_r,init_t );
//       matching.refinePosition(view_idx, r, t);
//       //std::cout<<i<<" on "<<precomputed_views<<std::endl;
//       Eigen::Vector3d diff_t = init_t - t;
//       if ( diff_t.norm() < 0.2  )
//       {
// //         std::cout<<"init : "<<init_t.transpose()<<std::endl;
// //         std::cout<<"res : "<<t.transpose()<<std::endl;
//         std::vector<cv::Point2f> proj_pts;
//         std::vector<float> normals;
//         obj_model_ptr->projectRasterPoints(view_idx, r, t, proj_pts, normals);
//         double score = evaluateScore(img_stats_p, proj_pts, normals);
//                   
//         if (score > 0.75 )
//         {
//           PoseCandidate candiate;
//           candiate.score = score;
//           candiate.r = r;
//           candiate.t = t;
//           local_poses.push_back(candiate);
//           cv_ext::drawPoints ( dbg_img, proj_pts,cv::Scalar (  255, 0, 0 ) );
//           cv_ext::showImage(dbg_img, "tmp", true, 10  );
//         }
//       }
//     }
//   }
//   
//   for( int th_id = 0; th_id < n_threads; th_id++)
//   {
//     std::vector<PoseCandidate> &local_poses = pose_candidates[th_id];
//     for(int i = 0; i < local_poses.size(); i++)
//     {
//       out_poses.push_back(local_poses[i]);
//     }
//   }
//   
//   std::cout <<"globalMatch() elapsed time : "<<timer.elapsedTimeMs() /1000.0<<std::endl;
//   
//   double max_score = 0; 
//   Eigen::Quaternion<double> best_r;
//   Eigen::Vector3d best_t;
//   for( int i = 0; i < out_poses.size(); i++)
//   {
//     Eigen::Quaternion<double> r = out_poses[i].r;
//     Eigen::Vector3d t = out_poses[i].t;
//     dc_matching.refinePosition(r, t);
//     std::vector<cv::Point2f> proj_pts;
//     std::vector<float> normals;
//     obj_model_ptr->projectRasterPoints(proj_pts, normals);
//     double score = evaluateScore(img_stats_p, proj_pts, normals);
// 
//     if( score > max_score )
//     {
//       max_score = score;
//       best_r = r;
//       best_t = t;
//     }
//   }
//   obj_model_ptr->setModelView( best_r, best_t );
//   std::vector<cv::Point2f> proj_pts;
//   obj_model_ptr->projectRasterPoints(proj_pts);
//   cv_ext::drawPoints ( dbg_img, proj_pts,cv::Scalar (  0, 0, 255 ) );
//   cv_ext::showImage(dbg_img, "tmp");
//   PoseCandidate best;
//   best.score = max_score;
//   best.r = best_r;
//   best.t = best_t;
//   out_poses.push_back(best);
// }
// 
// void ObjectRecognition::testPrecomputedViews( RasterObjectModel3DPtr &obj_model_ptr )
// {
//   cv::Mat dbg_img;
//   cv::cvtColor(dbg_input_img,dbg_img, cv::COLOR_GRAY2BGR );
//   std::vector<cv::Point> points;
//   double scene_pt[] = {0, 0, 0}, proj_pt[2];
//   std::cout<<obj_model_ptr->numPrecomputedModelsViews()<<std::endl;
//   for( int i = 0; i < obj_model_ptr->numPrecomputedModelsViews(); i++)
//   {
//     Eigen::Quaterniond quat;
//     Eigen::Vector3d t;
//     obj_model_ptr->setModelView(i);
//     obj_model_ptr->modelView(quat,t );
//     obj_model_ptr->cameraModel().quatRTProject(quat, t, scene_pt, proj_pt);
//     int x = round(proj_pt[0]), y = round(proj_pt[1]);
//     points.push_back(cv::Point(x,y));
//     std::cout<<x<<" "<<y<<std::endl;
//   }
//   cv_ext::drawPoints ( dbg_img, points,cv::Scalar (  0,0, 255 ) );
//   cv_ext::showImage(dbg_img, "tmp");
//   
//   for( int i = 0; i < obj_model_ptr->numPrecomputedModelsViews(); i++)
//   {
//     cv::cvtColor(dbg_input_img,dbg_img, cv::COLOR_GRAY2BGR );
//     std::vector<cv::Point2f> proj_pts;
//     obj_model_ptr->setModelView(i);
//     Eigen::Quaterniond quat;
//     Eigen::Vector3d t;
//     obj_model_ptr->modelView(quat,t);
//     
//     obj_model_ptr->projectRasterPoints(proj_pts);
//     cv_ext::drawPoints ( dbg_img, proj_pts,cv::Scalar (  0,0, 255 ) );
//     std::cout<<proj_pts.size()<<std::endl;
//     cv_ext::showImage(dbg_img, "tmp"/*, true, 10  */);
//   }
// }
// 
// void ObjectRecognition::project3dEigenPoints(const cv::Mat& K, const double scale, const std::vector<Eigen::Vector3d>& p_list, std::vector<cv::Point2f>& proj_pts)
// {
//   proj_pts.clear();
//   double fx=K.at<double>(0,0)/scale; double fy=K.at<double>(1,1)/scale; double cx=K.at<double>(0,2)/scale; double cy=K.at<double>(1,2)/scale;
//   for(int i=0;i<p_list.size();i++){
//     cv::Point2f p;
//     p.x=(float)(p_list[i](0)*fx + p_list[i](2)*cx)/p_list[i](2);
//     p.y=(float)(p_list[i](1)*fy + p_list[i](2)*cy)/p_list[i](2);
//     if(p.x>0&&p.x<cam_model_.imgWidth()&&p.y>0&&p.y<cam_model_.imgHeight())
//     {
//       proj_pts.push_back(p);
//     //  std::cout<<"pr_pts: "<<p.x<<" "<<p.y<<std::endl;
//     }
//   }
//   //std::cout<<"number pr_pts: "<<proj_pts.size()<<std::endl;
// }
// 
// 
// void ObjectRecognition::computeInitGuess(const Eigen::Vector3d& t, const Eigen::Matrix3d& R, std::vector<Eigen::Vector3d>& t_list)
// {
//   //t_list.clear();
//   t_list.push_back(t);
//   
//   double l_z=0;double l_y=.2; //.14
//   double l_x=.2;
//   double step=.05; //.02
//   Eigen::Affine3d T; T.linear()=R; T.translation()=t;
//   
//   for(double x=-l_x/2;x<l_x/2;x+=step){
//     for(double y=-(l_y/2);y<l_y/2;y+=step){
//       //for(double z=-(l_z/2);z<=l_z/2;z+=.02){
//         
//         Eigen::Vector3d p(x,y,0);
//         p=T*p;
//         t_list.push_back(p);
//       //}
//     }
//   }
//   
//   /////RANDOM/////
//   /*for (int i=0;i<5;i++){
//     double x=(double)(rand()%200-100)/500.0;
//     double y=(double)(rand()%200-100)/500.0;
//     double z=(double)(rand()%200-100)/4000.0;
//     Eigen::Vector3d p(x,y,z);
//     p=T*p;
//     t_list.push_back(p);
//   }*/
//   
//   //std::cout<<"number initial guess: "<<t_list.size()<<std::endl;
// }
// 
// void ObjectRecognition::computeStarOfRotations(const Eigen::Matrix3d& initial_orientation, std::vector< Eigen::Quaterniond >& quat_rotations)
// {
//   Eigen::Vector3d unit_x(initial_orientation.col(2));
//   Eigen::Vector3d unit_y(initial_orientation.col(1));
//   Eigen::Vector3d unit_z(initial_orientation.col(0));
// //   quat_rotations.clear();
//   std::vector<Eigen::Quaterniond> base_rotations;
//   base_rotations.push_back(Eigen::Quaterniond(Eigen::AngleAxisd(0,unit_y)));
//   base_rotations.push_back(Eigen::Quaterniond(Eigen::AngleAxisd(M_PI/2,unit_y)));
//   base_rotations.push_back(Eigen::Quaterniond(Eigen::AngleAxisd(M_PI,unit_y)));
//   base_rotations.push_back(Eigen::Quaterniond(Eigen::AngleAxisd(3*M_PI/2,unit_y)));
//   base_rotations.push_back(Eigen::Quaterniond(Eigen::AngleAxisd(M_PI/2,unit_z)));
//   base_rotations.push_back(Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI/2,unit_z)));
// 
//   for(size_t i=0; i<base_rotations.size();i++)
//   {
//     Eigen::Matrix3d base_rot=(base_rotations[i].toRotationMatrix()*initial_orientation);
//     Eigen::Vector3d rot_ax;
//       rot_ax=unit_x;
//     
//     for(double k=0;k<2*M_PI;k+=2*M_PI/10)
//     {
//       Eigen::Matrix3d R(Eigen::AngleAxisd(k,rot_ax));
//       R=R*base_rot;
//       Eigen::Quaterniond q(R);
//       quat_rotations.push_back(q);
//     }
//   }
// }
// 
// void ObjectRecognition::getCandidatesAvgDist(RasterObjectModel3DPtr& obj_model_ptr, const cv::Mat& img, 
//                                              std::vector< std::pair< double, Eigen::Affine3d > >& model_views)
// {
//   ImageTensorPtr dist_map_tensor_ptr;
//   DistanceTransform dc;
//   dc.computeDistanceMapTensor ( img, dist_map_tensor_ptr );
//   
//   DirectionalChamferMatching dc_matching(cam_model_, dist_map_tensor_ptr );
//   dc_matching.setTemplateModel( obj_model_ptr );
//   dc_matching.enableVerbouseMode ( false);
//   
//   for(int y=0; y<model_views.size(); y++){
//     Eigen::Quaterniond q(model_views[y].second.linear());
//     Eigen::Vector3d t=model_views[y].second.translation();
//     //dc_matching.performOptimization(q,t);
//     double avg_d=dc_matching.getAvgDistance( q, t );
//     model_views[y].first=avg_d;
//   }
//   
//   //debug
//   cv::Mat dbg_img_global;
//   std::vector<cv::Point2f> proj_pts;
//   std::vector<float> normals;
//   cv::cvtColor ( img, dbg_img_global, cv::COLOR_GRAY2BGR );
//   for(int i=0; i<model_views.size(); i++){
//     Eigen::Quaterniond r_q(model_views[i].second.linear());
//     //cv::cvtColor ( images[0], dbg_img_global, cv::COLOR_GRAY2BGR );
//     obj_model_ptr->setModelView(r_q, model_views[i].second.translation());
//     obj_model_ptr->projectRasterPoints ( proj_pts, normals );
//     cv_ext::drawPoints ( dbg_img_global, proj_pts,cv::Scalar ( 255,0,0 ) );
//     cv::imshow ( "new detection", dbg_img_global );
//     cv::waitKey(20);
//   }
//   cv::destroyAllWindows();
// }
// 
// void ObjectRecognition::getMultiCandidatesScore(std::vector< RasterObjectModel3DPtr >& multi_obj_model_ptr, const cv::Mat& img, std::vector< std::pair< int, std::pair< double, Eigen::Affine3d > > >& model_views)
// {
//   ImageTensorPtr dist_map_tensor_ptr;
//   DistanceTransform dc;
//   dc.computeDistanceMapTensor ( img, dist_map_tensor_ptr );
//   
//   cv_ext::ImageStatisticsPtr img_stats_p =
//   cv_ext::ImageStatistics::createImageStatistics ( img, true );
//   
//   for(int idx=0; idx<multi_obj_model_ptr.size(); idx++)
//   {
//     DirectionalChamferMatching dc_matching(cam_model_, dist_map_tensor_ptr );
//     dc_matching.setTemplateModel( multi_obj_model_ptr[idx] );
//     dc_matching.enableVerbouseMode ( false);
//     
//     for(int y=0; y<model_views.size(); y++){
//       if(model_views[y].first!=idx) continue;
//       Eigen::Quaterniond q(model_views[y].second.second.linear());
//       Eigen::Vector3d t=model_views[y].second.second.translation();
//       dc_matching.refinePosition(q,t);
// //       double avg_d=dc_matching.getAvgDistance( q, t );
//       std::vector<cv::Point2f> proj_pts;
//       std::vector<float> normals;
//       multi_obj_model_ptr[idx]->setModelView(q,t);
//       multi_obj_model_ptr[idx]->projectRasterPoints ( proj_pts, normals );
//       double score=evaluateScore(img_stats_p,proj_pts,normals);
//       
//       model_views[y].first=1/score;
//     }
//     
//     //debug
//     cv::Mat dbg_img_global;
//     std::vector<cv::Point2f> proj_pts;
//     std::vector<float> normals;
//     cv::cvtColor ( img, dbg_img_global, cv::COLOR_GRAY2BGR );
//     for(int i=0; i<model_views.size(); i++){
//       if(model_views[i].first!=idx) continue;
//       Eigen::Quaterniond r_q(model_views[i].second.second.linear());
//       //cv::cvtColor ( images[0], dbg_img_global, cv::COLOR_GRAY2BGR );
//       multi_obj_model_ptr[idx]->setModelView(r_q, model_views[i].second.second.translation());
//       multi_obj_model_ptr[idx]->projectRasterPoints ( proj_pts, normals );
//       cv_ext::drawPoints ( dbg_img_global, proj_pts,cv::Scalar ( 255,0,0 ) );
//       cv::imshow ( "new detection", dbg_img_global );
//       cv::waitKey(20);
//     }
//     cv::destroyAllWindows();
//   }
//   cv::destroyAllWindows();
// }
// 
// void ObjectRecognition::getCandidatesScore(RasterObjectModel3DPtr& obj_model_ptr, const cv::Mat& img, 
//                                            std::vector< std::pair< double, Eigen::Affine3d > >& model_views)
// {
//   ImageTensorPtr dist_map_tensor_ptr;
//   DistanceTransform dc;
//   dc.computeDistanceMapTensor ( img, dist_map_tensor_ptr );
//   
//   cv_ext::ImageStatisticsPtr img_stats_p =
//   cv_ext::ImageStatistics::createImageStatistics ( img, true );
//   
//   DirectionalChamferMatching dc_matching(cam_model_, dist_map_tensor_ptr );
//   dc_matching.setTemplateModel( obj_model_ptr );
//   dc_matching.enableVerbouseMode ( false);
//   
//   for(int y=0; y<model_views.size(); y++){
//     Eigen::Quaterniond q(model_views[y].second.linear());
//     Eigen::Vector3d t=model_views[y].second.translation();
//     dc_matching.refinePosition(q,t);
// //      double avg_d=dc_matching.getAvgDistance( q, t );
//     std::vector<cv::Point2f> proj_pts;
//     std::vector<float> normals;
//     obj_model_ptr->setModelView(q,t);
//     obj_model_ptr->projectRasterPoints ( proj_pts, normals );
//     double score=evaluateScore(img_stats_p,proj_pts,normals);
//     
//     model_views[y].first=1/score;
//   }
//   
//   //debug
//   cv::Mat dbg_img_global;
//   std::vector<cv::Point2f> proj_pts;
//   std::vector<float> normals;
//   cv::cvtColor ( img, dbg_img_global, cv::COLOR_GRAY2BGR );
//   for(int i=0; i<model_views.size(); i++){
//     Eigen::Quaterniond r_q(model_views[i].second.linear());
//     //cv::cvtColor ( images[0], dbg_img_global, cv::COLOR_GRAY2BGR );
//     obj_model_ptr->setModelView(r_q, model_views[i].second.translation());
//     obj_model_ptr->projectRasterPoints ( proj_pts, normals );
//     cv_ext::drawPoints ( dbg_img_global, proj_pts,cv::Scalar ( 255,0,0 ) );
//     cv::imshow ( "new detection", dbg_img_global );
//     cv::waitKey(20);
//   }
//   cv::destroyAllWindows();
// }
// 
// void ObjectRecognition::extractObjectCandidates( const int N, RasterObjectModel3DPtr &obj_model_ptr, const cv::Mat& img, 
//                                                  std::vector<std::pair<double,Eigen::Affine3d> >& model_views )
// {
//   cv::Mat_<double> r_vec = ( cv::Mat_<double> ( 3,1 ) << -2.61,0.38,0.15 ),
//                    t_vec = ( cv::Mat_<double> ( 3,1 ) << 0.07, -0.01,  0.23 );
//                    
//   Eigen::Vector3d r_vec_eigen, t_vec_eigen;
//   Eigen::Matrix3d initial_rotation;
//   std::vector<Eigen::Quaterniond> quat_rotations;
//   
//   cv::Mat dbg_img_global;
//   cv::cvtColor ( img, dbg_img_global, cv::COLOR_GRAY2BGR );
//     
//   std::vector<cv::Point2f> proj_pts;
//   std::vector<float> normals;
//   
//   bool exit_now=false;
//   cv::namedWindow("init_guess_img");
//   std::vector<Eigen::Vector3d> list_t;
//   cv_ext::BasicTimer timer;
//   while ( !exit_now )
//   {
// 
//     r_vec_eigen=Eigen::Vector3d(r_vec(0,0),r_vec(1,0),r_vec(2,0));
//     if(r_vec_eigen.norm()==0) initial_rotation=Eigen::Matrix3d::Identity();
//     else initial_rotation=Eigen::Matrix3d(Eigen::AngleAxisd(r_vec_eigen.norm(),r_vec_eigen/r_vec_eigen.norm()));
//     t_vec_eigen=Eigen::Vector3d(t_vec(0,0),t_vec(1,0),t_vec(2,0));
//     
//     Eigen::Quaterniond r_q(initial_rotation);
//     
//     Eigen::Affine3d init_T; init_T.linear()=initial_rotation; init_T.translation()=t_vec_eigen;
// 
//     //////VISUAL/////////////////
//     cv::cvtColor ( img,dbg_img_global, cv::COLOR_GRAY2BGR );
//     obj_model_ptr->setModelView(r_q, init_T.translation());
//     obj_model_ptr->projectRasterPoints ( proj_pts, normals );
//     cv_ext::drawPoints ( dbg_img_global, proj_pts,cv::Scalar ( 0,0,255 ) );
// 
//     
// //     computeInitGuess(t_vec_eigen, initial_rotation, list_t);
//     std::vector<cv::Point2f> pr_pts;
//     project3dEigenPoints(cam_model_.cameraMatrix(),/*cam_model_.sizeScaleFactor()*/ 1, list_t, pr_pts);
//     cv_ext::drawPoints ( dbg_img_global, pr_pts,cv::Scalar ( 0,255,0 ) );
//     
//     cv::imshow ( "init_guess_img", dbg_img_global );
//     //std::cout<<t_vec_eigen.transpose()<<"    "<<r_vec_eigen<<std::endl;
//     //////////////////////////////////7
//     
//    
//     int key = cv::waitKey();
//     //std::cout<< key <<std::endl;
//     switch ( key )
//     {
//       case 'i':
//       case 'I':
//         t_vec.at<double> ( 1,0 ) -= 0.01;
//         break;
//       case 'k':
//       case 'K':
//         t_vec.at<double> ( 1,0 ) += 0.01;
//         break;
//       case 'j':
//       case 'J':
//         t_vec.at<double> ( 0,0 ) -= 0.01;
//         break;
//       case 'l':
//       case 'L':
//         t_vec.at<double> ( 0,0 ) += 0.01;
//         break;
//       case 'u':
//       case 'U':
//         t_vec.at<double> ( 2,0 ) -= 0.01;
//         break;
//       case 'o':
//       case 'O':
//         t_vec.at<double> ( 2,0 ) += 0.01;
//         break;
//       case 'a':
//       case 'A':
//         r_vec.at<double> ( 1,0 ) += 0.01;
//         break;
//       case 's':
//       case 'S':
//         r_vec.at<double> ( 1,0 ) -= 0.01;
//         break;
//       case 'w':
//       case 'W':
//         r_vec.at<double> ( 0,0 ) += 0.01;
//         break;
//       case 'z':
//       case 'Z':
//         r_vec.at<double> ( 0,0 ) -= 0.01;
//         break;
//       case 'q':
//       case 'Q':
//         r_vec.at<double> ( 2,0 ) += 0.01;
//         break;
//       case 'e':
//       case 'E':
//         r_vec.at<double> ( 2,0 ) -= 0.01;
//         break;
//         
//       case ' ':
// //          computeStarOfRotations(initial_rotation,quat_rotations);
//          //// visual /////
// //          for(int tr=0;tr<list_t.size();tr++){
// //            for(int rq=0;rq<quat_rotations.size();rq++){
// //                cv::cvtColor ( img,dbg_img_global, cv::COLOR_GRAY2BGR );
// //                 obj_model_ptr->setModelView(quat_rotations[rq], list_t[tr]);
// //                 obj_model_ptr->projectRasterPoints ( proj_pts, normals );
// //                 cv_ext::drawPoints ( dbg_img_global, proj_pts,cv::Scalar ( 0,0,255 ) );
// //                 cv::imshow ( "cose", dbg_img_global );
// //                 cv::waitKey(0);
// //            }
// //          }
//          /////////////////
//         timer.reset();
//          detectObject(N, obj_model_ptr, img, quat_rotations, list_t, model_views);
//          std::cout<<"detection elapsed time: "<<timer.elapsedTimeMs()<<std::endl;
//          cv::cvtColor ( img,dbg_img_global, cv::COLOR_GRAY2BGR );
//          for(int i=0; i<model_views.size(); i++){
//             Eigen::Quaterniond r_q(model_views[i].second.linear());
//             //cv::cvtColor ( images[0], dbg_img_global, cv::COLOR_GRAY2BGR );
//             obj_model_ptr->setModelView(r_q, model_views[i].second.translation());
//             obj_model_ptr->projectRasterPoints ( proj_pts, normals );
//             cv_ext::drawPoints ( dbg_img_global, proj_pts,cv::Scalar ( 255,0,0 ) );
//             cv::imshow ( "detection", dbg_img_global );
//             cv::waitKey(20);
//             
//          }
//          
//          cv::imwrite("/home/marco/ap_datasets/results_images/detection.jpg", dbg_img_global);
// 
//         exit_now=true; 
//         break;
//         
//       case 'b':
//         computeInitGuess(t_vec_eigen, initial_rotation, list_t);
// //         computeStarOfRotations(initial_rotation,quat_rotations);
//         //quat_rotations.clear();
//         quat_rotations.push_back(Eigen::Quaterniond(initial_rotation));
//         break;
//     }
//   }
//   cv::destroyAllWindows();
// }
// 
// void ObjectRecognition::extractMultiObjectCandidates(const int N, std::vector< RasterObjectModel3DPtr >& multi_obj_model_ptr, 
//                                                      const cv::Mat& img, 
//                                                      std::vector< std::pair< int, std::pair< double, Eigen::Affine3d > > >& model_views)
// {
//   std::vector<std::pair< double, Eigen::Affine3d > > candidates;
//   std::map<double, std::pair<int,Eigen::Affine3d> > map_candidates;
//   for(int i=0; i<multi_obj_model_ptr.size(); i++)
//   {
//     candidates.clear();
//     extractObjectCandidates(N,multi_obj_model_ptr[i],img, candidates);
//     for(int c=0; c<candidates.size(); c++)
//     {
//       map_candidates.insert(std::pair<double, std::pair<int,Eigen::Affine3d> >(candidates[c].first, std::pair<int, Eigen::Affine3d>(i,candidates[c].second)));
//     }
//   }
//   
//   int count=0;
//   for( std::map<double, std::pair<int,Eigen::Affine3d> >::iterator it = map_candidates.begin(); it != map_candidates.end()&&count<N; ++it, count++ ) {
//     std::pair< int, std::pair< double, Eigen::Affine3d > > pair(it->second.first, std::pair<double,Eigen::Affine3d>(it->first, it->second.second));
//     model_views.push_back( pair );
//   }
// }
// 
