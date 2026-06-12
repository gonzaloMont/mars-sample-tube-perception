#include <sstream>

#include "cv_ext/image_statistics.h"
#include "cv_ext/debug_tools.h"

int main(int argc, char** argv)
{

  char *image_name;
  unsigned int pyr_levels = 4;
  double pyr_scale_factor = 2.0;
  for( int i = 1 ; i < argc - 1; i+=2 )
  {
    if( strcmp(argv[i],"--img") == 0)
      image_name = argv[i+1];
    else if( strcmp(argv[i],"--pyr_levels") == 0)
      pyr_levels = atoi(argv[i+1]);
    else if( strcmp(argv[i],"--pyr_scale_factor") == 0)
      pyr_scale_factor = atof(argv[i+1]);
  }
  cv::Mat input_img = cv::imread(image_name, cv::IMREAD_COLOR);
  cv_ext::ImageStatisticsPtr img_stats_ptr = 
    cv_ext::ImageStatistics::createImageStatistics(input_img, false, pyr_levels, pyr_scale_factor );
  
  std::vector<std::string> scale_str;
  for( int i = 0; i < img_stats_ptr->numPyrLevels(); i++)
  {
    std::ostringstream strs;
    strs << img_stats_ptr->getPyrScale(i);
    scale_str.push_back(strs.str());
  }
  
  for( int i = 0; i < img_stats_ptr->numPyrLevels(); i++)
  {
    std::ostringstream strs;
    strs << "Intensities, scale : "<<scale_str.at(i);
    cv_ext::showImage( img_stats_ptr->getIntensitiesImage(i), strs.str(), true, 10 );
  }
  while(cv_ext::waitKeyboard() != 27);
  cv::destroyAllWindows();
  
  for( int i = 0; i < img_stats_ptr->numPyrLevels(); i++)
  {
    std::ostringstream strs;
    strs << "Colors, scale : "<<scale_str.at(i);
    cv_ext::showImage( img_stats_ptr->getColorsImage(i), strs.str(), true, 10  );
  }
  while(cv_ext::waitKeyboard() != 27);
  cv::destroyAllWindows();
  
  for( int i = 0; i < img_stats_ptr->numPyrLevels(); i++)
  {
    std::ostringstream strs;
    strs << "Blurred, scale : "<<scale_str.at(i);
    cv_ext::showImage( img_stats_ptr->getBlurredImage(i), strs.str(), true, 10  );
  }
  while(cv_ext::waitKeyboard() != 27);
  cv::destroyAllWindows();
  
  for( int i = 0; i < img_stats_ptr->numPyrLevels(); i++)
  {
    std::ostringstream strs;
    strs << "Gradient directions, scale : "<<scale_str.at(i);
    cv_ext::showImage( img_stats_ptr->getGradientDirectionsImage(i), strs.str(), true, 10  );
  }
  while(cv_ext::waitKeyboard() != 27);
  cv::destroyAllWindows();
  
  for( int i = 0; i < img_stats_ptr->numPyrLevels(); i++)
  {
    std::ostringstream strs;
    strs << "Gradien magnitude, scale : "<<scale_str.at(i);
    cv_ext::showImage( img_stats_ptr->getGradientMagnitudesImage(i), strs.str(), true, 10  );
  }
  while(cv_ext::waitKeyboard() != 27);
  cv::destroyAllWindows();
  
  for( int i = 0; i < img_stats_ptr->numPyrLevels(); i++)
  {
    std::ostringstream strs;
    strs << "Eigen magnitude, scale : "<<scale_str.at(i);
    cv_ext::showImage( img_stats_ptr->getEigenMagnitudesImage(i), strs.str(), true, 10  );
  }
  while(cv_ext::waitKeyboard() != 27);
  cv::destroyAllWindows();
  
  return 0;
}