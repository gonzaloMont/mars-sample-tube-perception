#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

class HDRCreator
{

public:
  enum HDRCreatorWeightsType
  {
    HDRCreatorWeightsType_Triangular = 0,
    HDRCreatorWeightsType_Gaussian,
    HDRCreatorWeightsType_Plateau
  };

  enum HDRCreatorResponseType
  {
    HDRCreatorResponseType_Linear = 0,
    HDRCreatorResponseType_Gamma,
    HDRCreatorResponseType_Log10
  };

  enum HDRCreatorMappingMethod
  {
    HDRCreatorMappingMethod_Linear = 0,
    HDRCreatorMappingMethod_Gamma_1_4,
    HDRCreatorMappingMethod_Gamma_1_8,
    HDRCreatorMappingMethod_Gamma_2_2,
    HDRCreatorMappingMethod_Gamma_2_6,
    HDRCreatorMappingMethod_Log10
  };


private:


  HDRCreatorWeightsType e_WeightsType;
  HDRCreatorResponseType e_ResponseType;
  HDRCreatorMappingMethod e_MappingMethod;

  float hist_win_min, hist_win_max;
  int min_response, max_response;

  // ---- Constructors - Destructors - Initialization
public:
  HDRCreator ();
  ~HDRCreator();

private:
  void InitBasicVars();

  // ---- Methods
public:

  cv::Mat ComputeHDR ( std::vector<cv::Mat> &ldr_images, std::vector<float> &arrayofexptime );
  cv::Mat MapToLDR ( cv::Mat &hdrimage, double targetexposure = 0 );

  // ---- Calculation Methods
private:
  void applyResponse ( std::vector<cv::Mat> &ldr_images, const std::vector<float> &arrayofexptime, cv::Mat &hdr_out,
                       const float* response, const float* w, const int pix_levels );

  void weightsTriangle ( float* w, int M );
  void weightsGauss ( float* w, int M, int Mmin, int Mmax, float sigma  = 8.0f );
  void exposureWeightsIcip06 ( float* w, int M, int Mmin, int Mmax );

  void responseLinear ( float* response, int M );
  void responseGamma ( float* response, int M );
  void responseLog10 ( float* response, int M );

  typedef int ( HDRCreator::*MappingFunc ) ( float x );
  inline int getMappingLinear ( float x );
  inline int getMappingGamma_1_4 ( float x );
  inline int getMappingGamma_1_8 ( float x );
  inline int getMappingGamma_2_2 ( float x );
  inline int getMappingGamma_2_6 ( float x );
  inline int getMappingLog10 ( float x );

  void fitToDynamicRange ( cv::Mat &hdrimage );

  template<class T> T clamp ( T val, T min, T max );
};
