#pragma once

#include <cv_ext/types.h>

namespace cv_ext
{
/**
 * @brief Utility class used to allocate cv::Mat with specific memory aligned (16, 32, 64, .. bytes)
 */ 
class AlignedMat : public cv::Mat
{
public:
    
  /**
   * @brief Empty constructor: the mat should be created using the method create()
   * 
   * @param [in] align The data will be aligned at the requested boundary, see cv_ext::MemoryAlignment
   */  
  AlignedMat( MemoryAlignment align );

  /**
   * @brief Constructor: create a preallocated matrix with the size provided in input
   * 
   * @param [in] rows New matrix number of rows
   * @param [in] cols New matrix number of columns
   * @param [in] data_type New matrix type, use cv::DataType< T >::type 
   * @param [in] align The data will be aligned at the requested boundary, see cv_ext::MemoryAlignment
   */  
  AlignedMat( int rows, int cols, int data_type, MemoryAlignment align );

  /**
   * @brief Constructor: create a preallocated matrix with the size provided in input
   * 
   * @param [in] size New matrix size: cv::Size(cols, rows)
   * @param [in] data_type New matrix type, use cv::DataType< T >::type 
   * @param [in] align The data will be aligned at the requested boundary, see cv_ext::MemoryAlignment
   */  
  AlignedMat( cv::Size size, int data_type, MemoryAlignment align );

  /**
   * @brief Constructor: create a preallocated matrix with the size provided in input
   * 
   * @param [in] rows New matrix number of rows
   * @param [in] cols New matrix number of columns
   * @param [in] data_type New matrix type, use cv::DataType< T >::type 
   * @param [in] s A value used to initialize each matrix element 
   * @param [in] align The data will be aligned at the requested boundary, see cv_ext::MemoryAlignment
   */  
  AlignedMat( int rows, int cols, int data_type, const cv::Scalar& s, MemoryAlignment align );

  /**
   * @brief Constructor: create a preallocated matrix with the size provided in input
   * 
   * @param [in] size New matrix size: cv::Size(cols, rows)
   * @param [in] data_type New matrix type, use cv::DataType< T >::type
   * @param [in] s A value used to initialize each matrix element 
   * @param [in] align The data will be aligned at the requested boundary, see cv_ext::MemoryAlignment
   */  
  AlignedMat( cv::Size size, int data_type, const cv::Scalar& s, MemoryAlignment align );
  
  /**
   * @brief Destructor: in case, release the manually allocated aigned memory
   */  
  virtual ~AlignedMat(){};
  
private:
    
  void setAllocator( cv_ext::MemoryAlignment align );
  
};

}
