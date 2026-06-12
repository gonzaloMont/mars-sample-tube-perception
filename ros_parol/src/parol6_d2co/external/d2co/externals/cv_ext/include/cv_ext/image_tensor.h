#pragma once

#include <cv_ext/types.h>

namespace cv_ext
{
/**
 * @brief ImageTensor represents a 3D tensor (i.e., a 3D matrix) by means a simple vector of Open CV images (cv::Mat). It supports to specify 
 *        the memory aligned (16, 32, 64, .. bytes).
 */ 
class ImageTensor
{
public:
    
  /**
   * @brief Empty constructor: the tensor should be created using the method create()
   */  
  ImageTensor();
  
  /**
   * @brief Destructor: in case, release the manually allocated aigned memory
   */  
  ~ImageTensor();
  
  /**
   * @brief Constructor: create a preallocated tensor with the size provided in input
   * 
   * @param [in] rows New tensor number of rows
   * @param [in] cols New tensor number of columns
   * @param [in] depth New tensor number of depths (i.e., the number of images)
   * @param [in] data_type New matrix type, use cv::DataType< T >::type 
   * @param [in] align If not MEM_ALIGN_NONE, the data will be aligned at the requested boundary
   * 
   * Allocate a vector of depth cv::Mat, each one with size rows X cols and type provided by tha parameter data_type
   */  
  ImageTensor( int rows, int cols, int depth, int data_type, MemoryAlignment align = MEM_ALIGN_NONE );

  /**
   * @brief Allocates a new tensor with the size provided in input
   * 
   * @param [in] rows New number of rows
   * @param [in] cols New number of columns
   * @param [in] depth New number of depths (i.e., the number of images)
   * @param [in] data_type New matrix type, use cv::DataType< T >::type 
   * @param [in] align If not MEM_ALIGN_NONE, the data will be aligned at the requested boundary
   * 
   * Allocate a vector of depth cv::Mat, each one with size rows X cols and type provided by tha parameter data_type
   */
  void create( int rows, int cols, int depth, int data_type, MemoryAlignment align = MEM_ALIGN_NONE );
  
  /**
   * @brief Provide the number of rows of each matrix
   */
  int rows() const{ return rows_; };
  
  /**
   * @brief Provide the number of cols of each matrix
   */
  int cols() const{ return cols_; };

  /**
   * @brief Provide the depth of the tensor, i.e. the number of matrices that compose it
   */
  int depth() const{ return data_.size(); };
  
  /**
   * @brief Provide a reference to the i-th matrix, i.e. the i-th level of the tensor
   */  
  inline cv::Mat &operator[]( int i ){ return data_[i]; };

  /**
   * @brief Provide a reference to the i-th matrix, i.e. the i-th level of the tensor
   */    
  inline const cv::Mat &operator[]( int i ) const { return data_[i]; };

  /**
   * @brief Provide a reference to the i-th matrix, i.e. the i-th level of the tensor
   */  
  inline cv::Mat &at( int i ){ return data_[i]; };

  /**
   * @brief Provide a reference to the i-th matrix, i.e. the i-th level of the tensor
   */  
  inline const cv::Mat &at( int i ) const { return data_[i]; };
  
private:
    
  void releaseBuf();
  
  int rows_, cols_;
  std::vector< cv::Mat > data_;
  void *data_buf_;
};

typedef std::shared_ptr< ImageTensor > ImageTensorPtr;
}
