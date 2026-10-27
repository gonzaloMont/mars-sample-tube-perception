#include "cv_ext/image_tensor.h"

#if defined(__arm__) || defined(__aarch64__)

#include <cstdlib>
#define ALIGNED_MALLOC(size, alignment) aligned_alloc((size_t)(alignment), (size_t)(size))
#define ALIGNED_FREE(p) free((void *)(p))

#else

#include <mm_malloc.h> 
#define ALIGNED_MALLOC(size, alignment) _mm_malloc((size_t)(size), (size_t)(alignment)) 
#define ALIGNED_FREE(p) _mm_free((void *)(p)) 

#endif


using namespace cv;
using namespace cv_ext;

ImageTensor::ImageTensor ( int rows, int cols,  int depth, int data_type, MemoryAlignment align )
{
  create ( rows, cols, depth, data_type, align );
}

ImageTensor::ImageTensor() : rows_(0), cols_(0), data_buf_(nullptr)
{

}

ImageTensor::~ImageTensor() 
{
  releaseBuf();
}

void ImageTensor::releaseBuf()
{
  if( data_buf_ )
    ALIGNED_FREE( data_buf_ );
  data_buf_ = nullptr;
}

void ImageTensor::create ( int rows, int cols, int depth, int data_type, MemoryAlignment align )
{
  rows_ = rows;
  cols_ = cols;
  
  releaseBuf();
  data_.clear();
  data_.reserve( depth );

  if( align == MEM_ALIGN_NONE )
  {
    for( int i = 0; i < depth; i++ )
      data_.push_back( Mat(Size(cols_,rows_), data_type ) );
  }
  else
  {
    int elem_size = int(CV_ELEM_SIZE((data_type & CV_MAT_TYPE_MASK)));
    int vec_size = align/elem_size;
    int tensor_data_step = cols_ + ((cols_%vec_size) ? (vec_size - (cols_%vec_size)) : 0);
    int data_chunk_size = rows_ * tensor_data_step * elem_size;
    // Allocate just one memory portion for the whole vector of tensors
    data_buf_ = ALIGNED_MALLOC (data_chunk_size*depth, (size_t)align );
    char *tensor_data_chunk = (char *)data_buf_;
    
    for( int i = 0; i < depth; i++ )
    {
      data_.push_back( Mat( rows_ , cols_, data_type, tensor_data_chunk, tensor_data_step * elem_size ) ); 
      tensor_data_chunk += data_chunk_size;
    }
  }
}
