#include "cv_ext/aligned_mat.h"

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

class AlignedAllocator : public MatAllocator
{
public:
  
  AlignedAllocator(MemoryAlignment align) : align_(align){}; 
  
  // Code modified from OpenCV cv::StdMatAllocator
  UMatData* allocate ( int dims, const int* sizes, int type,
                       void* data0, size_t* step, int /*flags*/, UMatUsageFlags /*usageFlags*/ ) const
  {
    
    size_t total = CV_ELEM_SIZE ( type );
    for ( int i = dims-1; i >= 0; i-- )
    {
      if ( step )
      {
        if ( data0 && step[i] != CV_AUTOSTEP )
        {
          CV_Assert ( total <= step[i] );
          total = step[i];
        }
        else
        {
          step[i] = total;
        }
      }
      total *= sizes[i];
      // Align "rows"
      if(i == dims - 1)
      {
        if( total%align_ )
        {
          total += align_ - total%align_;
        }
      }
    }
    uchar* data = data0 ? ( uchar* ) data0 : ( uchar* ) ALIGNED_MALLOC (total , (size_t)align_ );
    UMatData* u = new UMatData ( this );
    u->data = u->origdata = data;
    u->size = total;
    if ( data0 )
    {
      u->flags |= UMatData::USER_ALLOCATED;
    }

    return u;
  }

  bool allocate ( UMatData* u, int /*accessFlags*/, UMatUsageFlags /*usageFlags*/ ) const
  {
    if ( !u )
    {
      return false;
    }
    return true;
  }

  void deallocate ( UMatData* u ) const
  {
    if ( !u )
    {
      return;
    }

    CV_Assert ( u->urefcount == 0 );
    CV_Assert ( u->refcount == 0 );
    if ( ! ( u->flags & UMatData::USER_ALLOCATED ) )
    {
      ALIGNED_FREE ( u->origdata );
      u->origdata = 0;
    }
    delete u;
  }
private:
  
  MemoryAlignment align_;
  
};

static AlignedAllocator& getAlignedAllocatorNone()
{
  static AlignedAllocator instance(MEM_ALIGN_NONE);
  return instance;
}

static AlignedAllocator& getAlignedAllocator16()
{
  static AlignedAllocator instance(MEM_ALIGN_16);
  return instance;
}
static AlignedAllocator& getAlignedAllocator32()
{
  static AlignedAllocator instance(MEM_ALIGN_32);
  return instance;
}
static AlignedAllocator& getAlignedAllocator64()
{
  static AlignedAllocator instance(MEM_ALIGN_64);
  return instance;
}

static AlignedAllocator& getAlignedAllocator128()
{
  static AlignedAllocator instance(MEM_ALIGN_128);
  return instance;
}
static AlignedAllocator& getAlignedAllocator256()
{
  static AlignedAllocator instance(MEM_ALIGN_256);
  return instance;
}
static AlignedAllocator& getAlignedAllocator512()
{
  static AlignedAllocator instance(MEM_ALIGN_512);
  return instance;
}

AlignedMat::AlignedMat( MemoryAlignment align ) : Mat ()
{
  setAllocator( align );
}

AlignedMat::AlignedMat ( int rows, int cols, int data_type, MemoryAlignment align ) : Mat ()
{
  setAllocator( align );
  create(cv::Size(cols,rows), data_type );
}

AlignedMat::AlignedMat ( Size size, int data_type, MemoryAlignment align ) : Mat ()
{
  setAllocator( align );
  create(size, data_type);
}

AlignedMat::AlignedMat ( int rows, int cols, int data_type, const Scalar& s, MemoryAlignment align ) : Mat ()
{
  setAllocator( align );
  create(cv::Size(cols,rows), data_type );
  this->Mat::operator=(s);
}

AlignedMat::AlignedMat ( Size size, int data_type, const Scalar& s, MemoryAlignment align ) : Mat ()
{
  setAllocator( align );
  create(size, data_type);
  this->Mat::operator=(s);
}

void AlignedMat::setAllocator( MemoryAlignment align )
{
  switch(align)
  {
    case MEM_ALIGN_NONE:
      allocator = &getAlignedAllocatorNone();
      break;
    case MEM_ALIGN_16:
      allocator = &getAlignedAllocator16();
      break;
    case MEM_ALIGN_32:
      allocator = &getAlignedAllocator32();
      break;
    case MEM_ALIGN_64:
      allocator = &getAlignedAllocator64();
      break;
    case MEM_ALIGN_128:
      allocator = &getAlignedAllocator128();
      break;
    case MEM_ALIGN_256:
      allocator = &getAlignedAllocator256();
      break;
    case MEM_ALIGN_512:
      allocator = &getAlignedAllocator512();
      break;
    default:
      allocator = &getAlignedAllocatorNone();
      break;
  }
}
