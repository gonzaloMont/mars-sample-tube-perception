#include "cv_ext/cv_ext.h"

#include <iostream>

using namespace cv;
using namespace cv_ext;
using namespace std;

int main ( int argc, char**argv )
{
  int width, height;

  AlignedMat m16(MEM_ALIGN_16), m32(MEM_ALIGN_32), m64(MEM_ALIGN_64), m128(MEM_ALIGN_128), m256(MEM_ALIGN_256), m512(MEM_ALIGN_512);

  bool test_passed = true;
  const int num_tests = 100;
  
  for( int i = 0; i < num_tests; i++ )
  {
    width = rand()%4096, height = rand()%4096;
    
    if(!width) width++;
    if(!height) height++;
    
    m16.create(cv::Size(width,height),DataType<ushort>::type);
    m32.create(cv::Size(width,height),DataType<ushort>::type);
    m64.create(cv::Size(width,height),DataType<ushort>::type);
    m128.create(cv::Size(width,height),DataType<ushort>::type);
    m256.create(cv::Size(width,height),DataType<ushort>::type);
    m512.create(cv::Size(width,height),DataType<ushort>::type);  

    for(int r = 0; r < height; r++)
    {
      ushort *p16 = m16.ptr<ushort>(r);
      ushort *p32 = m32.ptr<ushort>(r);
      ushort *p64 = m64.ptr<ushort>(r);
      ushort *p128 = m128.ptr<ushort>(r);
      ushort *p256 = m256.ptr<ushort>(r);
      ushort *p512 = m512.ptr<ushort>(r);    

      if(uint64_t(p16)%16)
      {
        cout<<"Failed align 16: row "<<r<<" address "<<p16<<endl;
        test_passed = false;
      }
      
      if(uint64_t(p32)%32)
      {
        cout<<"Failed align 32: row "<<r<<" address "<<p32<<endl;
        test_passed = false;
      }

      if(uint64_t(p64)%64)
      {
        cout<<"Failed align 64: row "<<r<<" address "<<p64<<endl;
        test_passed = false;
      }
      
      if(uint64_t(p128)%128)
      {
        cout<<"Failed align 128: row "<<r<<" address "<<p128<<endl;
        test_passed = false;
      }
      
      if(uint64_t(p256)%256)
      {
        cout<<"Failed align 256: row "<<r<<" address "<<p256<<endl;
        test_passed = false;
      }

      if(uint64_t(p512)%512)
      {
        cout<<"Failed align 512: row "<<r<<" address "<<p512<<endl;
        test_passed = false;
      }
    }
    if(!test_passed)
      break;
  }
  
  if( !test_passed )
    std::cout<<"Warning!!! Test NOT passed!"<<std::endl;
  else
    std::cout<<"OK: Test passed"<<std::endl;    
}