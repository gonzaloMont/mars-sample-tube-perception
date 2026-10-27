#pragma once

#include <string>
#include <opencv2/opencv.hpp>

namespace cv_ext
{
 
enum RotationFormat{ROT_FORMAT_MATRIX, ROT_FORMAT_AXIS_ANGLE};

/** @brief Check if a filename has .yaml or .yml extension, lowercase or uppercase. 
 *         Otherwise, it adds a ".yml" extension 
 * 
 * @param[in] filename A input string
 * 
 * @return The input string, with added the ".yml" extension if necessary
 */
std::string generateYAMLFilename( const std::string &filename );

/** @brief Read from file a 3D rigid body transformation, i.e. a rotation and a translation
 * 
 * @param[out] rotation A 3x1 rotation vector or a 3x3 rotation matrix, depending on the rf parameter
 * @param[out] translation A 3x1 translation vector 
 * @param[in] rf Rotation format
 * 
 * If rf is set to ROT_FORMAT_AXIS_ANGLE (default), the rotation will returned by means a 3x1 rotation 
 * vector (in Axis-Angle representation), while if rf is set to ROT_FORMAT_MATRIX the rotation will 
 * returned by means a 3x3 rotation matrix
 */
void read3DTransf( std::string filename, cv::Mat &rotation, cv::Mat &translation, 
                   RotationFormat rf = ROT_FORMAT_AXIS_ANGLE );

/** @brief Write to file a 3D rigid body transformation, i.e. a rotation and a translation
 * 
 * @param[in] rotation A 3x1 or a 1x3 rotation vector, or a 3x3 rotation matrix
 * @param[in] translation A 3x1 or a 1x3 translation vector 
 */
void write3DTransf( std::string filename, const cv::Mat &rotation, const cv::Mat &translation );

}