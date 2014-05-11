#ifndef ACTION_HOG_UTILS_H
#define ACTION_HOG_UTILS_H

#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"

using std::string;
using namespace cv;

const float PI = 3.14159265f;

int getGradients(const Mat &img, Mat &gradx, Mat &grady);

int getHOGatKey(const KeyPoint &key, const Mat &gradx, const Mat &grady, int nGrids, int nBins, Mat &desc);

int getHOGatPatch(const Mat &gradx, const Mat &grady, int nGrids, int nBins, Mat &desc);

#endif