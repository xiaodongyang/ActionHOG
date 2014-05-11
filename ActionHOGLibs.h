#ifndef ACTION_HOG_LIBS_H
#define ACTION_HOG_LIBS_H

#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"

using std::string;

using namespace cv;

// compute dense optical flow based on each patch or entire frame 
#define OPT_PATCH 
// minimum intensity difference to be considered as motion
const double MHI_DIFF_THRESH = 20.0;
// motion duration
const double MHI_DURATION = 20.0;
// minimum threshold to binarize before morphlogical operation
const double MHI_MORPH_THRESH = 100.0;
// minimum threshold for pixel intensity 
const double MHI_MIN_THRESH = 200.0;
// minimum threshold for pixel motion
const double OPT_MIN_THRESH = 1.0;
// maximum threshold for pixel motion
const double OPT_MAX_THRESH = 50.0;
// a frame duration (ms) in visualization
const int fdur = 20;

class ActionHOG {
public:
	ActionHOG(string detName, string featChan, int imgNGrids, int imgNBins,
			  int mhiNGrids, int mhiNBins, int optNGrids, int optNBins, bool flag);

	~ActionHOG();

	// compute ActionHOG feature
	int check(string vidFileName, string featFileName);

	int writeHeader();

	// set video properties
	int setVidProp(int f, int h, int w);

	// compute ActionHOG
	int comp();

	// detect interest points based on a detector
	int detKeys(const Mat &img, vector<KeyPoint> &keys);

	// compute MHI
	int getMotionHistoryImage(int idx, const Mat &pre, const Mat &cur, Mat &mhi8U, Mat &mhi32F);

	// remove static points with insufficient motion
	int filterKeysByMotion(const vector<KeyPoint> &srcKeys, vector<KeyPoint> &dstKeys,
						   const Mat &mhi, const Mat &pre, const Mat &cur, const Mat &src);

	// compute HOG on IMG channel
	int getImageHOG(const Mat &img, const vector<KeyPoint> &keys, Mat &hog);

	// compute HOG on MHI channel
	int getMotionHistoryImageHOG(const Mat &mhi, const vector<KeyPoint> &keys, Mat &hog);

	// compute HOG on OPT channel
	int getOpticalFlowHOG(const Mat &pre, const Mat &cur, const vector<KeyPoint> &keys, Mat &hog);

	// write points and descriptors
	int ActionHOG::writeKeyDesc(int idx, const vector<KeyPoint> &keys);

private:
	// video file
	VideoCapture vid;

	// feature file
	FILE *fp;

	// detector and channels
	string det;
	string chan;

	// channel flag
	bool imgflag, mhiflag, optflag;

	// visualization flag
	bool vis;

	// number of spatial grids of each channel
	int imgngs;
	int mhings;
	int optngs;

	// number of gradient orientation bins of each channel
	int imgnbs;
	int mhinbs;
	int optnbs;

	// frame size
	int nframes;
	int height;
	int width;

	// descriptor dimensions
	int imgHOGDims, mhiHOGDims, optHOGDims;

	// descriptors of each frame on different channels
	Mat imgHOG, mhiHOG, optHOG;
};

#endif