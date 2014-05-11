#include <iostream>
#include "ActionHOGUtils.h"
#include "ActionHOGLibs.h"

using std::cout;
using std::cerr;

ActionHOG::ActionHOG(string detName, string featChan, int imgNGrids, int imgNBins,
					 int mhiNGrids, int mhiNBins, int optNGrids, int optNBins, bool flag) {
	det = detName;
	
	chan = featChan;
	imgflag = mhiflag = optflag = true;
	
	vis = flag;

	imgngs = imgNGrids; imgnbs = imgNBins;
	mhings = mhiNGrids; mhinbs = mhiNBins;
	optngs = optNGrids; optnbs = optNBins;

	imgHOGDims = imgngs*imgngs*imgnbs;
	mhiHOGDims = mhings*mhings*mhinbs;
	optHOGDims = optngs*optngs*optnbs;
	
	fp = NULL;
}


ActionHOG::~ActionHOG() {
	vid.release();

	fclose(fp); fp = NULL;
}


int ActionHOG::check(string vidFileName, string featFileName) {
	if (!vid.open(vidFileName)) {
		cerr << "Cannot open the video file: " << vidFileName << "!\n";
		system("pause");
		exit(0);
	}

	fp = fopen(featFileName.c_str(), "w");
	if (fp == NULL) {
		cerr << "Cannot open the feature file" << featFileName << "!\n";
		system("pause");
		exit(0);
	}

	setVidProp( (int)vid.get(CV_CAP_PROP_FRAME_COUNT),
			    (int)vid.get(CV_CAP_PROP_FRAME_HEIGHT),
			    (int)vid.get(CV_CAP_PROP_FRAME_WIDTH) );

	writeHeader();

	return 1;
}


int ActionHOG::writeHeader() {
	fprintf(fp, "# frameWidth: %d frameHeight: %d frameNumber: %d\n", width, height, nframes);
	fprintf(fp, "# detector: %s channel: %s\n", det.c_str(), chan.c_str());

	if (chan == "IMG") {
		mhiflag = optflag = false;
		fprintf(fp, "# t x y s imgHOG(%d)\n", imgHOGDims);
	} else if (chan == "MHI") {
		imgflag = optflag = false;
		fprintf(fp, "# t x y s mhiHOG(%d)\n", mhiHOGDims);
	} else if (chan == "OPT") {
		imgflag = mhiflag = false;
		fprintf(fp, "# t x y s optHOG(%d)\n", optHOGDims);
	} else if (chan == "IMG_MHI") {
		optflag = false;
		fprintf(fp, "# t x y s imgHOG(%d) mhiHOG(%d)\n", imgHOGDims, mhiHOGDims);
	} else if (chan == "IMG_OPT") {
		mhiflag = false;
		fprintf(fp, "# t x y s imgHOG(%d) optHOG(%d)\n", imgHOGDims, optHOGDims);
	} else if (chan == "MHI_OPT") {
		imgflag = false;
		fprintf(fp, "# t x y s mhiHOG(%d) optHOG(%d)\n", mhiHOGDims, optHOGDims);
	} else if (chan == "IMG_MHI_OPT") {
		fprintf(fp, "# t x y s imgHOG(%d) mhiHOG(%d) optHOG(%d)\n", imgHOGDims, mhiHOGDims, optHOGDims);	
	} else {
		cerr << "Feature channel: " << chan << " is not available!\n";
		system("pause");
		exit(0);
	}

	return 1;
}


int ActionHOG::setVidProp(int f, int h, int w) {
	nframes = f;
	height = h;
	width = w;

	return 1;
}


int ActionHOG::comp() {
	// source, preceding, and current frames
	Mat src, pre, cur;
	// motion history images with different data types
	Mat mhi8U, mhi32F; 
	// number of detected points
	int count = 0;
	// time counter at starting point
	double stimer = (double)getTickCount();

	// do the job
	for (int i = 0; i < nframes; ++i) {
		// average speed and total number of points in every 100 frames
		if (i % 100 == 0) {
			double timer = (getTickCount() - stimer) / getTickFrequency();
			cout << "at frame: " << i << " -> speed: avg fps = " << i / timer << " -> points: total num = " << count << "\n";
		}

		if (!vid.read(src))
			continue;
		
		if (src.channels() == 3)
			cvtColor(src, cur, CV_RGB2GRAY);
		else
			src.copyTo(cur);

		if (pre.empty())
			cur.copyTo(pre);

		vector<KeyPoint> srcKeys;
		detKeys(pre, srcKeys);

		// if no point detected in a certain frame
		if (srcKeys.size() == 0) {
			cur.copyTo(pre);
			getMotionHistoryImage(i, pre, cur, mhi8U, mhi32F);

			continue;
		}

		// update MHI
		getMotionHistoryImage(i, pre, cur, mhi8U, mhi32F);

		// filter interest points by MHI and OPT
		vector<KeyPoint> dstKeys;
		filterKeysByMotion(srcKeys, dstKeys, mhi8U, pre, cur, src);

		// update the total number
		count += (int)dstKeys.size();
		
		// compute HOG for image channel
		if (imgflag)
			getImageHOG(pre, dstKeys, imgHOG);

		// compute HOG for MHI channel
		if (mhiflag)
			getMotionHistoryImageHOG(mhi8U, dstKeys, mhiHOG);

		// compute HOG for OPT channel
		if (optflag)
			getOpticalFlowHOG(pre, cur, dstKeys, optHOG);

		// update preceding frame
		cur.copyTo(pre);

		// write keys and descriptors
		writeKeyDesc(i, dstKeys);
	}

	return 1;
}


int ActionHOG::detKeys(const Mat &img, vector<KeyPoint> &keys) {
	// SURF detector with no orientation normalization
	if (det == "SURF") {
		double hessThresh = 400.0;
		int nOctaves = 3;
		int nLayers = 4;
		bool extended = false;
		bool upright = true;
		SURF detector(hessThresh, nOctaves, nLayers, extended, upright);
		Mat mask = Mat::ones(img.rows, img.cols, CV_8UC1);
		detector(img, mask, keys);

	} else {
		cerr << det << " is not an available detector!\n";
		system("pause");
		exit(0);
	}

	return 1;	
}


int ActionHOG::getMotionHistoryImage(int idx, const Mat &pre, const Mat &cur, Mat &mhi8U, Mat &mhi32F) {
	if (idx == 0) 
		mhi32F = Mat::zeros(pre.rows, pre.cols, CV_32FC1);

	Mat diff = abs(cur - pre);

	threshold(diff, diff, MHI_DIFF_THRESH, 1, THRESH_BINARY); 

	double timestamp = (double)idx;
	updateMotionHistory(diff, mhi32F, timestamp, MHI_DURATION);

	if (timestamp < MHI_DURATION)
		mhi32F.convertTo(mhi8U, CV_8UC1, 255.0/MHI_DURATION, (timestamp - MHI_DURATION)*255.0/MHI_DURATION);
	else
		mhi32F.convertTo(mhi8U, CV_8UC1, 255.0/MHI_DURATION, (MHI_DURATION - timestamp)*255.0/MHI_DURATION);

	Mat mei(mhi8U.rows, mhi8U.cols, CV_8UC1);
	threshold(mhi8U, mei, MHI_MORPH_THRESH, 255, THRESH_BINARY);
	erode(mei, mei, Mat(), Point(-1, -1), 1);
	dilate(mei, mei, Mat(), Point(-1, -1), 6);

	Mat temp;
	mhi8U.copyTo(temp, mei);
	mhi8U = temp;

	if (vis) {
		imshow("MHI", mhi8U);
		waitKey(fdur);
	}

	return 1;
}


int ActionHOG::filterKeysByMotion(const vector<KeyPoint> &srcKeys, vector<KeyPoint> &dstKeys, const Mat &mhi,
								  const Mat &pre, const Mat &cur, const Mat &src) {
	// filter keys by MHI
	Mat mei(mhi.rows, mhi.cols, CV_8UC1);
	threshold(mhi, mei, MHI_MIN_THRESH, 255, THRESH_BINARY);

	int nSrcKeys = (int)srcKeys.size();
	vector<KeyPoint> tempKeys;

	for (int i = 0; i < nSrcKeys; ++i) {
		int ix = (int)(srcKeys[i].pt.x + 0.5);
		int iy = (int)(srcKeys[i].pt.y + 0.5);
		
		if (mei.at<uchar>(iy, ix) == 0)
			continue;
		
		tempKeys.push_back(srcKeys[i]);
	}

	// filter keys by OPT
	int nTempKeys = tempKeys.size();
	vector<Point2f> preKeys(nTempKeys);

	for (int i = 0; i < nTempKeys; ++i)
		preKeys[i] = tempKeys[i].pt;

	// compute optical flows of those keys preserved after MHI filtering
	vector<float> err;
	vector<uchar> status;
	vector<Point2f> curKeys;

	// only for visualization
	vector<Point2f> optPre;
	vector<Point2f> optCur;


	if (nTempKeys > 0)
		calcOpticalFlowPyrLK(pre, cur, preKeys, curKeys, status, err);
	
	for (int i = 0; i < nTempKeys; ++i) {
		if (status[i] == 0 || err[i] > 50)
			continue;

		Point2f p1 = Point2f(preKeys[i].x, preKeys[i].y);
		Point2f p2 = Point2f(curKeys[i].x, curKeys[i].y);
		double dist = std::sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));

		if (dist < OPT_MIN_THRESH || dist > OPT_MAX_THRESH)
			continue;

		// keep optical flow vector for visualization
		optPre.push_back(p1);
		optCur.push_back(p2);

		// preserve keys with sufficient motions
		dstKeys.push_back(tempKeys[i]);
	}

	// visualize keys
	if (vis) {
		Mat imgDstKeys = src.clone();
		int nDstKeys = dstKeys.size();
		
		for (int i = 0; i < nDstKeys; ++i) {
			// draw circle
			circle(imgDstKeys, dstKeys[i].pt, (int)dstKeys[i].size/2, Scalar(0, 0, 255), 2);
			
			// draw arrow line
			line(imgDstKeys, optPre[i], optCur[i], Scalar(0, 255, 0));
			
			double dx = optCur[i].x - optPre[i].x;
			double dy = optCur[i].y - optPre[i].y;
			double len = 0.3 * std::sqrt(dx*dx + dy*dy);
			double ang = atan2(optPre[i].y - optCur[i].y, optPre[i].x - optCur[i].x);
			Point2f temp;
			
			temp.x = (float)(optCur[i].x + len * std::cos(ang + 3.1416/4));
			temp.y = (float)(optCur[i].y + len * std::sin(ang + 3.1416/4));
			line(imgDstKeys, temp, optCur[i], Scalar(0, 255, 0));

			temp.x = (float)(optCur[i].x + len * std::cos(ang - 3.1416/4));
			temp.y = (float)(optCur[i].y + len * std::sin(ang - 3.1416/4));
			line(imgDstKeys, temp, optCur[i], Scalar(0, 255, 0));
		}

		imshow("keys", imgDstKeys);
		waitKey(fdur);
	}

	return 1;
}


int ActionHOG::getImageHOG(const Mat &img, const vector<KeyPoint> &keys, Mat &hog) {
	Mat gradx, grady;
	getGradients(img, gradx, grady);

	int nKeys = keys.size();
	hog.create(nKeys, imgngs*imgngs*imgnbs, CV_32FC1);

	for (int i = 0; i < nKeys; ++i) {
		Mat desc;
		getHOGatKey(keys[i], gradx, grady, imgngs, imgnbs, desc);
		desc.copyTo(hog.row(i));
	}

	return 1;
}


int ActionHOG::getMotionHistoryImageHOG(const Mat &mhi, const vector<KeyPoint> &keys, Mat &hog) {
	Mat gradx, grady;
	getGradients(mhi, gradx, grady);

	int nKeys = keys.size();
	hog.create(nKeys, mhings*mhings*mhinbs, CV_32FC1);

	for (int i = 0; i < nKeys; ++i) {
		Mat desc;
		getHOGatKey(keys[i], gradx, grady, mhings, mhinbs, desc);
		desc.copyTo(hog.row(i));
	}

	return 1;
}

#ifdef OPT_PATCH
int ActionHOG::getOpticalFlowHOG(const Mat &pre, const Mat &cur, const vector<KeyPoint> &keys, Mat &hog) {
	int nRows = pre.rows;
	int nCols = pre.cols;

	int nKeys = keys.size();
	hog.create(nKeys, optngs*optngs*optnbs, CV_32FC1);

	for (int i = 0; i < nKeys; ++i) {
		int x = (int)(keys[i].pt.x + 0.5f);
		int y = (int)(keys[i].pt.y + 0.5f);
		int rad = (int)(keys[i].size / 2.0f + 0.5f);

		int x1 = x - rad; x1 = (x1 < 0) ? 0 : x1;
		int x2 = x + rad; x2 = (x2 > nCols) ? nCols : x2;
		int y1 = y - rad; y1 = (y1 < 0) ? 0 : y1;
		int y2 = y + rad; y2 = (y2 > nRows) ? nRows : y2;

		Mat preROI(pre, Range(y1, y2), Range(x1, x2));
		Mat curROI(cur, Range(y1, y2), Range(x1, x2));

		double scale  = 0.5;
		int levels = 1;
		int winsize = 5;
		int iterations = 10;
		int polyn = 5;
		double polysigma = 1.1;

		Mat opt;
		calcOpticalFlowFarneback(preROI, curROI, opt, scale, levels, winsize, iterations, polyn, polysigma, OPTFLOW_FARNEBACK_GAUSSIAN);

		Mat velx(opt.rows, opt.cols, CV_32FC1);
		Mat vely(opt.rows, opt.cols, CV_32FC1);

		for (int ir = 0; ir < opt.rows; ++ir) {
			float *ptro = opt.ptr<float>(ir);
			float *ptrx = velx.ptr<float>(ir);
			float *ptry = vely.ptr<float>(ir);

			for (int ic = 0; ic < opt.cols; ++ic) {
				ptrx[ic] = ptro[2*ic+0];
				ptry[ic] = ptro[2*ic+1];
			}
		}

		Mat desc;
		getHOGatPatch(velx, vely, optngs, optnbs, desc);
		desc.copyTo(hog.row(i));
	}	

	return 1;
}
#else
int ActionHOG::getOpticalFlowHOG(const Mat &pre, const Mat &cur, const vector<KeyPoint> &keys, Mat &hog) {
	int nRows = pre.rows;
	int nCols = pre.cols;
	int nKeys = keys.size();
	hog.create(nKeys, optngs*optngs*optnbs, CV_32FC1);

	double scale  = 0.5;
	int levels = 1;
	int winsize = 3;
	int iterations = 10;
	int polyn = 5;
	double polysigma = 1.1;

	Mat opt;
	calcOpticalFlowFarneback(pre, cur, opt, scale, levels, winsize, iterations,
							 polyn, polysigma, OPTFLOW_FARNEBACK_GAUSSIAN);

	Mat velx = Mat::zeros(opt.rows, opt.cols, CV_32FC1);
	Mat vely = Mat::zeros(opt.rows, opt.cols, CV_32FC1);

	for (int ir = 0; ir < opt.rows; ++ir) {
		float *ptro = opt.ptr<float>(ir);
		float *ptrx = velx.ptr<float>(ir);
		float *ptry = vely.ptr<float>(ir);

		for (int ic = 0; ic < opt.cols; ++ic) {
			float vx = ptro[2*ic+0];
			float vy = ptro[2*ic+1];
			float dist = vx*vx + vy*vy;

			if (dist > 1) {
				ptrx[ic] = vx;
				ptry[ic] = vy;
			}
		}
	}

	for (int i = 0; i < nKeys; ++i) {
		Mat desc;
		getHOGatKey(keys[i], velx, vely, optngs, optnbs, desc);
		desc.copyTo(hog.row(i));
	}

	return 1;
}
#endif


int ActionHOG::writeKeyDesc(int idx, const vector<KeyPoint> &keys) {
	int nKeys = keys.size();
	for (int i = 0; i < nKeys; ++i) {
		fprintf(fp, "%d ", idx);
		fprintf(fp, "%f %f %f ", keys[i].pt.x, keys[i].pt.y, keys[i].size);

		if (imgflag) {
			const float *pimg = imgHOG.ptr<float>(i);
			for (int j = 0; j < imgHOGDims; ++j)
				fprintf(fp, "%f ", pimg[j]);
		}

		if (mhiflag) {
			const float *pmhi = mhiHOG.ptr<float>(i);
			for (int j = 0; j < mhiHOGDims; ++j)
				fprintf(fp, "%f ", pmhi[j]);
		}

		if (optflag) {
			const float *popt = optHOG.ptr<float>(i);
			for (int j = 0; j < optHOGDims; ++j)
				fprintf(fp, "%f ", popt[j]);
		}

		fprintf(fp, "\n");
	}

	return 1;
}