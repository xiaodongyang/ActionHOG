#include "ActionHOGUtils.h"

int getGradients(const Mat &img, Mat &gradx, Mat &grady) {
	Sobel(img, gradx, CV_32FC1, 1, 0, 3);
	Sobel(img, grady, CV_32FC1, 0, 1, 3);

	return 1;
}


int getHOGatKey(const KeyPoint &key, const Mat &gradx, const Mat &grady, int nGrids, int nBins, Mat &desc) {
	desc = Mat::zeros(1, nGrids*nGrids*nBins, CV_32FC1);

	int xkey = (int)(key.pt.x + 0.5);
	int ykey = (int)(key.pt.y + 0.5);

	int patchSize = (int)(key.size + 0.5);
	int patchSizeHalf = (int)(patchSize / 2.0 + 0.5);
	int nRows = gradx.rows; 
	int nCols = gradx.cols;

	int block = 0;
	float oriStep = 360.0f / nBins; 
	int x1, x2, y1, y2;
	float *ptrDesc = desc.ptr<float>(0);

	for (int ix = 1; ix <= nGrids; ++ix) {
		x1 = (int)(xkey - patchSizeHalf + patchSize*(float)(ix-1)/(float)nGrids);
		x1 = x1 < 0 ? 0 : x1;

		x2 = (int)(xkey - patchSizeHalf + patchSize*(float)ix/(float)nGrids);
		x2 = x2 > (nCols-1) ? (nCols-1) : x2;

		for (int iy = 1; iy <= nGrids; ++iy) {
			y1 = (int)(ykey - patchSizeHalf + patchSize*(float)(iy-1)/(float)nGrids);
			y1 = y1 < 0 ? 0 : y1;

			y2 = (int)(ykey - patchSizeHalf + patchSize*(float)iy/(float)nGrids);
			y2 = y2 > (nRows-1) ? (nRows-1) : y2;

			for (int y = y1; y <= y2; ++y) {
				const float *ptrGradx = gradx.ptr<float>(y);
				const float *ptrGrady = grady.ptr<float>(y);

				for (int x = x1; x <= x2; ++x) {
					int bin = -1;
					float xval = ptrGradx[x];
					float yval = ptrGrady[x];
					float mag = std::sqrt(xval*xval + yval*yval);

					if (mag > 0) {
						float ori = std::acos(xval / mag) * 180.0f / PI;
						if (yval < 0) {ori = 360.0f - ori;}
						bin = (int)(((int)(ori + oriStep / 2.0f) % 360) / oriStep);
					}

					if (bin >= 0)
						ptrDesc[block*nBins + bin] += mag;
				}
			}

			float norm = 0.0f;
			for (int i = 0; i < nBins; ++i)
				norm += ptrDesc[block*nBins + i] * ptrDesc[block*nBins + i];
			
			norm = std::sqrt(norm);

			if (norm > 0) {
				for (int i = 0; i < nBins; ++i)
					ptrDesc[block*nBins + i] /= norm;
			}

			++block;
		}
	}

	return 1;
}


int getHOGatPatch(const Mat &gradx, const Mat &grady, int nGrids, int nBins, Mat &desc) {
	desc = Mat::zeros(1, nGrids*nGrids*nBins, CV_32FC1);

	int block = 0;
	float oriStep = 360.0f / nBins; 
	int yPatchSize = gradx.rows; 
	int xPatchSize = gradx.cols;
	int x1, x2, y1, y2;
	float *ptrDesc = desc.ptr<float>(0);

	for (int ix = 1; ix <= nGrids; ++ix) {
		x1 = (int)(xPatchSize*(float)(ix-1)/(float)nGrids);
		x2 = (int)(xPatchSize*(float)ix/(float)nGrids);

		for (int iy = 1; iy <= nGrids; ++iy) {
			y1 = (int)(yPatchSize*(float)(iy-1)/(float)nGrids);
			y2 = (int)(yPatchSize*(float)iy/(float)nGrids);

			for (int y = y1; y < y2; ++y) {
				const float *ptrGradx = gradx.ptr<float>(y);
				const float *ptrGrady = grady.ptr<float>(y);
				
				for (int x = x1; x < x2; ++x) {
					int bin = -1;
					float xval = ptrGradx[x];
					float yval = ptrGrady[x];
					float mag = std::sqrt(xval*xval + yval*yval);

					if (mag > 0) {
						float ori = std::acos(xval / mag) * 180.0f / PI;
						if (yval < 0) {ori = 360.0f - ori;}
						bin = (int)(((int)(ori+oriStep/2.0f) % 360) / oriStep);
					}

					if (bin >= 0)
						ptrDesc[block*nBins + bin] += mag;
				}
			}

			float norm = 0.0f;
			for (int i = 0; i < nBins; ++i)
				norm += ptrDesc[block*nBins + i];

			if (norm > 0)
				for (int i = 0; i < nBins; ++i)
					ptrDesc[block*nBins + i] /= norm;

			++block;
		}
	}

	return 1;
}