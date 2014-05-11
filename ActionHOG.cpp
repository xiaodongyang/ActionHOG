#include <iostream>
#include "ActionHOGLibs.h"

using std::cerr;

int main(int argc, char **argv) {
	// setup default parameters 
	string detName("SURF"); 		// interest point detector (no dominant orientations)
	string featChan("IMG_MHI_OPT"); // feature channels
	bool vis = false; 				// no visualization

	int imgNGrids = 3; // number of grids in x and y for image channel
	int imgNBins = 8;  // number of bins to quantize gradient orientation for image channel

	int mhiNGrids = 3; // number of grids in x and y for MHI channel
	int mhiNBins = 8;  // number of bins to quantize gradient orientation for MHI channel
	
	int optNGrids = 3; // number of grids in x and y for optical flow channel
	int optNBins = 8;  // number of bins to quantize gradient orientation for optical flow channel

	// print command line help
	if (argc == 1) {
		cerr << "./ActionHOG -i vidFile -o featFile [options]\n"
			 << "  detector type                 -det SURF\n"
			 << "  feature channels              -chan IMG_MHI_OPT\n"
			 << "  visualization                 -vis\n"
			 << "  spatial bins num (img)        -nsIMG 3\n"
			 << "  orientation bins num (img)    -noIMG 8\n"
			 << "  spatial bins num (mhi)        -nsMHI 3\n"
			 << "  orientation bins num (mhi)    -noMHI 8\n"
			 << "  spatial bins num (opt)        -nsOPT 3\n"
			 << "  orientation bins num (opt)    -noOPT 8\n";
		return 0;
	}

	// read arguments
	string vidFileName, featFileName;
	int arg = 0;
	while (++arg < argc) {
		if (!strcmp(argv[arg], "-i"))
			vidFileName = argv[++arg];
		if (!strcmp(argv[arg], "-o"))
			featFileName = argv[++arg];
		if (!strcmp(argv[arg], "-det"))
			detName = argv[++arg];
		if (!strcmp(argv[arg], "-chan"))
			featChan = argv[++arg] ;
		if (!strcmp(argv[arg], "-vis"))
			vis = true;
		if (!strcmp(argv[arg], "-nsIMG"))
			imgNGrids = atoi(argv[++arg]);
		if (!strcmp(argv[arg], "-noIMG"))
			imgNBins = atoi(argv[++arg]);
		if (!strcmp(argv[arg], "-nsMHI"))
			mhiNGrids = atoi(argv[++arg]);
		if (!strcmp(argv[arg], "-noMHI"))
			mhiNBins = atoi(argv[++arg]);
		if (!strcmp(argv[arg], "-nsOPT"))
			optNGrids = atoi(argv[++arg]);
		if (!strcmp(argv[arg], "-noOPT"))
			optNBins = atoi(argv[++arg]);
	}

	//initialize ActionHOG
	ActionHOG feat( detName, featChan,
				    imgNGrids, imgNBins,
				    mhiNGrids, mhiNBins,
				    optNGrids, optNBins,
				    vis );

	// check video file and feature file
	feat.check(vidFileName, featFileName);

	// compute ActionHOG
	feat.comp();

	return 0;
}