// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <ctime>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d/features2d.hpp>
using namespace std;
using namespace cv;
using namespace xfeatures2d;

#ifdef USE_GPU
	#include <opencv2/cudafeatures2d.hpp>
	using cuda::GpuMat;
#endif

