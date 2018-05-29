// GridMatch.cpp : Defines the entry point for the console application.

//#define USE_GPU 

#include "Header.h"
#include "gms_matcher.h"
#include "stdio.h"

void GmsMatch(Mat &img1, Mat &img2, Mat &result, Point2f &lastPoint, Point2f &curPoint);
void imrotate(Mat& src, double& angle, Mat& result);



void runImagePair(std::ofstream &out){
	//Mat img1 = imread("/home/amor/Desktop/image/地图序列/1.jpg");//0-left.jpg");
	char str[40];
	Point2f Point0(212, 458);
	Point2f lastPoint = Point0, curPoint = Point0;
	//0827和0930采集时比地图海拔要低，存在一个尺度差。
	Mat img2 = imread("/home/amor/Desktop/image/匹配图0827.jpg");//0-right.jpg");
	//cvtColor(img2,img2,CV_BGR2GRAY);
	
	//img2 = cv:rotate
//	imresize(img1, 480);
//	imresize(img2, 480);
	for( int i= 1;i < 1640;i+=10)
	{
		//Mat img1 = imread("/home/amor/Desktop/image/地图序列/%d.jpg",i);//0-left.jpg");
		Mat result;
		sprintf(str,"/home/amor/Desktop/image/飞行图像0930/%d.jpg",i);
		Mat img1 = imread(str);
		//cvtColor(img1,img1,CV_BGR2GRAY);
		
		
		double angle = 30;
		//imrotate(img1,angle,img1);
		int height = 360;
		//imresize(img1,240);
		lastPoint = curPoint;
		GmsMatch(img1, img2, result, lastPoint, curPoint);
	//	out<<curPoint.x<<"\t\t"<<curPoint.y<<"\t\t"<<endl;
	//	sprintf(str,"/home/amor/Desktop/XL/Visual_Navigation/result_all/Robust_method/%d.jpg",i);
	//	imwrite(str,result);
		
	}
}


int main()
{
#ifdef USE_GPU
	int flag = cuda::getCudaEnabledDeviceCount();
	if (flag != 0){ cuda::setDevice(0); }
#endif // USE_GPU
	ofstream out("position.txt");
	runImagePair(out);
	out.close();

	return 0;
}


void GmsMatch(Mat &img1, Mat &img2, Mat &result, Point2f &lastPoint, Point2f &curPoint){
	vector<KeyPoint> kp1, kp2;
	Mat d1, d2;
	vector<DMatch> matches_all, matches_gms;

	char str[40];
	/*Ptr<ORB> orb = ORB::create(10000);
	orb->setFastThreshold(0);
	orb->detectAndCompute(img1, Mat(), kp1, d1);
	orb->detectAndCompute(img2, Mat(), kp2, d2);*/
	//原始算法采用orb,改成sift看效果
	Ptr<FeatureDetector> surf = SURF::create(400);
	//surf->setFastThreshold(0);
	surf->detectAndCompute(img1, Mat(), kp1, d1);
	surf->detectAndCompute(img2, Mat(), kp2, d2);
	

#ifdef USE_GPU
	GpuMat gd1(d1), gd2(d2);
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	matcher->match(gd1, gd2, matches_all);
#else
	//BFMatcher matcher(NORM_HAMMING);
	BFMatcher matcher(NORM_L1);//surf operator
	matcher.match(d1, d2, matches_all);
#endif

	// GMS filter
	int num_inliers = 0;
	std::vector<bool> vbInliers;
	gms_matcher gms(kp1,img1.size(), kp2,img2.size(), matches_all);
	num_inliers = gms.GetInlierMask(vbInliers, false, false);

	cout << "Get total " << num_inliers << " matches." << endl;

	// draw matches
	for (size_t i = 0; i < vbInliers.size(); ++i)
	{
		if (vbInliers[i] == true)
		{
			matches_gms.push_back(matches_all[i]);
		}
	}

	curPoint = lastPoint;
	Mat show = DrawInlier(img1, img2, kp1, kp2, matches_gms, 1, curPoint);
	show.copyTo(result);
	imshow("show", show);
	
	waitKey(2);
}
void imrotate(Mat& src, double& angle, Mat& result)
{
	Point2f center = Point2f(src.cols/2, src.rows/2);
	Mat rot = getRotationMatrix2D(center,angle,1.0);
	Size2d size(src.rows,src.cols);
	warpAffine(src,result,rot,size);
}

