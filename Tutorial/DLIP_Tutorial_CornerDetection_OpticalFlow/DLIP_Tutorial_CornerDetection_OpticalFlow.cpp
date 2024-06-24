//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc.hpp"
//#include <iostream>
//using namespace cv;
//using namespace std;
//Mat src, src_gray;
//int thresh = 200;
//int max_thresh = 255;
//const char* source_window = "Source image";
//const char* corners_window = "Corners detected";
//void cornerHarris_demo(int, void*);
//
//int main(int argc, char** argv)
//{
//    src = imread("C:\\Users\\skrua\\source\\repos\\DLIP\\Image\\cornerdetectionImg\\checkedPattern.png", 1);
//    if (src.empty())
//    {
//        cout << "Could not open or find the image!\n" << endl;
//        cout << "Usage: " << argv[0] << " <Input image>" << endl;
//        return -1;
//    }
//    cvtColor(src, src_gray, COLOR_BGR2GRAY);
//    namedWindow(source_window);
//    createTrackbar("Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo);
//    imshow(source_window, src);
//    cornerHarris_demo(0, 0);
//    waitKey(0);
//    return 0;
//}
//
//void cornerHarris_demo(int, void*)
//{
//    int blockSize = 2;
//    int apertureSize = 3;
//    double k = 0.04;
//    Mat dst = Mat::zeros(src.size(), CV_32FC1);
//    cornerHarris(src_gray, dst, blockSize, apertureSize, k);	// 0~1 사이로 결과 나옴
//    Mat dst_norm, dst_norm_scaled;
//    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());	// dst 행렬을 0~255 범위로 정규화함
//    convertScaleAbs(dst_norm, dst_norm_scaled);	// dst_norm을 8비트로 변환하여 표시 준비
//    for (int i = 0; i < dst_norm.rows; i++)
//    {
//        for (int j = 0; j < dst_norm.cols; j++)
//        {
//            if ((int)dst_norm.at<float>(i, j) > thresh)
//            {
//                circle(dst_norm_scaled, Point(j, i), 5, Scalar(0), 2, 8, 0);
//            }
//        }
//    }
//    namedWindow(corners_window);
//    imshow(corners_window, dst_norm_scaled);
//}





////////////////////////////////////////////////////////////////////////////////

 //Machine VIsion - YKKim 2016.11
 //OpticalFlow_Demo.cpp 
 //Optical flow demonstration
 //Modification of OpenCV tutorial code


#include "opencv.hpp" 
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

#define MOVIE	0
#define CAM		1
#define IMAGE	0


void printHelp()
{

	cout << "\n A simple Demo of Lukas-Kanade optical flow ,\n"
		"Using OpenCV version " << CV_VERSION << endl;
	cout << "\tESC - quit the program\n"
		"\tr - auto-initialize tracking\n"
		"\tc - delete all the points\n"
		"\tn - switch the \"night\" mode on/off\n" << endl;
}


int main()
{
	printHelp();
	Mat image;

#if MOVIE	
	VideoCapture cap;
	Mat frame;
	cap.open("road1.mp4");
	if (!cap.isOpened()) {
		cout << " Video not read \n";
		return 1;
	}
	//int rate=cap.get(CV_CAP_PROP_FPS);
	cap.read(frame);
#endif MOVIE

#if CAM
	VideoCapture cap;
	Mat frame;
	//VideoCapture capture;	
	cap.open(0);
	if (!cap.isOpened())
	{

		cout << "***Could not initialize capturing...***\n";
		cout << "Current parameter's value: \n";

		return -1;
	}
	cout << "Camera Read OK \n";
	cap >> frame;  // Read from cam

#endif CAM

#if IMAGE
	image = imread("Traffic1.jpg");
#endif IMAGE

	// Optical flow control parameters
	const int MAX_COUNT = 500;
	bool needToInit = false;
	bool nightMode = false;
	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);
	vector<Point2f> points[2], initialP;
	vector<uchar> status;
	vector<float> err;

	Mat gray, prevGray;
	bool bInitialize = true;

	for (;;)
	{

#if CAM | MOVIE
		cap >> frame;
		if (frame.empty())
			break;
		frame.copyTo(image);
#endif CAM

		cvtColor(image, gray, COLOR_BGR2GRAY);

		// Finding initial feature points to track 
		if (bInitialize)
		{
			goodFeaturesToTrack(gray, points[0], MAX_COUNT, 0.01, 10); // , Mat(), 3, 0, 0.04);
			cornerSubPix(gray, points[0], subPixWinSize, Size(-1, -1), termcrit);
			initialP = points[0];
			gray.copyTo(prevGray);
			bInitialize = false;
		}

		if (nightMode)
			image = Scalar::all(0);


		// run  optic flow 
		calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize, 3, termcrit, 0, 0.001);

		// draw tracked features on the image
		for (int i = 0; i < points[1].size(); i++)
		{

			//line(image, points[0][i], points[1][i], Scalar(255, 255, 0));
			line(image, initialP[i], points[1][i], Scalar(255, 255, 0));
			circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
		}

		// Save current values as the new prev values
		std::swap(points[1], points[0]);
		cv::swap(prevGray, gray);

		flip(image, image, 1);
		namedWindow("LK Demo", WINDOW_AUTOSIZE);
		imshow("LK Demo", image);

		char c = (char)waitKey(10);
		if (c == 27)
			break;
		switch (c)
		{
		case 'r':
			bInitialize = true;
			break;
		case 'c':
			points[0].clear();
			points[1].clear();
			bInitialize = true;
			break;
		case 'n':
			nightMode = !nightMode;
			break;
		}
	}

	return 0;
}