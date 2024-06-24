/*------------------------------------------------------/
* Image Proccessing with Deep Learning
* OpenCV : Filter Demo - Video
* Created: 2021-Spring
------------------------------------------------------*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <windows.h>

using namespace std;
using namespace cv;

int main()
{
	/*  open the video camera no.0  */
	VideoCapture cap(0);

	if (!cap.isOpened())	// if not success, exit the programm
	{
		cout << "Cannot open the video cam\n";
		return -1;
	}

	namedWindow("MyVideo", WINDOW_AUTOSIZE);

	int key = 0;
	int kernel_size = 11;
	int filter_type = 0;
	while (1)
	{
		Mat src, dst;

		/*  read a new frame from video  */
		bool bSuccess = cap.read(src);

		if (!bSuccess)	// if not success, break loop
		{
			cout << "Cannot find a frame from  video stream\n";
			break;
		}


		key = waitKey(30);
		if (key == 27) // wait for 'ESC' press for 30ms. If 'ESC' is pressed, break loop
		{
			cout << "ESC key is pressed by user\n";
			break;
		}
		else if (key == 'b' || key == 'B')
		{
			cout << "blur" << endl;
			filter_type = 1;	// blur
		}
		else if (key == 'G' || key == 'g')
		{
			cout << "Gaussian" << endl;
			filter_type = 2;	// Gaussian 
		}
		else if (key == 'M' || key == 'm')
		{
			cout << "Median" << endl;
			filter_type = 3;	// Median 
		}
		else if (key == 'O' || key == 'o')
		{
			cout << "Original" << endl;
			filter_type = 4;	// Original
		}
		else if (key == 'U' || key == 'u')
		{
			cout << "kernel up" << endl;
			kernel_size += 2;
		}
		else if (key == 'D' || key == 'd')
		{
			cout << "kernel down" << endl;
			kernel_size -= 2;
		}

		if (kernel_size < 3)
			kernel_size = 3;

		if (filter_type == 1)
			blur(src, dst, Size(kernel_size, kernel_size), Point(-1, -1));
		else if (filter_type == 2)
			GaussianBlur(src, dst, Size(kernel_size, kernel_size), 0, 0);
		else if (filter_type == 3)
			medianBlur(src, dst, kernel_size);
		else
			src.copyTo(dst);

		flip(dst, dst, 1);
		imshow("MyVideo", dst);
	}
	return 0;
}