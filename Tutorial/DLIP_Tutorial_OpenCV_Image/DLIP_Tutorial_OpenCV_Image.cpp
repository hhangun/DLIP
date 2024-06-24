#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	/*  read image  */
	String HGU_logo = "C:/Users/skrua/source/repos/DLIP/Image/HGU_logo.jpg";

	Mat src = imread(HGU_logo);

	Mat src_gray = imread(HGU_logo, 0);  // read in grayscale

	// Check for failure
	if (src.empty())
	{
		cout << "Could not open or find the imagesssss" << endl;
		cin.get(); //wait for any key press
		return -1;
	}


	/*  write image  */
	//String fileName = "writeImage.jpg";
	//imwrite(fileName, src);

	/*  display image  */
	namedWindow("src", WINDOW_AUTOSIZE);
	imshow("src", src);

	namedWindow("src_gray", WINDOW_AUTOSIZE);
	imshow("src_gray", src_gray);

	waitKey(0);
}