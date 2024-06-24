//#include <opencv2/opencv.hpp>
//using namespace cv;
//using namespace std;
//
//int main(int argc, char** argv)
//{
//	Mat src, gray;
//
//	//String filename = "C:\\Users\\skrua\\source\\repos\\DLIP\\Image\\EdgeLineImages\\pillsetc.png";
//	//const char* filename = "C:\\Users\\skrua\\source\\repos\\DLIP\\Image\\EdgeLineImages\\TrafficSign1.png";
//	//const char* filename = "C:\\Users\\skrua\\source\\repos\\DLIP\\Image\\EdgeLineImages\\coins.png";
//	const char* filename = "C:\\Users\\skrua\\source\\repos\\DLIP\\Image\\EdgeLineImages\\eyepupil.png";
//
//
//	/* Read the image */
//	src = imread(filename, 1);
//
//	if (!src.data)
//	{
//		printf(" Error opening image\n");
//		return -1;
//	}
//
//	cvtColor(src, gray, COLOR_BGR2GRAY);
//	
//	int si = 4;
//	int lowThreshold;
//
//
//	/* smooth it, otherwise a lot of false circles may be detected */
//	GaussianBlur(gray, gray, Size(5, 5), si, si); // �ڿ� 2, 2�� �̹����� �󸶳� �帴���������� ����(Ŭ���� Ŀ�� �߽ɿ��� �� �ָ� �ִ� �ȼ��� ���� ��ħ -> �帲 ȿ��)
//
//	vector<Vec3f> circles;
//	//HoughCircles(gray, circles, 3, 1, gray.rows / 6, 100, 40);
//	HoughCircles(gray, circles, 3, 1, gray.rows / 50, 100, 40);
//	for (size_t i = 0; i < circles.size(); i++)
//	{
//		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
//		int radius = cvRound(circles[i][2]);
//
//		/* draw the circle center */
//		circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
//
//		/* draw the circle outline */
//		circle(src, center, radius, Scalar(255, 0, 0), 3, 8, 0);
//	}
//
//	resize(src, src, Size(500, 500));
//	imshow("circles", src);
//
//	/* Wait and Exit */
//	waitKey();
//	return 0;
//}