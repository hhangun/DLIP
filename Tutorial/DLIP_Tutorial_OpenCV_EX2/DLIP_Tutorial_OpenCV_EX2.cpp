#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


int main()
{
    // Load the image in gray-scale
    Mat src = imread("../../../Image/HGU_logo.jpg", 0);
    
    if (src.empty())
    {
        cout << "Error: Couldn't open the image." << endl;
        return -1;
    }
    
    // Calculate the sum of pixel intensities using 'at' function
    double sumIntensity = 0.0;
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            sumIntensity += src.at<uchar>(i, j);
        }
    }

    // Calculate the total number of pixels in the image
    int pixelCount = src.rows * src.cols;

        // Calculate the average intensity of the image
    double avgIntensity = sumIntensity / pixelCount;

        // Print the results
        cout << "Sum of intensity: " << sumIntensity << endl;
        cout << "Number of pixels: " << pixelCount << endl;
        cout << "Average intensity: " << avgIntensity << endl;

        flip(src, src, 1);
    // Display the gray-scale image
    imshow("src", src);
    waitKey(0);

    return 0;

    
    
    /////////////////////////// Video ///////////////////////////

    //VideoCapture cap(0);

    //if (!cap.isOpened()) {
    //    cout << "Cannot open the video camera\n";
    //    return -1;
    //}

    //namedWindow("Myvideo", WINDOW_NORMAL);

    //bool flipHorizontal = false;

    //while (true)
    //{
    //    Mat frame;

    //    // Read a new frame from the video feed
    //    bool readSuccess = cap.read(frame);

    //    // Check if reading the frame was successful
    //    if (!readSuccess)
    //    {
    //        cout << "Cannot find a frame from the video stream\n";
    //        break;
    //    }

    //    flip(frame, frame, 1);

    //    // Display the frame in the "MyVideo" window
    //    imshow("MyVideo", frame);

    //    // Wait for 30ms and check if the 'ESC' key is pressed
    //    if (waitKey(30) == 27)
    //    {
    //        cout << "ESC key is pressed by the user\n";
    //        break;
    //    }
    //}
    //return 0;
}
