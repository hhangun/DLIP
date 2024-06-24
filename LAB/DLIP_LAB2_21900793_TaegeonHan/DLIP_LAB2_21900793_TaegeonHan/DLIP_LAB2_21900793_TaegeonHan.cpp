/*
 * DLIP_LAB2.cpp
 * author: Taegeon Han
 * Parter: Gyeonheal An
 * Date: 2024-04-28
 * DLIP_LAB2_Dimension Measurement with 2D camera
 * Measure rectangular objects whole dimension with a smartphone
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include "cameraParam.h"

using namespace std;
using namespace cv;

// Global variables for Morphology
int element_shape = MORPH_RECT;      // MORPH_RECT, MORPH_ELIPSE, MORPH_CROSS
int n = 3;
Mat element = getStructuringElement(element_shape, Size(n, n));

Mat src, src_gray, result, result2, result3;
Mat gray, thresh;

// Global variables to store the coordinates of the marked point
Point markedPoint(-1, -1);

vector<Point2f> clickedPoints;
vector<Point2f> secondClickedPoints;  // Store the second set of points
int pointSelectionStage = 1; // Track which set of points is being select
int selectedPointIndex = -1;  // Index of the point being dragged
Mat firstWarpedImage, secondWarpedImage;

// Function headers
void CallBackFunc(int event, int x, int y, int flags, void* userdata);
void updateWarpedImage();
void findBoundingRect(const Mat& image);
void onMouse(int event, int x, int y, int flags, void* userdata);


vector<vector<Point>> contours;
vector<Vec4i> hierarchy;
double scale = 1, scale2 = 1, scale3 = 1;
double sc;
int cnt = 0;
double real_sc;

int box1x;   
int box1y;   
int box2x;   
int box2y;   
int clickCount = 0;  // Variables that track the number of clicks


int main()
{
    // Calibration and read image file
    cameraParam param("lab2.xml");
    src = imread("angle3.jpg");
    result = param.undistort(src);

    if (src.empty()) {
        std::cout << "Could not open or find the image!" << endl;
        return -1;
    }
    namedWindow("Source Image", WINDOW_NORMAL);

    // Convert the image to Gray
    cvtColor(result, src_gray, COLOR_BGR2GRAY);

    setMouseCallback("Source Image", CallBackFunc, nullptr);



    imshow("Source Image", result);

    while (true) {
        char key = static_cast<char>(waitKey(0)); // Wait indefinitely for a user key press

        if (key == 'c' || key == 'C') {
            clickedPoints.clear();        // Clear first set of points
            firstWarpedImage.release();   // Release the image created by first warping
            secondClickedPoints.clear();  // Clear second set of points
            secondWarpedImage.release();  // Release the image created by second warping

            pointSelectionStage = 1;      // Reset the point selection stage to the first set

            ::imshow("Source Image", result);  // Redisplay the original image
        }

        if (key == 27)
            break;
    }

    waitKey(0);
    return 0;
}


void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
    vector<Point2f>* currentPoints = (pointSelectionStage == 1) ? &clickedPoints : &secondClickedPoints;

    if (event == EVENT_LBUTTONDOWN) {
        selectedPointIndex = -1;
        for (int i = 0; i < currentPoints->size(); ++i) {
            if (norm(Mat((*currentPoints)[i]), Mat(Point2f(x, y))) < 5) {
                selectedPointIndex = i;
                break;
            }
        }
        if (selectedPointIndex == -1 && currentPoints->size() < 4) {
            currentPoints->push_back(Point2f((float)x, (float)y));
            std::cout << "Point " << currentPoints->size() << ": (" << x << ", " << y << ")" << endl;
            if (currentPoints->size() == 4 && pointSelectionStage == 1) {
                updateWarpedImage();
                pointSelectionStage = 2; // Move to second stage after first set of points
            }

        }
    }
    else if (event == EVENT_MOUSEMOVE && selectedPointIndex != -1) {
        (*currentPoints)[selectedPointIndex] = Point2f(x, y);
    }
    else if (event == EVENT_LBUTTONUP) {
        selectedPointIndex = -1;
    }


    // Redraw all points
    result2 = result.clone();
    for (size_t i = 0; i < clickedPoints.size(); ++i) {
        circle(result2, clickedPoints[i], 5, Scalar(0, 0, 255), FILLED);
        if (i > 0) line(result2, clickedPoints[i - 1], clickedPoints[i], Scalar(255, 0, 0), 2);
        if (clickedPoints.size() == 4) line(result2, clickedPoints[3], clickedPoints[0], Scalar(255, 0, 0), 2);
    }

    imshow("Source Image", result2);
}


void updateWarpedImage() {
    if (clickedPoints.size() == 4) {
        int W = 500, H = 400;  // Arbitrary dimensions
        vector<Point2f> dstCorners = { Point2f(0, 0), Point2f(W - 1, 0), Point2f(W - 1, H - 1), Point2f(0, H - 1) };
        Mat transformMatrix = getPerspectiveTransform(clickedPoints, dstCorners);
        warpPerspective(result, firstWarpedImage, transformMatrix, Size(W, H));
        imshow("firstWarpedImage", firstWarpedImage);
        findBoundingRect(firstWarpedImage);
    }
}


void findBoundingRect(const Mat& image) {

    result3 = image.clone();

    // change to grayscale
    cvtColor(image, gray, COLOR_BGR2GRAY);
    threshold(gray, thresh, 220, 255, THRESH_BINARY);
    // Finding contours
    findContours(thresh, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++) {

        Rect boundingBox = boundingRect(contours[i]);

        // Restrict of boundingBox
        if (boundingBox.width < 40)
            continue;

        rectangle(firstWarpedImage, boundingBox, Scalar(0, 255, 0), 2); // Draw a bounding box in green
        imshow("firstWarpedImage", firstWarpedImage);    
    }

    if (clickCount == 0) setMouseCallback("firstWarpedImage", onMouse, nullptr);
}


// Mouse callback function
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        // Specify coordinates based on the number of clicks
        switch (clickCount) {
        case 0:
            box1x = x;
            box1y = y;
            break;
        case 1:
            box2x = x;
            box2y = y;
            break;
        }
        clickCount++;  // Increase click count

        if (clickCount == 2) {
            cout << "Box1 Coordinates: (" << box1x << ", " << box1y << ")" << endl;
            cout << "Box2 Coordinates: (" << box2x << ", " << box2y << ")" << endl;
            for (size_t i = 0; i < contours.size(); i++) {

                Rect boundingBox = boundingRect(contours[i]);

                // Restrict of boundingBox
                if (boundingBox.width < 40)
                    continue;

                rectangle(firstWarpedImage, boundingBox, Scalar(0, 255, 0), 2); // Draw a bounding box in green

                if (boundingBox.width > 45 && cnt == 0) {
                    // Calculate corner points of the bounding box
                    Point2f topLeft(boundingBox.x, boundingBox.y);
                    Point2f topRight(boundingBox.x + boundingBox.width, boundingBox.y);
                    Point2f bottomRight(boundingBox.x + boundingBox.width, boundingBox.y + boundingBox.height);
                    Point2f bottomLeft(boundingBox.x, boundingBox.y + boundingBox.height);

                    double knownWidth = 50.0;  // Desired width in actual units
                    double knownLength = 50.0;  // Desired length in actual units
                    double knwonHeight = 50.0;  // Desired height in actual units

                    scale = knownWidth / boundingBox.width;  // Calculate scale based on the first contour
                    scale2 = knownLength / boundingBox.height;  // Calculate scale based on the first contour
                    scale3 = knwonHeight / (sqrt(pow(box1x - bottomLeft.x, 2) + pow(box1y - bottomLeft.y, 2)));
                    cnt++;

                    line(result3, bottomLeft, bottomRight, Scalar(0, 255, 0), 2);
                    line(result3, topLeft, bottomLeft, Scalar(0, 255, 0), 2);
                    line(result3, bottomLeft, Point(box1x, box1y), Scalar(0, 255, 255), 2);

                    // Calculate the size text
                    string sizeText_W = to_string(static_cast<int>(ceil(boundingBox.width * scale)));
                    string sizeText_L = to_string(static_cast<int>(ceil(boundingBox.height * scale2)));
                    string sizeText_H = to_string(static_cast<int>(ceil(sqrt(pow(box1x - bottomLeft.x, 2) + pow(box1y - bottomLeft.y, 2)) * scale3)));

                    // Add the size text next to the lines
                    putText(result3, sizeText_W, bottomLeft - Point2f(-20, 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1.5);
                    putText(result3, sizeText_L, topLeft - Point2f(-5, -40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1.5);
                    putText(result3, sizeText_H, bottomLeft - Point2f(30, -30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1.5);

                }
                else {

                    // Calculate corner points of the bounding box
                    Point2f topLeft2(boundingBox.x, boundingBox.y);
                    Point2f topRight2(boundingBox.x + boundingBox.width, boundingBox.y);
                    Point2f bottomRight2(boundingBox.x + boundingBox.width, boundingBox.y + boundingBox.height);
                    Point2f bottomLeft2(boundingBox.x, boundingBox.y + boundingBox.height);

                    double distance = sqrt(pow(box2x - bottomRight2.x, 2) + pow(box2y - bottomRight2.y, 2));
                    double realWidth = 100.0;
                    sc = realWidth / boundingBox.width; // scale of object about width
                    real_sc = sc / scale;   // scale between reference object and object that we wanted to calculate

                    line(result3, topRight2, bottomRight2, Scalar(0, 255, 0), 2);
                    line(result3, bottomRight2, bottomLeft2, Scalar(0, 255, 0), 2);
                    line(result3, bottomRight2, Point(box2x, box2y), Scalar(0, 255, 255), 2);

                    // Calculate the size text
                    double actualWidth = boundingBox.width * scale * real_sc;
                    double actualLength = boundingBox.height * scale2 * real_sc;
                    double actualHeight = distance * scale3 * real_sc;

                    string sizeText2_W = to_string(static_cast<int>(ceil(actualWidth)));
                    string sizeText2_L = to_string(static_cast<int>(ceil(actualLength)));
                    string sizeText2_H = to_string(static_cast<int>(ceil(actualHeight)));

                    // Add the size text next to the lines
                    putText(result3, sizeText2_W, bottomRight2 - Point2f(90, -20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1.5);
                    putText(result3, sizeText2_L, topRight2 - Point2f(-10, -80), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1.5);
                    putText(result3, sizeText2_H, bottomRight2 - Point2f(25, -40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1.5);
                }

                resize(result3, result3, Size(500, 400));
                imshow("Result", result3);
            }
            // Initialize click count (for additional clicks)
            clickCount = 0;
        }
    }

}
