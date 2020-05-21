#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\objdetect\objdetect.hpp>
#include <stdio.h>
#include <iostream>
#define sampleSize 10
#define frameWidth 540
#define frameHeight 360

using namespace cv;
using namespace std;

Mat image;
Mat imageHLS;
Mat imageBW;
Mat background;
Mat imageEdges;
Mat imageFiltered;
Mat imageCut;
Mat Mog_Mask;
Mat locationRectangle;
Ptr<BackgroundSubtractorMOG2> mog2;
VideoCapture cap;
int locationSize;
Scalar filter[sampleSize][2];
int counter;
int detectionWait;
vector<int> numbers;
vector<Point> fingers;


void setFlippedImage(VideoCapture cap, Mat image) {
	cap >> image;
	flip(image, image, 1);
	image(Rect(frameWidth / 3, 0, frameWidth + 100 - frameWidth / 3, frameHeight)).copyTo(imageCut);
}

float distance(Point x, Point y) {
	return sqrt((x.x - y.x) * (x.x - y.x) + (x.y - y.y) * (x.y - y.y));
}

float angle(Point x, Point y, Point z) {
	float dot = (x.x - y.x)*(z.x - y.x) + (x.y - y.y)*(z.y - y.y);
	float angle = acos(dot / (distance(x, y)*distance(y, z)));
	return angle * 180 / 3.14159;
}

int median(vector<int> vect) {
	int median;
	sort(vect.begin(), vect.end());
	return vect[vect.size() / 2];

}


void setFilter(int loc, Point checkLocation, int locationSize) {
	Mat rectangle;
	imageHLS(Rect(checkLocation.x, checkLocation.y, locationSize, locationSize)).copyTo(rectangle);
	int h = 0, s = 0, l = 0;
	vector<int> hVector, lVector, sVector;
	for (int i = 0; i < rectangle.rows; i++) {
		for (int j = 0; j < rectangle.cols; j++) {
			hVector.push_back(rectangle.data[rectangle.channels() * (rectangle.cols*i + j)]);
			lVector.push_back(rectangle.data[rectangle.channels() * (rectangle.cols*i + j) + 1]);
			sVector.push_back(rectangle.data[rectangle.channels() * (rectangle.cols*i + j) + 2]);
		}
	}
	h = median(hVector);
	l = median(lVector);
	s = median(sVector);

	filter[loc][0] = Scalar(h - 40, l - 60, s - 100);
	filter[loc][1] = Scalar(h + 40, l + 60, s + 100);
}

int getMostFrequent(vector<int> numbers) {
	int frequent, current;
	sort(numbers.begin(), numbers.end());
	int max = 0, count = 0;

	for (int i = 0; i < numbers.size(); i++) {
		if (i == 0) {
			current = numbers[i];
			frequent = current;
		}
		if (numbers[i] == current) { count++; }
		else {
			if (count > max) {
				max = count;
				frequent = current;
			}
			current = numbers[i];
			count = 1;
		}
	}
	return frequent;
}

int calculateNumber(Rect handRect) {
	if (handRect.height > imageCut.rows / 3) {
		numbers.push_back(fingers.size());
		if (counter > 15) {
			counter = 0;
			int mostFrequent = getMostFrequent(numbers);
			numbers.clear();
			return mostFrequent;
		}
		else {
			counter++;
			return -1;
		}
	}
	else {
		return -1;
	}
}

void calculateFingers(vector<vector<Vec4i>> defects, int handContour, vector<vector<Point>> contours, Rect handRect, vector<vector<Point>> points) {
	fingers.clear();
	int i = 0;
	while (i < defects[handContour].size()) {
		Vec4i vec = defects[handContour][i];
		if (i == 0) {
			fingers.push_back(contours[handContour][vec[0]]);
		}
		fingers.push_back(contours[handContour][vec[1]]);
		i++;
	}
	if (fingers.size() == 0) {
		Point everest(0, imageCut.rows);
		int j = 0;
		while (j < contours[handContour].size()) {
			Point v = contours[handContour][j];
			if (v.y < everest.y) {
				everest = v;
			}
			j++;
		}
		bool one = true;
		j = 0;
		while (j < points[handContour].size()) {
			Point v = points[handContour][j];
			if (v.y < everest.y + handRect.height / 6 && v.y != everest.y && v.x != everest.x) {
				one = false;
			}
			j++;
		}if (one) {
			fingers.push_back(everest);
		}
	}
	//draw fingertips
	for (int i = 0; i < fingers.size(); i++) {
		circle(imageCut, fingers[i], 4, Scalar(100, 255, 100), 3);
	}
}



void track(Mat ContourMatrix, int y) {
	Moments moment = moments(ContourMatrix);
	Point p(moment.m10 / moment.m00, ((moment.m01 / moment.m00) + y) / 2);
	circle(imageCut, p, 5, Scalar(0, 0, 255), 4);
}

void detectBackground() {
	mog2 = createBackgroundSubtractorMOG2();
	for (int i = 0; i < 50; i++) {
		setFlippedImage(cap, image);
		blur(imageCut, imageCut, Size(3, 3));
		mog2->apply(imageCut, Mog_Mask);
		threshold(Mog_Mask, Mog_Mask, 120, 255, CV_THRESH_BINARY);
		waitKey(33);
	}
}

void removeBackground() {
	setFlippedImage(cap, image);
	blur(imageCut, imageCut, Size(3, 3));
	mog2->apply(imageCut, Mog_Mask, 0);
	threshold(Mog_Mask, Mog_Mask, 90, 255, CV_THRESH_BINARY);
	waitKey(33);
}


int main() {
	locationSize = 20;
	detectionWait = 0;
	cap.open(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, frameWidth);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, frameHeight);
	namedWindow("Camera", WINDOW_NORMAL);
	cap >> image;

	Point checkLocations[sampleSize][2] = {
		 Point(image.cols * 2.5 / 3.5, image.rows / 1.5), Point(image.cols * 2.5 / 3.5 + locationSize, image.rows / 1.5 + locationSize) ,
		 Point(image.cols * 2.5 / 3.5, image.rows / 2), Point(image.cols * 2.5 / 3.5 + locationSize, image.rows / 2 + locationSize) ,
		 Point(image.cols * 3.75 / 4.75, image.rows / 1.5), Point(image.cols * 3.75 / 4.75 + locationSize, image.rows / 1.5 + locationSize) ,
		 Point(image.cols * 1.75 / 2.75, image.rows / 2), Point(image.cols * 1.75 / 2.75 + locationSize, image.rows / 2 + locationSize) ,
		 Point(image.cols * 1.75 / 2.75, image.rows / 1.5), Point(image.cols * 1.75 / 2.75 + locationSize, image.rows / 1.5 + locationSize) ,
		 Point(image.cols * 3 / 4, image.rows / 4), Point(image.cols * 3 / 4 + locationSize, image.rows / 4 + locationSize) ,
		 Point(image.cols * 2 / 3, image.rows / 4), Point(image.cols * 2 / 3 + locationSize, image.rows / 4 + locationSize) ,
		 Point(image.cols * 3 / 4, image.rows / 3), Point(image.cols * 3 / 4 + locationSize, image.rows / 3 + locationSize) ,
		 Point(image.cols * 2 / 3, image.rows / 3), Point(image.cols * 2 / 3 + locationSize, image.rows / 3 + locationSize) ,
		 Point(image.cols * 3.75 / 4.75, image.rows / 2), Point(image.cols * 3.75 / 4.75 + locationSize, image.rows / 2 + locationSize)
	};
	detectBackground();
	for (int i = 0; i < 100; i++) {
		setFlippedImage(cap, image);
		removeBackground();
		for (int j = 0; j < sampleSize; j++) {
			rectangle(image, checkLocations[j][0], checkLocations[j][1], Scalar(125, 125, 0), 1);
		}
		imshow("Camera", image);
		waitKey(33);
	}


	setFlippedImage(cap, image);
	cvtColor(image, imageHLS, CV_BGR2HLS);
	for (int j = 0; j < sampleSize; j++) {
		setFilter(j, checkLocations[j][0], locationSize);
	}
	imshow("Camera", image);
	waitKey(33);

	int noFinger = 0;
	int counter = 0;
	vector<int> numbers;
	// */
	while (1) {
		setFlippedImage(cap, image);
		removeBackground();
		pyrDown(imageCut, imageHLS);
		blur(imageHLS, imageHLS, Size(3, 3));
		cvtColor(imageHLS, imageHLS, COLOR_BGR2HLS);

		Mat imageTemp[sampleSize];
		for (int i = 0; i < sampleSize; i++) {
			imageTemp->push_back(Mat(imageHLS.rows, imageHLS.cols, CV_8U));
			inRange(imageHLS, filter[i][0], filter[i][1], imageTemp[i]);
			if (i == 0) imageTemp[i].copyTo(imageFiltered);
			else imageFiltered += imageTemp[i];
		}
		medianBlur(imageFiltered, imageFiltered, 5);
		pyrUp(imageFiltered, imageFiltered);
		imshow("Color 1", imageFiltered);
		//canny edges

		cvtColor(imageCut, imageBW, COLOR_BGR2GRAY);
		blur(imageBW, imageBW, Size(3, 3));
		Canny(imageBW, imageEdges, 20, 80);
		vector<vector<Point>> cannyContours;
		findContours(imageEdges, cannyContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		for (int i = 0; i < cannyContours.size(); i++) {
			drawContours(imageFiltered, cannyContours, i, Scalar(0, 0, 0), 2, 8, vector<Vec4i>(), 0, Point());
		}

		vector<vector<Point>> contours;
		vector<vector<Point>> colorContours;
		vector<Vec4i> newDefects;
		int handContour = 0;
		int colorContour = 0;

		findContours(imageFiltered, colorContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		for (int i = 0; i < colorContours.size(); i++) {
			if (colorContours[i].size() > colorContours[colorContour].size()) colorContour = i;
		}
		imshow("Color 2", imageFiltered);
		imshow("Background 1", Mog_Mask);
		Mat ColorMatrix = Mat(colorContours[colorContour]);
		Rect colorRect = boundingRect(ColorMatrix);
		locationRectangle = imageFiltered;
		locationRectangle.setTo(255);
		rectangle(locationRectangle, colorRect.tl(), colorRect.br(), 0, -1);
		Mog_Mask = Mog_Mask - locationRectangle;

		findContours(Mog_Mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		for (int i = 0; i < contours.size(); i++) {
			if (contours[i].size() > contours[handContour].size()) handContour = i;
		}
		if (contours.size() == 0 || colorContours.size() == 0) { cout << "hand cannot be detected" << endl; }
		else {
			vector<vector<Point>> points = vector<vector<Point>>(contours.size());
			vector<vector<int>> indices = vector<vector<int>>(contours.size());
			vector<vector<Vec4i>> defects = vector<vector<Vec4i>>(contours.size());

			Mat ContourMatrix = Mat(contours[handContour]);
			convexHull(ContourMatrix, indices[handContour], false, false);
			convexHull(ContourMatrix, points[handContour], false, true);
			approxPolyDP(points[handContour], points[handContour], 18, true);
			Rect handRect = boundingRect(ContourMatrix);

			//convexity defects

			convexityDefects(contours[handContour], indices[handContour], defects[handContour]);//


			int i = 0;
			while (i < defects[handContour].size()) {
				Vec4i vec = defects[handContour][i];
				Point Point1(contours[handContour][vec[0]]);
				Point Point2(contours[handContour][vec[1]]);
				Point Point3(contours[handContour][vec[2]]);
				if (distance(Point1, Point2) > handRect.height / 5 && distance(Point2, Point3) > handRect.height / 5 && angle(Point1, Point3, Point2) < 100 && Point2.y < (handRect.y + handRect.height * 3 / 4) && Point1.y < (handRect.y + handRect.height * 3 / 4)) {
					newDefects.push_back(vec);
				}
				i++;
			}
			defects[handContour].swap(newDefects);
			//remove overlapping points

			for (int i = 0; i < defects[handContour].size(); i++) {
				for (int j = i; j < defects[handContour].size(); j++) {
					Point startPoint(contours[handContour][defects[handContour][i][0]]);
					Point endPoint(contours[handContour][defects[handContour][i][1]]);
					Point start2Point(contours[handContour][defects[handContour][j][0]]);
					Point end2Point(contours[handContour][defects[handContour][j][1]]);
					if (distance(startPoint, end2Point) < handRect.width / 7) {
						contours[handContour][defects[handContour][i][0]] = end2Point;
					}
					if (distance(endPoint, start2Point) < handRect.width / 7) {
						contours[handContour][defects[handContour][j][0]] = endPoint;
					}
				}
			}

			//draw results
			drawContours(imageCut, points, handContour, Scalar(200, 0, 0), 2, 8, vector<Vec4i>(), 0, Point());

			rectangle(imageCut, colorRect.tl(), colorRect.br(), Scalar(0, 0, 200));

			//calculate fingertip locations
			calculateFingers(defects, handContour, contours, handRect, points);
			//get gesture
			int result = calculateNumber(handRect);
			detectionWait++;
			if (result != -1 && detectionWait > 20) {
				cout << "gesture number: " << result << endl;
			}
			track(ContourMatrix, handRect.y);
		}
		imshow("Input Image", image);
		imshow("Calculations", imageCut);
		imshow("Background 2", Mog_Mask);
		imshow("Hand Rectangle", locationRectangle);
		imshow("Canny Edges", imageEdges);
		waitKey(33);
	}
}