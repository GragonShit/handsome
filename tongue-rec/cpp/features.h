/******************************************************************************
* extract features for eyes-rec
* author: zhouming402@163.com
* date: 2016-07-24
******************************************************************************/

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
#include <string>


// eye feature function
typedef std::vector<float> (*FEATURE_EYE)(cv::Mat&, dlib:: \
		full_object_detection&, dlib::full_object_detection&);

// get eye regions
void eye_region(const dlib::full_object_detection& shape, 
		dlib::rectangle& rectl, dlib::rectangle& rectr) {
	// eyes' points
	std::vector<cv::Point> lefteye;
	std::vector<cv::Point> righteye;

	for(int k = 36; k <= 41; ++ k) {
		lefteye.push_back(cv::Point(shape.part(k).x(), shape.part(k).y()));
	}
	for(int k = 42; k <=47; ++ k) {
		righteye.push_back(cv::Point(shape.part(k).x(), shape.part(k).y()));
	}
	
	// eyes' region
	cv::Rect rect_lefteye = cv::boundingRect(lefteye);
	cv::Rect rect_righteye = cv::boundingRect(righteye);
	float leftx = rect_lefteye.x + rect_lefteye.width/2.0;
	float lefty = rect_lefteye.y + rect_lefteye.height/2.0;
	float rightx = rect_righteye.x + rect_righteye.width/2.0;
	float righty = rect_righteye.y + rect_righteye.height/2.0;
	// special location?
	float w = (rightx - leftx) * 3 / 10;
	float h = (rightx - leftx) / 10;
	// return 
	rectl.set_left(leftx - w);
	rectl.set_top(lefty - h);
	rectl.set_right(leftx + w);
	rectl.set_bottom(lefty + h);
	rectr.set_left(rightx - w);
	rectr.set_top(righty - h);
	rectr.set_right(rightx + w);
	rectr.set_bottom(righty + h);
}
