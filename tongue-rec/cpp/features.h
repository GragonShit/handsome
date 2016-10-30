/******************************************************************************
* extract features for tongue-rec
* author: zhouming402@163.com
* date: 2016-10-29
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


// tongue feature function
typedef std::vector<float> (*FEATURE_TONGUE)(cv::Mat&, dlib:: \
		full_object_detection&, dlib::full_object_detection&);

// get tongue regions
void tongue_region(const dlib::full_object_detection& shape, 
		dlib::rectangle& rect) {
	// tongue's points
	std::vector<cv::Point> tongue;

	for(int k = 48; k <= 59; ++ k) {
		tongue.push_back(cv::Point(shape.part(k).x(), shape.part(k).y()));
	}
	
	// tongue's region
	cv::Rect rect_tongue = cv::boundingRect(tongue);
	// return 
	rect.set_left(rect_tongue.tl().x);
	rect.set_top(rect_tongue.tl().y);
	rect.set_right(rect_tongue.br().x);
	rect.set_bottom(rect_tongue.br().y);
}

// get lip region
void lip_region(const dlib::full_object_detection& shape, 
		dlib::rectangle& rect) {
	// lip's points
	std::vector<cv::Point> lip;

	for(int k = 65; k <= 67; ++ k) {
		lip.push_back(cv::Point(shape.part(k).x(), shape.part(k).y()));
	}
	for(int k = 55; k <= 59; ++ k) {
		lip.push_back(cv::Point(shape.part(k).x(), shape.part(k).y()));
	}
	
	// lip's region
	cv::Rect rect_lip = cv::boundingRect(lip);
	// return 
	rect.set_left(rect_lip.tl().x);
	rect.set_top(rect_lip.tl().y);
	rect.set_right(rect_lip.br().x);
	rect.set_bottom(rect_lip.br().y);
}

