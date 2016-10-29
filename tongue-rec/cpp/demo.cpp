/*******************************************************************
* demo.cpp
* author: zhouming402@163.com
* date: 2016-10-29
*******************************************************************/


#include <iostream>
#include <fstream>
#include "tiny_cnn/tiny_cnn.h"

#include <opencv2/opencv.hpp>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>

#include "features.h"

using namespace tiny_cnn;
using namespace tiny_cnn::activation;

// variable macro
#define SCALE_MIN -1.0
#define SCALE_MAX 1.0
#define WIDTH 32
#define HEIGHT 32
#define RATIO 0.5

// function macro
#define TIMER_INFO(t, tag) std::cout << tag << ": " << t.elapsed() << "s." << std::endl

void construct_lenet(network<sequential>& nn) {
	// connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
	static const bool tbl[] = {
		O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
		O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
		O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
		X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
		X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
		X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
	};
#undef O
#undef X

    // construct nets
    nn << convolutional_layer<tan_h>(32, 32, 5, 1, 6)  // C1, 1@32x32-in, 6@28x28-out
       << average_pooling_layer<tan_h>(28, 28, 6, 2)   // S2, 6@28x28-in, 6@14x14-out
       << convolutional_layer<tan_h>(14, 14, 5, 6, 16,
            connection_table(tbl, 6, 16))              // C3, 6@14x14-in, 16@10x10-in
       << average_pooling_layer<tan_h>(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out
       << convolutional_layer<tan_h>(5, 5, 5, 16, 120) // C5, 16@5x5-in, 120@1x1-out
       << fully_connected_layer<tan_h>(120, 2);       // F6, 120-in, 10-out
}

void convert_image(const cv::Mat& eye,
    double minv,
    double maxv,
    int w,
    int h,
    vec_t& data) {
    if (eye.data == nullptr) return; // cannot open, or it's not an image

    cv::Mat_<uint8_t> resized;
    cv::resize(eye, resized, cv::Size(w, h));

    std::transform(resized.begin(), resized.end(), std::back_inserter(data),
        [=](uint8_t c) { return (c) * (maxv - minv) / 255.0 + minv; });
}

std::pair<double, int> recognize(network<sequential>& nn, const cv::Mat& gray, 
		const dlib::rectangle& rect) {
    // convert imagefile to vec_t
    vec_t data;
    convert_image(gray(cv::Range(rect.top(),rect.bottom()+1), 
				cv::Range(rect.left(),rect.right())), 
			SCALE_MIN, 
			SCALE_MAX, 
			32, 
			32, 
			data);

    // recognize
    auto res = nn.predict(data);
	std::vector<std::pair<double, int> > scores;

    // sort 
    for (int i = 0; i < 2; i++)
        scores.emplace_back(res[i], i);

	std::sort(scores.begin(), scores.end(), std::greater<std::pair<double, int>>());

	return scores[0];
}

int main(int argc, char** argv) {
    try
    {
        if (argc == 1)
        {
			std::cout << "LIKE THIS:" << std::endl;
			std::cout << "./demo /path/to/shape_predictor_face /path/to/NN" << std::endl;
            return 0;
        }
	
		// gui
		cv::VideoCapture cap(0);
		dlib::image_window win;
		
		// face detector
		dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
		// face landmarker
		dlib::shape_predictor sp_face;
		dlib::deserialize(argv[1]) >> sp_face;
		// tongue status classfier
		network<sequential> nn;
		construct_lenet(nn);
		std::ifstream ifs(argv[2]);
		ifs >> nn;
		
		timer t; // timer
		while(!win.is_closed())
        {
			try {
				cv::Mat frame; // source image 
				cap >> frame; 
				// resize by RATIO
				cv::Mat resized;
				cv::resize(frame, resized, cv::Size(0,0), RATIO, RATIO);
				dlib::cv_image<dlib::bgr_pixel> img(resized); 
				// gray image for status classifier
				cv::Mat gray;
				cv::cvtColor(resized, gray, CV_BGR2GRAY);
				
				t.restart();
				std::vector<dlib::rectangle> dets = detector(img);
				TIMER_INFO(t, "face detection");

				for (unsigned long j = 0; j < dets.size(); ++j)
				{
					t.restart();
					dlib::full_object_detection shape = sp_face(img, dets[j]);
					TIMER_INFO(t, "face landmark");
					for(unsigned long k = 0; k < 55; ++ k) {
						draw_solid_circle(img, shape.part(k), 2, dlib::rgb_pixel(0,255,0));
					}
					for(unsigned long k = 60; k < 65; ++ k) {
						draw_solid_circle(img, shape.part(k), 2, dlib::rgb_pixel(0,255,0));
					}

					dlib::rectangle rect;
					tongue_region(shape, rect);

					t.restart();
					std::pair<double, int> flag = recognize(nn, gray, rect);
					TIMER_INFO(t, "tongue status");
					if(flag.second == 0) {
						for(unsigned long k = 65; k < 68; ++ k) {
							draw_solid_circle(img, shape.part(k), 2, dlib::rgb_pixel(0,255,0));
						}
						for(unsigned long k = 55; k < 60; ++ k) {
							draw_solid_circle(img, shape.part(k), 2, dlib::rgb_pixel(0,255,0));
						}
					}
					win.add_overlay(dlib::image_window::overlay_rect(rect, dlib::rgb_pixel(255,0,0), 
								std::to_string(flag.first)));
				}
				win.set_image(img);
			}
			catch (std::exception& e) {
				std::cout << e.what() << std::endl;
			}
        }
    } catch (std::exception& e)
    {
		std::cout << "\nexception thrown!" << std::endl;
		std::cout << e.what() << std::endl;
    }

	return 0;
}

