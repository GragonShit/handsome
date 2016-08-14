/*************************************************************************
* test trained object detector
* author: zhouming402@163.com
* date: 2016-08-13
*************************************************************************/

#include <dlib/svm_threaded.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/data_io.h>

#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{  

    try
    {
        if (argc < 2)
        {
            cout << "Give the path to the examples/faces directory as the argument to this" << endl;
            cout << "program.  For example, if you are in the examples folder then execute " << endl;
            cout << "this program by running: " << endl;
            cout << "   ./fhog_object_detector_ex face_directory face_detector" << endl;
            cout << endl;
            return 0;
        }
		const std::string faces_directory = argv[1];
        dlib::array<array2d<unsigned char> > images_test;
        std::vector<std::vector<rectangle> > face_boxes_test;

        load_image_dataset(images_test, face_boxes_test, faces_directory+"/testing.xml");

        upsample_image_dataset<pyramid_down<2> >(images_test,  face_boxes_test);
        cout << "num testing images:  " << images_test.size() << endl;
		
		if(argc == 3) {
			typedef object_detector<scan_fhog_pyramid<pyramid_down<6> > > face_detector; 
			face_detector detector;
			
			std::ifstream in(argv[2]);
			dlib::deserialize(detector, in);
			cout << "testing results:  " << test_object_detection_function(detector, images_test, face_boxes_test) << endl;
		} else {
			dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
			cout << "testing results:  " << test_object_detection_function(detector, images_test, face_boxes_test) << endl;
		}


        // Now for the really fun part.  Let's display the testing images on the screen and
        // show the output of the face detector overlaid on each image.  You will see that
        // it finds all the faces without false alarming on any non-faces.
        // image_window win; 
		/*
		cout << "saving img..." << endl;
        for (unsigned long i = 0; i < images_test.size(); ++i)
        {
            // Run the detector and get the face detections.
            std::vector<rectangle> dets = detector(images_test[i]);
			for(int j = 0; j < dets.size(); ++ j) {
				dlib::draw_rectangle(images_test[i], dets[j], dlib::rgb_pixel(255,255,255));
			}
			dlib::save_jpeg(images_test[i], "jpg/"+to_string(i)+".jpg");
        }*/
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

