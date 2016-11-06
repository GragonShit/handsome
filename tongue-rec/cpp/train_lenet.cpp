/*******************************************************************************
* train lenet for tongue recognition
* author: zhouming402@163.com
* date: 2016-10-29
*******************************************************************************/

#include <iostream>
#include <fstream>
#include "tiny_cnn/tiny_cnn.h"

#include <opencv2/opencv.hpp>

using namespace tiny_cnn;
using namespace tiny_cnn::activation;


#define SCALE_MIN -1.0
#define SCALE_MAX 1.0
#define WIDTH 32
#define HEIGHT 32

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
       << fully_connected_layer<softmax>(120, 3);       // F6, 120-in, 10-out
}

void construct_lenet_fix(network<sequential>& nn) {
    // construct nets
    nn << convolutional_layer<activation::identity>(32, 32, 5, 3, 6)  // C1, 1@32x32-in, 6@28x28-out
       << max_pooling_layer<relu>(28, 28, 6, 2)   // S2, 6@28x28-in, 6@14x14-out
       << convolutional_layer<activation::identity>(14, 14, 5, 6, 16)
       << max_pooling_layer<relu>(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out
       << convolutional_layer<activation::identity>(5, 5, 5, 16, 64) // C5, 16@5x5-in, 120@1x1-out
       << fully_connected_layer<relu>(64, 128)       // F6, 120-in, 10-out
       << fully_connected_layer<softmax>(128, 3);       // F6, 120-in, 10-out
}

template <typename N>
void construct_net(N& nn) {
    typedef convolutional_layer<activation::identity> conv;
    typedef max_pooling_layer<relu> pool;

    const int n_fmaps = 32; ///< number of feature maps for upper layer
    const int n_fmaps2 = 64; ///< number of feature maps for lower layer
    const int n_fc = 64; ///< number of hidden units in fully-connected layer

    nn << conv(32, 32, 5, 3, n_fmaps, padding::same)
        << pool(32, 32, n_fmaps, 2)
        << conv(16, 16, 5, n_fmaps, n_fmaps, padding::same)
        << pool(16, 16, n_fmaps, 2)
        << conv(8, 8, 5, n_fmaps, n_fmaps2, padding::same)
        << pool(8, 8, n_fmaps2, 2)
        << fully_connected_layer<activation::identity>(4 * 4 * n_fmaps2, n_fc)
        << fully_connected_layer<softmax>(n_fc, 3);
}
// list.txt each line like: "image_name\image_label\n"
void load_dataset_color(std::string path, std::vector<vec_t> &images, 
		std::vector<label_t> &labels) {
	std::ifstream ifs(path);
	std::string line;
	path = path.substr(0, path.rfind("/"));
	while(std::getline(ifs, line)) {
		std::string name = line.substr(0, line.find("\t"));

		label_t label = (label_t)std::stoi(line.substr(line.find("\t")+1));

		vec_t image;
		image.resize(WIDTH*HEIGHT*3);
		
		// auto img = cv::imread(path+"/"+name, cv::IMREAD_GRAYSCALE);
		auto img = cv::imread(path+"/"+name, cv::IMREAD_COLOR);
		if (img.data == nullptr) continue; // cannot open, or it's not an image

		cv::Mat resized;
		cv::resize(img, resized, cv::Size(WIDTH, HEIGHT));
		for(int k = 0; k < 3; ++ k) {
			for(int i = 0; i < HEIGHT; ++ i) {
				for(int j = 0; j < WIDTH; ++ j) {
					image[k*HEIGHT*WIDTH+i*WIDTH+j] = resized.data[i*WIDTH*3+j*3+k] * (SCALE_MAX - SCALE_MIN) / 255.0 + SCALE_MIN;
				}
			}
		}
		images.push_back(image);
		labels.push_back(label);
	}
}

void load_dataset(std::string path, std::vector<vec_t> &images, 
		std::vector<label_t> &labels) {
	std::ifstream ifs(path);
	std::string line;
	path = path.substr(0, path.rfind("/"));
	while(std::getline(ifs, line)) {
		std::string name = line.substr(0, line.find("\t"));

		label_t label = (label_t)std::stoi(line.substr(line.find("\t")+1));

		vec_t image;
		
		auto img = cv::imread(path+"/"+name, cv::IMREAD_GRAYSCALE);
		if (img.data == nullptr) continue; // cannot open, or it's not an image
		
		// equalization
		cv::Mat equalized;
		cv::equalizeHist(img, equalized);

		cv::Mat_<uint8_t> resized;
		cv::resize(equalized, resized, cv::Size(WIDTH, HEIGHT));
		std::transform(resized.begin(), resized.end(), std::back_inserter(image),
				[=](uint8_t c) { return (c) * (SCALE_MAX - SCALE_MIN) / 255.0 + SCALE_MIN; });
		images.push_back(image);
		labels.push_back(label);
	}
}

void train_net(std::string train_path, std::string test_path) {
    // specify loss-function and learning strategy
    network<sequential> nn;
    // adagrad optimizer;
	momentum optimizer;
	optimizer.alpha = 0.01;
	optimizer.lambda = 0.0005;
	optimizer.mu = 0.9;

    construct_lenet(nn);

    std::cout << "load models..." << std::endl;

    // load dataset
    std::vector<label_t> train_labels, test_labels;
    std::vector<vec_t> train_images, test_images;

	load_dataset(train_path, train_images, train_labels);
	std::cout << "training set num: " << train_images.size() << std::endl;
	load_dataset(test_path, test_images, test_labels);
	std::cout << "testing set num: " << test_images.size() << std::endl;

    std::cout << "start training" << std::endl;

    progress_display disp(train_images.size());
    timer t;
	int index = 0;
    int minibatch_size = 32;
    int num_epochs = 15;

    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;
		std::cout << "epoch " << index << std::endl;
		t.restart();
        tiny_cnn::result res = nn.test(test_images, test_labels);
		std::cout << t.elapsed()/res.num_total  << "s per instance: ";
        std::cout << res.num_success << "/" << res.num_total << std::endl;

        disp.restart(train_images.size());
        t.restart();
		index ++;
    };

    auto on_enumerate_minibatch = [&](){
        disp += minibatch_size;
    };

    // training
    nn.train<cross_entropy>(optimizer, train_images, train_labels, minibatch_size, num_epochs,
             on_enumerate_minibatch, on_enumerate_epoch);

    std::cout << "end training." << std::endl;

    // test and show results
    nn.test(test_images, test_labels).print_detail(std::cout);

    // save networks
    std::ofstream ofs("Net-weights");
    ofs << nn;
}

int main(int argc, char **argv) {
	train_net(argv[1], argv[2]);
}


