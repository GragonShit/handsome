/*******************************************************************************
* train lenet for eye recognition
* author: zhouming402@163.com
* date: 2016-07-26
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

void construct_net(network<sequential>& nn) {
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
       << fully_connected_layer<tan_h>(120, 10);       // F6, 120-in, 10-out
}

// list.txt each line like: "image_name\image_label\n"
void load_dataset(std::string path, std::vector<vec_t> &images, 
		std::vector<label_t> &labels) {
	std::ifstream ifs(path+"/list.txt");
	std::string line;
	while(std::getline(ifs, line)) {
		std::string name = line.substr(line.find("\t"));

		label_t label = (label_t)std::stoi(line.substr(line.find("\t")+1));
		labels.push_back(label);

		vec_t image(HEIGHT*WIDTH, SCALE_MIN);

		auto img = cv::imread(path+"/"+name, cv::IMREAD_GRAYSCALE);
		if (img.data == nullptr) continue; // cannot open, or it's not an image

		cv::Mat_<uint8_t> resized;
		cv::resize(img, resized, cv::Size(WIDTH, HEIGHT));
		std::transform(resized.begin(), resized.end(), std::back_inserter(data),
				[=](uint8_t c) { return (255 - c) * (SCALE_MAX - SCALE_MIN) / 255.0 + SCALE_MIN; });
		images.push_back(image);
	}
}

void train_lenet(std::string train_path, std::string test_path) {
    // specify loss-function and learning strategy
    network<sequential> nn;
    adagrad optimizer;

    construct_net(nn);

    std::cout << "load models..." << std::endl;

    // load dataset
    std::vector<label_t> train_labels, test_labels;
    std::vector<vec_t> train_images, test_images;

	load_dataset(train_path, train_images, train_labels);
	load_dataset(test_path, test_images, test_labels);

    std::cout << "start training" << std::endl;

    progress_display disp(train_images.size());
    timer t;
    int minibatch_size = 10;
    int num_epochs = 30;

    optimizer.alpha *= static_cast<tiny_cnn::float_t>(std::sqrt(minibatch_size));

    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;
        tiny_cnn::result res = nn.test(test_images, test_labels);
        std::cout << res.num_success << "/" << res.num_total << std::endl;

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&](){
        disp += minibatch_size;
    };

    // training
    nn.train<mse>(optimizer, train_images, train_labels, minibatch_size, num_epochs,
             on_enumerate_minibatch, on_enumerate_epoch);

    std::cout << "end training." << std::endl;

    // test and show results
    nn.test(test_images, test_labels).print_detail(std::cout);

    // save networks
    std::ofstream ofs("LeNet-weights");
    ofs << nn;
}

int main(int argc, char **argv) {
	train_lenet(argv[1], argv[2]);
}


