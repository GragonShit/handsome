/***************************************************************
* test lenet for eye recognition
* author: zhouming402@163.com
* date: 2016-07-31
***************************************************************/

#include <iostream>
#include <fstream>
#include "tiny_cnn/tiny_cnn.h"

#include <opencv2/opencv.hpp>

using namespace tiny_cnn;
using namespace tiny_cnn::activation;
using namespace std;

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
       << fully_connected_layer<tan_h>(120, 2);       // F6, 120-in, 10-out
}

// rescale output to 0-100
template <typename Activation>
double rescale(double x) {
    Activation a;
    return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

// convert tiny_cnn::image to cv::Mat and resize
cv::Mat image2mat(image<>& img) {
    cv::Mat ori(static_cast<int>(img.height()), static_cast<int>(img.width()), CV_8U, &img.at(0, 0));
    cv::Mat resized;
    cv::resize(ori, resized, cv::Size(), 3, 3, cv::INTER_AREA);
    return resized;
}

void convert_image(const std::string& imagefilename,
    double minv,
    double maxv,
    int w,
    int h,
    vec_t& data) {
    auto img = cv::imread(imagefilename, cv::IMREAD_GRAYSCALE);
    if (img.data == nullptr) return; // cannot open, or it's not an image

    cv::Mat_<uint8_t> resized;
    cv::resize(img, resized, cv::Size(w, h));

    std::transform(resized.begin(), resized.end(), std::back_inserter(data),
        [=](uint8_t c) { return (c) * (maxv - minv) / 255.0 + minv; });
}

void recognize(const std::string& dictionary, const std::string& filename) {
    network<sequential> nn;

    construct_lenet(nn);

    // load nets
    ifstream ifs(dictionary.c_str());
    ifs >> nn;

    // convert imagefile to vec_t
    vec_t data;
    convert_image(filename, SCALE_MIN, SCALE_MAX, 32, 32, data);

    // recognize
    auto res = nn.predict(data);
    vector<pair<double, int> > scores;

    // sort & print top-3
    for (int i = 0; i < 2; i++)
        scores.emplace_back(rescale<tan_h>(res[i]), i);

    sort(scores.begin(), scores.end(), greater<pair<double, int>>());

	if(scores[0].second == 1) {
		cout << "eye-open" << endl;
	} else {
		cout << "eye-close" << endl;
	}

    // visualize outputs of each layer
    for (size_t i = 0; i < nn.layer_size(); i++) {
        auto out_img = nn[i]->output_to_image();
        cv::imshow("layer:" + std::to_string(i), image2mat(out_img));
    }
    // visualize filter shape of first convolutional layer
    auto weight = nn.at<convolutional_layer<tan_h>>(0).weight_to_image();
    cv::imshow("weights:", image2mat(weight));

    cv::waitKey(0);
}
int main(int argc, char **argv) {
	recognize(argv[1], argv[2]);
}

