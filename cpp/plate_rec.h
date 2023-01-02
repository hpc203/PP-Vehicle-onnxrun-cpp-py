#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;

class TextRecognizer
{
public:
	TextRecognizer();
	string detect(Mat cv_image);
	Mat fourPointsTransform(const Mat& frame, Point2f vertices[4]);
	Mat get_rotate_crop_image(const Mat& frame, Point2f vertices[4]);
private:
	Mat preprocess(Mat srcimg);
	void normalize_(Mat img);
	const int inpWidth = 320;
	const int inpHeight = 48;
	float max_wh_ratio;
	int imgW;
	vector<float> input_image_;
	vector<string> alphabet;
	int names_len;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "CRNN");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};