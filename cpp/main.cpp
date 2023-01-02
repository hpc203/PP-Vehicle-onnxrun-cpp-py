#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include"plate_det.h"
#include"plate_rec.h"

using namespace cv;
using namespace std;
using namespace Ort;

typedef struct BoxInfo
{
	int xmin;
	int ymin;
	int xmax;
	int ymax;
	float score;
	string name;
} BoxInfo;

class PP_YOLOE
{
public:
	PP_YOLOE(string model_path, float confThreshold);
	vector<BoxInfo> detect(Mat cv_image);
	void draw_pred(Mat& srcimg, vector<BoxInfo> boxs);
private:
	float confThreshold;
	const int num_class = 1;

	Mat preprocess(Mat srcimg);
	void normalize_(Mat img);
	int inpWidth;
	int inpHeight;
	vector<float> input_image_;
	vector<float> scale_factor = { 1,1 };

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "pp-yoloe");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

PP_YOLOE::PP_YOLOE(string model_path, float confThreshold)
{
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	this->confThreshold = confThreshold;
}

Mat PP_YOLOE::preprocess(Mat srcimg)
{
	Mat dstimg;
	cvtColor(srcimg, dstimg, COLOR_BGR2RGB);
	resize(dstimg, dstimg, Size(this->inpWidth, this->inpHeight), INTER_LINEAR);
	return dstimg;
}

void PP_YOLOE::normalize_(Mat img)
{
	//img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + c];
				this->input_image_[c * row * col + i * col + j] = pix;
			}
		}
	}
}

vector<BoxInfo> PP_YOLOE::detect(Mat srcimg)
{
	Mat dstimg = this->preprocess(srcimg);
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };
	array<int64_t, 2> scale_shape_{ 1, 2 };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	vector<Value> ort_inputs;
	ort_inputs.push_back(Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size()));
	ort_inputs.push_back(Value::CreateTensor<float>(allocator_info, scale_factor.data(), scale_factor.size(), scale_shape_.data(), scale_shape_.size()));
	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, input_names.data(), ort_inputs.data(), 2, output_names.data(), output_names.size());
	const float* outs = ort_outputs[0].GetTensorMutableData<float>();
	const int* box_num = ort_outputs[1].GetTensorMutableData<int>();

	const float ratioh = float(srcimg.rows) / this->inpHeight;
	const float ratiow = float(srcimg.cols) / this->inpWidth;
	vector<BoxInfo> boxs;
	for (int i = 0; i < box_num[0]; i++)
	{
		if (outs[0] > -1 && outs[1] > this->confThreshold)
		{
			boxs.push_back({ int(outs[2] * ratiow), int(outs[3] * ratioh), int(outs[4] * ratiow), int(outs[5] * ratioh), outs[1], "vehicle" });
		}
		outs += 6;
	}

	return boxs;
}

void PP_YOLOE::draw_pred(Mat& srcimg, vector<BoxInfo> boxs)
{
	for (size_t i = 0; i < boxs.size(); ++i)
	{
		rectangle(srcimg, Point(boxs[i].xmin, boxs[i].ymin), Point(boxs[i].xmax, boxs[i].ymax), Scalar(0, 0, 255), 2);
		string label = format("%.2f", boxs[i].score);
		label = boxs[i].name + ":" + label;
		putText(srcimg, label, Point(boxs[i].xmin, boxs[i].ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	}
}

class VehicleAttr
{
public:
	VehicleAttr();
	void detect(Mat cv_image, string& color_res_str, string& type_res_str);
private:
	const float color_threshold = 0.5;
	const float type_threshold = 0.5;
	const char* color_list[10] = { "yellow", "orange", "green", "gray", "red", "blue", "white", "golden", "brown", "black" };
	const char* type_list[9] = { "sedan", "suv", "van", "hatchback", "mpv", "pickup", "bus", "truck", "estate" };
	const float mean[3] = { 0.485, 0.456, 0.406 };
	const float std[3] = { 0.229, 0.224, 0.225 };

	Mat preprocess(Mat srcimg);
	void normalize_(Mat img);
	int inpWidth;
	int inpHeight;
	int num_out;
	vector<float> input_image_;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "VehicleAttr");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

VehicleAttr::VehicleAttr()
{
	string model_path = "weights/vehicle_attribute_model.onnx";
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	num_out = output_node_dims[0][1];
}

Mat VehicleAttr::preprocess(Mat srcimg)
{
	Mat dstimg;
	cvtColor(srcimg, dstimg, COLOR_BGR2RGB);
	resize(dstimg, dstimg, Size(this->inpWidth, this->inpHeight), INTER_LINEAR);
	return dstimg;
}

void VehicleAttr::normalize_(Mat img)
{
	//img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + c];
				this->input_image_[c * row * col + i * col + j] = (pix / 255.0 - this->mean[c]) / this->std[c];
			}
		}
	}
}

void VehicleAttr::detect(Mat cv_image, string& color_res_str, string& type_res_str)
{
	Mat dstimg = this->preprocess(cv_image);
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
	int color_idx;
	float max_prob = -1;
	for (int i = 0; i < 10; i++)
	{
		if (pdata[i] > max_prob)
		{
			max_prob = pdata[i];
			color_idx = i;
		}
	}
	int type_idx;
	max_prob = -1;
	for (int i = 10; i < num_out; i++)
	{
		if (pdata[i] > max_prob)
		{
			max_prob = pdata[i];
			type_idx = i - 10;
		}
	}

	if (pdata[color_idx] >= this->color_threshold)
	{
		color_res_str += this->color_list[color_idx];
	}
	else
	{
		color_res_str += "Unknown";
	}

	if (pdata[type_idx + 10] > this->type_threshold)
	{
		type_res_str += this->type_list[type_idx];
	}
	else
	{
		type_res_str += "Unknown";
	}
}

int main()
{
	PP_YOLOE detect_vehicle_model("weights/mot_ppyoloe_s_36e_ppvehicle.onnx", 0.6);
	VehicleAttr rec_vehicle_attr_model;
	PlateDetector detect_plate_model;
	TextRecognizer recognition;

	string imgpath = "images/street_00001.jpg";
	Mat srcimg = imread(imgpath);
	vector<BoxInfo> boxs = detect_vehicle_model.detect(srcimg);
	for (size_t n = 0; n < boxs.size(); ++n)
	{
		Rect rect;
		rect.x = boxs[n].xmin;
		rect.y = boxs[n].ymin;
		rect.width = boxs[n].xmax - boxs[n].xmin;
		rect.height = boxs[n].ymax - boxs[n].ymin;
		Mat crop_img = srcimg(rect);
		string color_res_str = "Color: ";
		string type_res_str = "Type: ";
		rec_vehicle_attr_model.detect(crop_img, color_res_str, type_res_str);
		vector< vector<Point2f> > results = detect_plate_model.detect(crop_img);
		
		/*detect_plate_model.draw_pred(crop_img, results);
		namedWindow("detect-plate", WINDOW_NORMAL);
		imshow("detect-plate", crop_img);
		waitKey(0);
		destroyAllWindows();*/
		
		for (size_t i = 0; i < results.size(); i++)
		{
			Point2f vertices[4];
			for (int j = 0; j < 4; ++j)
			{
				vertices[j].x = results[i][j].x;
				vertices[j].y = results[i][j].y;
			}
			
			Mat plate_img = recognition.get_rotate_crop_image(crop_img, vertices);
			/*imshow("plate_img", plate_img);
			waitKey(0);
			destroyAllWindows();*/

			string plate_text = recognition.detect(plate_img);
			cout << plate_text << endl;
		}
		string label = type_res_str + " , " + color_res_str;
		rectangle(srcimg, Point(boxs[n].xmin, boxs[n].ymin), Point(boxs[n].xmax, boxs[n].ymax), Scalar(0, 0, 255), 2);
		putText(srcimg, label, Point(boxs[n].xmin, boxs[n].ymin - 10), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
	}

	imwrite("result.jpg", srcimg);
	/*static const string kWinName = "Deep learning object detection in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();*/
}