#include"plate_rec.h"

TextRecognizer::TextRecognizer()
{
	string model_path = "weights/ch_PP-OCRv3_rec_infer.onnx";
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
	
	max_wh_ratio = (float)this->inpWidth / (float)this->inpHeight;
	imgW = int(this->inpHeight * max_wh_ratio);

	ifstream ifs("rec_word_dict.txt");
	string line;
	while (getline(ifs, line))
	{
		this->alphabet.push_back(line);
	}
	names_len = this->alphabet.size();
}

Mat TextRecognizer::preprocess(Mat srcimg)
{
	Mat dstimg;
	cvtColor(srcimg, dstimg, COLOR_BGR2RGB);
	int h = srcimg.rows;
	int w = srcimg.cols;
	const float ratio = w / float(h);
	int resized_w = int(ceil((float)this->inpHeight * ratio));
	if (ceil(this->inpHeight*ratio) > imgW)
	{
		resized_w = imgW;
	}
	
	resize(dstimg, dstimg, Size(resized_w, this->inpHeight), INTER_LINEAR);
	return dstimg;
}

void TextRecognizer::normalize_(Mat img)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(this->inpHeight * imgW * img.channels());

	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < imgW; j++)
			{
				if (j < col)
				{
					float pix = img.ptr<uchar>(i)[j * 3 + c];
					this->input_image_[c * row * imgW + i * imgW + j] = (pix / 255.0 - 0.5) / 0.5;
				}
				else
				{
					this->input_image_[c * row * imgW + i * imgW + j] = 0;
				}
			}
		}
	}
}

Mat TextRecognizer::fourPointsTransform(const Mat& frame, Point2f vertices[4])
{
	int left = 10000;
	int right = 0;
	int top = 10000;
	int bottom = 0;
	for (int i = 0; i < 4; i++)
	{
		if (vertices[i].x < left)
		{
			left = int(vertices[i].x);
		}
		if (vertices[i].y < top)
		{
			top = int(vertices[i].y);
		}
		if (vertices[i].x > right)
		{
			right = int(vertices[i].x);
		}
		if (vertices[i].y > bottom)
		{
			bottom = int(vertices[i].y);
		}
	}
	
	const Size outputSize = Size(right - left, bottom - top);

	Point2f targetVertices[4] = { Point(0, outputSize.height - 1),
								  Point(0, 0), Point(outputSize.width - 1, 0),
								  Point(outputSize.width - 1, outputSize.height - 1),
	};
	Mat rotationMatrix = getPerspectiveTransform(vertices, targetVertices);
	Mat result;
	warpPerspective(frame, result, rotationMatrix, outputSize);
	return result;
}

Mat TextRecognizer::get_rotate_crop_image(const Mat& frame, Point2f vertices[4])
{
	int left = 10000;
	int right = 0;
	int top = 10000;
	int bottom = 0;
	for (int i = 0; i < 4; i++)
	{
		if (vertices[i].x < left)
		{
			left = int(vertices[i].x);
		}
		if (vertices[i].y < top)
		{
			top = int(vertices[i].y);
		}
		if (vertices[i].x > right)
		{
			right = int(vertices[i].x);
		}
		if (vertices[i].y > bottom)
		{
			bottom = int(vertices[i].y);
		}
	}
	Rect rect;
	rect.x = left;
	rect.y = top;
	rect.width = right - left;
	rect.height = bottom - top;
	Mat crop_plate = frame(rect);

	const Size outputSize = Size(rect.width, rect.height);

	Point2f targetVertices[4] = { Point(0, outputSize.height - 1),
								  Point(0, 0), Point(outputSize.width - 1, 0),
								  Point(outputSize.width - 1, outputSize.height - 1),
	};
	for (int i = 0; i < 4; i++)
	{
		vertices[i].x -= left;
		vertices[i].y -= top;
	}
	Mat rotationMatrix = getPerspectiveTransform(vertices, targetVertices);
	Mat result;
	warpPerspective(crop_plate, result, rotationMatrix, outputSize);
	return result;
}

string TextRecognizer::detect(Mat cv_image)
{
	Mat dstimg = this->preprocess(cv_image);
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->imgW };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
	
	int i = 0, j = 0;
	int h = ort_outputs.at(0).GetTensorTypeAndShapeInfo().GetShape().at(2);
	int w = ort_outputs.at(0).GetTensorTypeAndShapeInfo().GetShape().at(1);
	int* preb_label = new int[w];
	for (i = 0; i < w; i++)
	{
		int one_label_idx = 0;
		float max_data = -10000;
		for (j = 0; j < h; j++)
		{
			float data_ = pdata[i*h + j];
			if (data_ > max_data)
			{
				max_data = data_;
				one_label_idx = j;
			}
		}
		preb_label[i] = one_label_idx;
	}

	vector<int> no_repeat_blank_label;
	for (size_t elementIndex = 1; elementIndex < w; ++elementIndex)
	{
		if (preb_label[elementIndex] != 0 &&
			preb_label[elementIndex - 1] != preb_label[elementIndex])
		{
			no_repeat_blank_label.push_back(preb_label[elementIndex] - 1);
		}
	}

	delete[] preb_label;
	int len_s = no_repeat_blank_label.size();
	string plate_text;
	for (i = 0; i < len_s; i++)
	{
		plate_text += alphabet[no_repeat_blank_label[i]];
	}
	return plate_text;
}