# PP-Vehicle-onnxrun-cpp-py
使用ONNXRuntime部署百度飞桨开源PP-Vehicle车辆分析，包含车辆检测，识别车型和车辆颜色，车牌检测，车牌识别5个功能。
起初想使用opencv做部署的，但是opencv的dnn模块读取onnx文件出错了。
因此，使用ONNXRuntime做推理，彻底摆脱对PaddlePaddle的依赖。

onnx文件在百度云盘，下载链接：https://pan.baidu.com/s/1Z1Phr7KPubsAqhHW5edMvQ 
提取码：xw2i

由于opencv不支持在图片里写汉字的，而车牌号码的开头的汉字。因此在python程序里，是调用
pillow库加载simhei.ttf文件实现在图片里写汉字的功能。c++程序里，没有把识别的车牌号码，
写在图片里的，其次，在程序里解析rec_word_dict.txt有出现乱码，原因可能是txt文档编码格式有问题，
也有可能是c++程序里解析rec_word_dict.txt的代码要做调整。
在c++程序里，要实现在图片里写汉字，需要依赖opencv和opencv-contrib库的。
具体代码实现，可参考https://gitee.com/cvmart/ev_sdk_demo4.0_pedestrian_intrusion/blob/master/include/ji_utils.h
