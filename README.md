# PP-Vehicle-onnxrun-cpp-py
使用ONNXRuntime部署百度飞桨开源PP-Vehicle车辆分析，包含车辆检测，车辆属性识别，车牌检测，车牌检测4个功能。
起初想使用opencv做部署的，但是opencv的dnn模块读取onnx文件出错了。
因此，使用ONNXRuntime做推理，彻底摆脱对PaddlePaddle的依赖。

onnx文件在百度云盘，下载链接：https://pan.baidu.com/s/1Z1Phr7KPubsAqhHW5edMvQ 
提取码：xw2i
