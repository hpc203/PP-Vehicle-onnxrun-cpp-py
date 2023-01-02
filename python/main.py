import os
import argparse
import cv2
import numpy as np
import onnxruntime
from plate_det import PlateDetector
from plate_rec import TextRecognizer,puttext_chinese
os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'

class PP_YOLOE():
    def __init__(self, prob_threshold=0.8):
        self.class_names = ['vehicle']
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.session = onnxruntime.InferenceSession('weights/mot_ppyoloe_s_36e_ppvehicle.onnx', so)
        self.input_size = (
            self.session.get_inputs()[0].shape[3], self.session.get_inputs()[0].shape[2])  ###width, height
        self.confThreshold = prob_threshold
        self.scale_factor = np.array([1., 1.], dtype=np.float32)

    def preprocess(self, srcimg):
        img = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img = np.transpose(img, [2, 0, 1])
        return img

    def detect(self, srcimg):
        img = self.preprocess(srcimg)
        inputs = {'image': img[None, :, :, :], 'scale_factor': self.scale_factor[None, :]}
        ort_inputs = {i.name: inputs[i.name] for i in self.session.get_inputs() if i.name in inputs}
        output = self.session.run(None, ort_inputs)
        bbox, bbox_num = output
        keep_idx = (bbox[:, 1] > self.confThreshold) & (bbox[:, 0] > -1)
        bbox = bbox[keep_idx, :]
        ratioh = srcimg.shape[0] / self.input_size[1]
        ratiow = srcimg.shape[1] / self.input_size[0]
        dets = []
        for (clsid, score, xmin, ymin, xmax, ymax) in bbox:
            xmin = xmin * ratiow
            ymin = ymin * ratioh
            xmax = xmax * ratiow
            ymax = ymax * ratioh
            dets.append([xmin, ymin, xmax, ymax, score, int(clsid)])
        return np.array(dets, dtype=np.float32)

class VehicleAttr():
    def __init__(self, color_threshold=0.5, type_threshold=0.5):
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.session = onnxruntime.InferenceSession('weights/vehicle_attribute_model.onnx', so)
        self.input_size = (
            self.session.get_inputs()[0].shape[3], self.session.get_inputs()[0].shape[2])  ###width, height
        self.color_threshold = color_threshold
        self.type_threshold = type_threshold
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1,1,3))
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1,1,3))
        self.color_list = [
            "yellow", "orange", "green", "gray", "red", "blue", "white",
            "golden", "brown", "black"
        ]
        self.type_list = [
            "sedan", "suv", "van", "hatchback", "mpv", "pickup", "bus", "truck",
            "estate"
        ]

    def preprocess(self, srcimg):
        img = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        img = (img.astype(np.float32)/255.0 - self.mean) / self.std
        img = np.transpose(img, [2, 0, 1])
        return img

    def detect(self, srcimg):
        img = self.preprocess(srcimg)
        ort_inputs = {i.name: img[None, :, :, :] for i in self.session.get_inputs()}
        res = self.session.run(None, ort_inputs)[0].flatten()

        color_res_str = "Color: "
        type_res_str = "Type: "
        color_idx = np.argmax(res[:10])
        type_idx = np.argmax(res[10:])

        if res[color_idx] >= self.color_threshold:
            color_res_str += self.color_list[color_idx]
        else:
            color_res_str += "Unknown"

        if res[type_idx + 10] >= self.type_threshold:
            type_res_str += self.type_list[type_idx]
        else:
            type_res_str += "Unknown"

        return color_res_str, type_res_str

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='images/street_00001.jpg', help="image path")
    parser.add_argument('--confThreshold', default=0.6, type=float, help='class confidence')
    args = parser.parse_args()

    detect_vehicle_model = PP_YOLOE(prob_threshold=args.confThreshold)
    rec_vehicle_attr_model = VehicleAttr()
    detect_plate_model = PlateDetector()
    recognition = TextRecognizer()

    srcimg = cv2.imread(args.imgpath)
    dets = detect_vehicle_model.detect(srcimg)
    for i in range(dets.shape[0]):
        xmin, ymin, xmax, ymax = int(dets[i, 0]), int(dets[i, 1]), int(dets[i, 2]), int(dets[i, 3])
        crop_img = srcimg[ymin:ymax, xmin:xmax,:]
        color_res_str, type_res_str = rec_vehicle_attr_model.detect(crop_img)

        box_list = detect_plate_model.detect(crop_img)
        text = ''
        if len(box_list) > 0:
            # plate_img = detect_plate_model.draw_plate(box_list, crop_img.copy())
            for point in box_list:
                point = detect_plate_model.order_points_clockwise(point)
                textimg = detect_plate_model.get_rotate_crop_image(crop_img, point.astype(np.float32))
                text = recognition.predict_text(textimg)
                # text = recognition.replace_cn_code(text)

        # winName = 'Deep learning plate detection in ONNXRuntime'
        # cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        # cv2.imshow(winName, plate_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.rectangle(srcimg, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
        if len(text)==0:
            cv2.putText(srcimg, type_res_str+' , '+color_res_str, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        else:
            srcimg = puttext_chinese(srcimg, type_res_str+' , '+color_res_str+' , '+text, (xmin, ymin - 10), (0, 255, 0))

    # winName = 'Deep learning object detection in ONNXRuntime'
    # cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    # cv2.imshow(winName, srcimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite('result.jpg', srcimg)
