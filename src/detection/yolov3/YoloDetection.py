import argparse
import time
from sys import platform
from .utils.datasets import *
from .utils import *
from .models import *
import subprocess
import os 
import cv2


class YoloDetection: 

	def __init__(self,
				cfg,
				data_cfg,
				weights,
				nms_thres=0.5,
				conf_thres=0.5,
				save_txt=False,
				img_size=416):
		
		self.device = torch_utils.select_device()
		torch.backends.cudnn.benchmark = False  # set False for reproducible results
		self.conf_thres = conf_thres
		self.nms_thres = nms_thres
		self.model = Darknet(cfg, img_size)
		self.img_size = img_size
		# if not os.path.exists(weights):
		# 	subprocess.call("")

		_ = load_darknet_weights(self.model, weights)
		# Fuse Conv2d + BatchNorm2d layers
		self.model.fuse()
		# Eval mode
		self.model.to(self.device).eval()

		# Get classes and colors
		# self.classes = load_classes(parse_data_cfg(data_cfg)['names'])
		# colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]


	def letterbox(self,img, new_shape=416, color=(127.5, 127.5, 127.5), mode='auto'):
		# Resize a rectangular image to a 32 pixel multiple rectangle
		# https://github.com/ultralytics/yolov3/issues/232
		shape = img.shape[:2]  # current shape [height, width]

		if isinstance(new_shape, int):
			ratio = float(new_shape) / max(shape)
		else:
			ratio = max(new_shape) / max(shape)  # ratio  = new / old
		ratiow, ratioh = ratio, ratio
		new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

		# Compute padding https://github.com/ultralytics/yolov3/issues/232
		if mode is 'auto':  # minimum rectangle
			dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
			dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
		elif mode is 'square':  # square
			dw = (new_shape - new_unpad[0]) / 2  # width padding
			dh = (new_shape - new_unpad[1]) / 2  # height padding
		elif mode is 'rect':  # square
			dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
			dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
		elif mode is 'scaleFill':
			dw, dh = 0.0, 0.0
			new_unpad = (new_shape, new_shape)
			ratiow, ratioh = new_shape / shape[1], new_shape / shape[0]

		top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
		left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
		img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)  # resized, no border
		img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
		return img, ratiow, ratioh, dw, dh


	def predict(self,img):


		im0 = img
		img, *_ = self.letterbox(im0, new_shape=self.img_size)

		# Normalize RGB
		img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
		img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
			
		img = torch.from_numpy(img).unsqueeze(0).to(self.device)
		pred, _ = self.model(img)
		det = non_max_suppression(pred, self.conf_thres, self.nms_thres)[0]
		points = []
		if det is not None and len(det) > 0:
			det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
			points_per_image =[]
			for *xyxy, conf, cls_conf, cls in det:
				points_per_image = [xyxy[0].item(),xyxy[1].item(),xyxy[2].item(),xyxy[3].item()]
				print(points_per_image)
				points.append(points_per_image)
		return points








# if __name__ == "__main__":
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
# 	parser.add_argument('--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
# 	parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
# 	# parser.add_argument('--images', type=str, default='data/samples', help='path to images')


# 	opt = parser.parse_args()

# 	dg = YoloDetection(opt.cfg,
# 	           opt.data_cfg,
# 	           opt.weights)
# 	with torch.no_grad():
# 		points = dg.predict("data/samples/zidane.jpg")
# 		dg.attentionMask("data/samples/zidane.jpg",points)

