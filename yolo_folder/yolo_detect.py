import argparse
import time
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *
import subprocess
import os 


class Detection: 

	def __init__(self,
				cfg,
				data_cfg,
				weights,
				nms_thres=0.5,
				conf_thres=0.5,
				save_txt=False,
				img_size=(640,480)):
		
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
		self.classes = load_classes(parse_data_cfg(data_cfg)['names'])
		# colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]


	def predict(self,img):
		dataloader = LoadImages(img, img_size=self.img_size)


		for i, (path, img, im0, vid_cap) in enumerate(dataloader):
			# Get detections
			img = torch.from_numpy(img).unsqueeze(0).to(self.device)
			pred, _ = self.model(img)
			det = non_max_suppression(pred, self.conf_thres, self.nms_thres)[0]

			save_path = 'prediction'
			for *xyxy, conf, cls_conf, cls in det:
				with open(save_path + '.txt', 'a') as file:
					file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))


# if __name__ == "__main__":
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
# 	parser.add_argument('--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
# 	parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
# 	# parser.add_argument('--images', type=str, default='data/samples', help='path to images')
# 	parser.add_argument('--img-size', type=int, default=(416), help='inference size (pixels)')

# 	opt = parser.parse_args()
# 	print(opt)

# 	dg = Detection(opt.cfg,
# 	           opt.data_cfg,
# 	           opt.weights,
# 	           img_size=opt.img_size)
# 	with torch.no_grad():
# 		dg.predict("data/samples")

