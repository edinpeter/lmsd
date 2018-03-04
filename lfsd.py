import cv2
import numpy as np
import os
import random
import torch
from sets import Set
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import time



class Net(nn.Module):
    def __init__(self):
        self.conv1_kernel = 10
        self.conv2_kernel = 5

    	self.network_width = 20 #out channels
        self.conv2_output_channels = 18
        self.mystery_2 = 20
        self.mystery_3 = 20

        self.outputs = 6

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, self.network_width, self.conv1_kernel)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(self.network_width, self.conv2_output_channels, self.conv2_kernel)
        self.fc1 = nn.Linear(self.conv2_output_channels * self.mystery_2 * self.mystery_3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.outputs)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.conv2_output_channels * self.mystery_2 * self.mystery_3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def get_circles(scene_img):
	scene_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
	circles = cv2.HoughCircles(scene_gray, cv2.HOUGH_GRADIENT, 1, 1, param1=35, param2=35, minRadius=1, maxRadius=20)
	return circles

def thresh_contours(scene_img):
	imgray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
	im_gauss = cv2.GaussianBlur(imgray, (11, 11), 0)
	ret, thresh = cv2.threshold(im_gauss, 100, 255, 0)
	# get contours
	im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	return contours

def get_contour_classes(contours, img):
	print type(contours)
	free_agents = list(contours)
	clusters = list()
	contours = ditch_large_contours(contours)
	rois = list()
	for con in contours:
		M = cv2.moments(con)
		center = None
		if M['m00'] != 0:
			con_location_x = int(M['m10']/M['m00'])
			con_location_y = int(M['m01']/M['m00'])
			center = (con_location_x, con_location_y)


			top, bottom, left, right = 0, 0, 0, 0
			if center[0] - 50 < 0:
				left = 0
				right = center[0] + (100 - center[0])
			
			elif center[0] + 50 > img.shape[1] - 1:
				left = img.shape[1] - 100
				right = img.shape[1] - 1

			else:
				left = center[0] - 50
				right = center[0] + 50

			if center[1] - 50 < 0:
				top = 0
				bottom = center[1] + (100 - center[1])
			
			elif center[1] + 50 > img.shape[0] - 1:
				top = img.shape[0] - 100
				bottom = img.shape[0] - 1

			else:
				top = center[1] - 50
				bottom = center[1] + 50
			

			#print left, right, top, bottom, right - left, bottom - top, img.shape, center[0], center[1]
			roi = img[top : bottom, left : right]
			#print roi.shape
			roi = cv2.resize(roi, (100,100))

			data = dict()
			data['dims'] = (left, right, top, bottom)
			data['image'] = roi
			rois.append(data)
			#cv2.rectangle(img,(left, top),(right, bottom),(0,255,0),3)
	sample = list()

	model = torch.load('/home/peter/Desktop/ftfd/dice_classifier/classifier_cuda.pt')
	model_gray = torch.load('/home/peter/Desktop/ftfd/dice_classifier/classifier_cuda_gray.pt')
	rois_new = list()
	for roi in rois:
		if len(sample) < 4:
			roi_image_temp = np.array(roi['image'])
			roi['image'] = cv2.cvtColor(roi['image'], cv2.COLOR_BGRA2BGR)
			roi['image'] = np.transpose(roi['image'], (2,0,1))
			roi['image'] = torch.from_numpy(roi['image']).float()
			
			roi['image_gray'] = cv2.cvtColor(roi_image_temp, cv2.COLOR_BGRA2GRAY)
			roi['image_gray'] = np.expand_dims(roi['image_gray'], axis=0)

			roi['image_gray'] = torch.from_numpy(roi['image_gray']).float()

			sample.append(roi)
		else:
			if len(sample) < 4:
				while len(sample < 4):
					sample.append(sample[0])

			t = torch.cat((sample[0]['image'].unsqueeze(0), sample[1]['image'].unsqueeze(0), sample[2]['image'].unsqueeze(0), sample[3]['image'].unsqueeze(0)), dim=0)
			t_gray = torch.cat((sample[0]['image_gray'].unsqueeze(0), sample[1]['image_gray'].unsqueeze(0), sample[2]['image_gray'].unsqueeze(0), sample[3]['image_gray'].unsqueeze(0)), dim=0)

			print sample[0]['image']
			outputs = run_model(t, model)
			outputs_gray = run_model(t_gray, model_gray)


			_, predicted = torch.max(outputs.data, 1)
   			predicted = predicted + 1


			_, predicted_gray = torch.max(outputs_gray.data, 1)
   			predicted_gray = predicted_gray + 1

			soft = nn.Softmax(1)
			softed = soft(Variable(outputs.data))

			soft_gray = nn.Softmax(1)
			softed_gray = soft_gray(Variable(outputs_gray.data))

			print predicted
			print softed
   			for i in range(0, 4):

   				sample[i]['prediction'] = predicted[i]
   				sample[i]['confidence'] = softed.data[i][predicted[i] - 1]

   				sample[i]['prediction_gray'] = predicted[i]
   				sample[i]['confidence_gray'] = softed_gray.data[i][predicted_gray[i] - 1]

   				rois_new.append(sample[i])
   			sample = list()

   	for roi in rois_new:
   		#print roi['prediction'], roi['confidence']
   		print roi['confidence'], roi['confidence_gray']
   		if roi['confidence'] > 0.9:
   			cv2.rectangle(img,(roi['dims'][0], roi['dims'][2]),(roi['dims'][1], roi['dims'][3]),(0,0,255),3)
   		else:
   			cv2.rectangle(img,(roi['dims'][0], roi['dims'][2]),(roi['dims'][1], roi['dims'][3]),(0,255,0),3)
   			pass

   		if roi['prediction_gray'] != 7:# and roi['confidence'] > 0.7:
   			cv2.rectangle(img,(roi['dims'][0], roi['dims'][2]),(roi['dims'][1], roi['dims'][3]),(255,0,0),3)
   		else:
   			cv2.rectangle(img,(roi['dims'][0], roi['dims'][2]),(roi['dims'][1], roi['dims'][3]),(255,255,0),3)
   			pass

	return img

def run_model(tensor, model):
	soft = nn.Softmax(1)
	optim = Variable(tensor.cuda())
	return model(optim)



def ditch_large_contours(contours):
	ret = list()
	for con in contours:
		if cv2.contourArea(con) < 100 and cv2.contourArea(con) > 10:
			ret.append(con)
	return ret

def distance_contours(contour_1, contour_2):
	M = cv2.moments(contour_1)
	c1_location_x = int(M['m10']/M['m00'])
	c1_location_y = int(M['m01']/M['m00'])

	M = cv2.moments(contour_2)
	c2_location_x = int(M['m10']/M['m00'])
	c2_location_y = int(M['m01']/M['m00'])

	distance = (c1_location_x - c2_location_x) ** 2 + (c1_location_y - c2_location_y) ** 2
	return distance ** 0.5




scenes = os.listdir("scenes/")
scene = scenes[random.randint(0, len(scenes) - 1)]
#scene = "223.png"
scene_img = cv2.imread("scenes/" + scene, cv2.IMREAD_UNCHANGED)
#scene_img = cv2.medianBlur(scene_img, 5)

#circles = get_circles(scene_img)
circles = None

if circles is not None:
	circles = np.uint8(np.around(circles))

	for i in circles[0, :]:
		cv2.circle(scene_img,(i[0],i[1]),i[2],(0,255,0),2)
		cv2.circle(scene_img,(i[0],i[1]),2,(0,0,255),3)

	cv2.imshow('circles', scene_img)
	cv2.waitKey(0)

contours = thresh_contours(scene_img)
if contours is not None:
	scene_img = get_contour_classes(contours, scene_img)
	#cv2.drawContours(scene_img, contours, -1, (0,255,0), 3)

	cv2.imshow('circles', scene_img)
	cv2.waitKey(0)