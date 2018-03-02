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
        self.network_width = 10
        self.mystery_1 = 18
        self.mystery_2 = 20
        self.mystery_3 = 20
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, self.network_width, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(self.network_width, self.mystery_1, 5)
        self.fc1 = nn.Linear(self.mystery_1 * self.mystery_2 * self.mystery_3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.mystery_1 * self.mystery_2 * self.mystery_3)
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

	for roi in rois:
		if len(sample) < 4:
			roi['image'] = cv2.cvtColor(roi['image'], cv2.COLOR_BGRA2BGR)
			roi['image'] = np.transpose(roi['image'], (2,0,1))

			roi['image'] = torch.from_numpy(roi['image']).float()

			sample.append(roi)
		else:

			t = torch.cat((sample[0]['image'].unsqueeze(0), sample[1]['image'].unsqueeze(0), sample[2]['image'].unsqueeze(0), sample[3]['image'].unsqueeze(0)), dim=0)
			outputs = run_model(t, model)

			_, predicted = torch.max(outputs.data, 1)
   			predicted = predicted + 1
   			for i in range(0, 4):
   				sample[i]['prediction'] = predicted[i]

   	for roi in rois:
   		cv2.rectangle(img,(left, top),(right, bottom),(0,255,0),3)



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
#scene = scenes[random.randint(0, len(scenes) - 1)]
scene = "223.png"
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
	cv2.drawContours(scene_img, contours, -1, (0,255,0), 3)

	cv2.imshow('circles', scene_img)
	cv2.waitKey(0)