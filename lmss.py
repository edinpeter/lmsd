import cv2
import os
import random
import numpy as np
from skimage import color

backgrounds = os.listdir("backgrounds/")
dice = os.listdir("dice_no_bg_squared/")
bg_count = len(backgrounds)
dice_count = len(dice)

scenes = 10
dice_size = 70

def blur(image):
	image2 = image
	for i in range(random.randint(0,1)):
		blur_amount = random.randint(1,7)
		image2 = cv2.blur(image2, (blur_amount,blur_amount))
	return image2

def die_color(image):
	print image.shape
	blue = np.array([np.array([[255,150,0, 255]] * image.shape[1])] * image.shape[0], dtype = image.dtype)
	print blue.shape
	return cv2.addWeighted(image, 0.45, blue, 0.55, 0)

def color_(image):
	print image.shape
	blue = np.array([np.array([[255,150,150, 255]] * image.shape[1])] * image.shape[0], dtype = image.dtype)
	print blue.shape
	return cv2.addWeighted(image, 0.5, blue, 0.5, 0)

for i in range(len(backgrounds)):
	try:
		bg = backgrounds[random.randint(0, bg_count - 1)]
		dice_subset = list()
		for j in range(random.randint(1,6)):
			dice_subset.append(dice[random.randint(0, dice_count - 1)])


		bg_image = cv2.imread("backgrounds/" + bg, cv2.IMREAD_UNCHANGED)
		bg_image_original = cv2.imread("backgrounds/" + bg, cv2.IMREAD_UNCHANGED)
		bg_image = cv2.resize(bg_image, (640, 480))
		bg_image_original = cv2.resize(bg_image, (640,480))

		bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2BGRA)
		print bg_image.shape
		for die in dice_subset:
			print die
			die_image = cv2.imread("dice_no_bg_squared/" + die, cv2.IMREAD_UNCHANGED)
			#die_image = cv2.cvtColor(die_image, cv2.COLOR_BGR2BGRA)
			#cv2.imshow("win", die_image)
			#cv2.waitKey(0)

			die_image = cv2.resize(die_image, (dice_size, dice_size))
			die_image = die_color(die_image)
			die_image = blur(die_image)
			mask = np.ones(die_image.shape, die_image.dtype) * 255
			print mask
			print mask.shape
			print die_image

			#center = (random.randint(0, bg_image.shape[0]), random.randint(0, bg_image.shape[1]))

			#bg_image = cv2.seamlessClone(die_image, bg_image, mask, center, cv2.NORMAL_CLONE)
			#bg_image_height = 
			bg_x_left = random.randint(0, bg_image.shape[1] - dice_size - 1)
			bg_y_top = random.randint(0, bg_image.shape[0] - dice_size - 1)
			print bg_x_left
			upper_x = bg_x_left + dice_size

			print bg_y_top
			upper_y = bg_y_top + dice_size

			bg_image[bg_x_left : upper_x, bg_y_top : upper_y] = die_image

			c1,c2,c3,alpha = cv2.split(bg_image)
			c1_og, c2_og, c3_og = cv2.split(bg_image_original)
			c1 = np.where(alpha < 255, c1_og, c1)
			c2 = np.where(alpha < 255, c2_og, c2)
			c3 = np.where(alpha < 255, c3_og, c3)
			alpha = np.where(alpha < 255, 255, alpha)
			#c1 = np.where(alpha == 255, c1, c1_og)
			#c2 = np.where(alpha == 255, c2, c2_og)
			#c3 = np.where(alpha == 255, c3, c3_og)
			bg_image = cv2.merge((c1,c2,c3,alpha))
			bg_image = blur(bg_image)



		#bg_image = color_(bg_image)

		#cv2.imshow("win", bg_image)
		#cv2.waitKey(0)

		cv2.imwrite("scenes/" + bg, bg_image)
	except:
		pass