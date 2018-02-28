import cv2
import numpy as np

PADDING = 10

def strip_bg(bg, img):
	c1,c2,c3,c4 = cv2.split(img)

	c1_inv, c2_inv, c3_inv  = np.where(c1 == bg[0], 1, 0), np.where(c2 == bg[1], 1, 0), np.where(c3 == bg[2], 1, 0)
	alpha_and = np.zeros(c1.shape, dtype=np.uint8)
	alpha_and = np.where(np.logical_and(c1_inv, c2_inv), 1, 0)
	alpha_and = np.where(np.logical_and(alpha_and, c3_inv), 1, 0)
	new_alpha = np.where(alpha_and, 0, 255)
	new_alpha = np.array(new_alpha, dtype=np.uint8)

	print new_alpha, "\n\n", c1
	return cv2.merge((c1, c2, c3, new_alpha))

def crop(img, min_thresh=np.array((0,0,0,255)), max_thresh=np.array((255,255,255,255))):
	mask = cv2.inRange(img,min_thresh, max_thresh)
	#mask = np.where(mask == 0, 255, 0)
	im2, contours,hierarchy = cv2.findContours(mask, 1, 2)
	#cv2.drawContours(img, contours, -1, (0,255,0), 3)

	cnt = contours[0]
	perimeter = cv2.arcLength(cnt, True)
	epsilon = 0.01 * perimeter
	bb_c = cv2.approxPolyDP(cnt, epsilon, True)
	min_x, max_x, min_y, max_y = get_rect(bb_c)
	cv2.drawContours(img, [bb_c], -1, (0,0,255), 3)

	cv2.rectangle(img, (min_y, min_x), (max_y, max_x), (0,255,0))
	cv2.imshow("win", img)
	cv2.waitKey(0)

	max_dim = max(max_x - min_x, max_y - min_y)
	print "Max dim: %i" % (max_dim)
#	cropped_img = img[min_y - PADDING : max_y + PADDING, min_x - PADDING : max_x + PADDING]
	cropped_img = img[min_x - PADDING : max_x + PADDING, min_y - PADDING : max_y + PADDING]
	
	cv2.imshow("win", cropped_img)
	cv2.waitKey(0)
	
	cropped_img = cv2.resize(cropped_img, (400,400))
	return cropped_img

def get_rect(bounding_box):
	min_x = 99999
	min_y = 99999
	max_y = -1
	max_x = -1
	for point in bounding_box:
		print point
		min_x = min(min_x, point[0][1])
		max_x = max(max_x, point[0][1])
		max_y = max(max_y, point[0][0])
		min_y = min(min_y, point[0][0])
	return min_x, max_x, min_y, max_y

img = cv2.imread("dice_no_bg/1_image18_nbg.png", cv2.IMREAD_UNCHANGED)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
#print img
bg = (255,255,255)
img = crop(img)

cv2.imwrite("cropped_dice/3_image323_nbg.png", img)