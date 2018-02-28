import cv2
import numpy as np

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

img = cv2.imread("gate.png", cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
print img.shape
bg = (255,255,255)

cv2.imwrite("gate_trans.png", strip_bg(bg, img))