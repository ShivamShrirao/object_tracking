import cv2
import numpy as np

# redLow = np.array([166, 186, 58])
# redHigh = np.array([180, 231, 192])
Low = np.array([0, 89, 105])
High = np.array([180, 144, 234])
cam = cv2.VideoCapture(0)

while True:
	ret, img = cam.read()
	# if img is None:
		# break
	img = cv2.flip(img, 1)
	blurred = cv2.GaussianBlur(img, (7, 7), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, Low, High)
	ret,im_th = cv2.threshold(mask,150,255,cv2.THRESH_BINARY)
	im_th = cv2.erode(im_th,None,iterations = 1)
	im_th = cv2.dilate(im_th,None,iterations = 1)

	ctrs,_ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if len(ctrs) > 0:
		for c in ctrs:
			if cv2.contourArea(c) > 6500:
				# hull = cv2.convexHull(c)
				cv2.drawContours(img,[c],-1,(0,0,255),2)

	cv2.imshow("img",img)
	cv2.imshow("th",im_th)

	key = cv2.waitKey(30)
	if key == ord('q') or key == 27:
		break

cam.release()
cv2.destroyAllWindows()