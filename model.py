import cv2

from skimage.filters import threshold_local
import numpy as np
import imutils
import os

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

# from PIL import Image,ImageStat
# import math
# def brightness( im_file ):
#    im = Image.open(im_file)
#    stat = ImageStat.Stat(im)
#    r,g,b = stat.mean
#    return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
# brightness_value = brightness('/content/drive/My Drive/s/IMG_1123.jpg')
# print(brightness_value)
# from google.colab.patches import cv2_imshow
# import cv2
# img =cv2.imread('/content/drive/My Drive/s/IMG_1123.jpg')
# add = img + img
# cv2_imshow(img)
# cv2_imshow(add)

for filename in os.listdir('.\Bills'):
	if filename.endswith(".jpg") or filename.endswith(".png"):
		print(filename)
		image = cv2.imread('C:\\Users\\Shubham Vaity\\Desktop\\Hackathon\\TEST 2\\Bills\\'+filename)
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		ratio = image.shape[0] / 500.0
		orig = image.copy()
		image = imutils.resize(image, height = 500)
		# # convert the image to grayscale, blur it, and find edges
		# # in the image
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		bilateral = cv2.bilateralFilter(gray, 15, 75, 75) 
		ret3,edged = cv2.threshold(bilateral,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		# # show the original image and the edge detected image
		# print("STEP 1: Edge Detection")
		# cv2_imshow(image)
		# cv2_imshow(edged)
		cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
		# # loop over the contours
		c = cnts[0]
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		# if our approximated contour has four points, then we
		# can assume that we have found our screen
		if len(approx) == 4:
		  screenCnt = approx
		  #print("Screen Cnt is chosen from approx")
		else:
		  (x,y,w,h) = cv2.boundingRect(cnts[0])
		  vertices = [[x,y],[x,y+h],[x+w,y+h],[x+w,y]]
		  screenCnt = np.asarray(vertices)
		  #print(screenCnt)
		  #print("Screen Cnt is chosen from boundingRect")
		# show the contour (outline) of the piece of paper
		#print("STEP 2: Find contours of paper")
		#cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
		#cv2.imshow("Image",image)
		#cv2.waitKey(0)
		warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
		# convert the warped image to grayscale, then threshold it
		# to give it that 'black and white' paper effect
		warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
		T = threshold_local(warped, 11, offset = 10, method = "gaussian")
		warped = (warped > T).astype("uint8") * 255
		# show the original and scanned images
		#print("STEP 3: Apply perspective transform")
		#cv2.imshow("Warped",imutils.resize(warped, height = 650))
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
		path = "C:\\Users\\Shubham Vaity\\Desktop\\Hackathon\\TEST 2\\op2\\"
		cv2.imwrite(os.path.join(path ,filename), warped)
# cv2_imshow(imutils.resize(warped))










