import cv2
import numpy as np
import os

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

# Set this to True to enable image display (requires GUI)
DISPLAY_IMAGES = False

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	if DISPLAY_IMAGES:
		cv2.imshow("image", img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

def cd_color_segmentation(img, template=None):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	########## YOUR CODE STARTS HERE ##########

	# from GeeksForGeeks

	eroded_img = cv2.erode(img, np.ones((5, 5), np.uint8))
	# image_print(eroded_img)
	# image_print(cv2.erode(img, np.ones((10, 10), np.uint8)))
	dilated_img = cv2.dilate(eroded_img, np.ones((10,10), np.uint8))
	# image_print(dilated_img)
	hsv_version = cv2.cvtColor(dilated_img, cv2.COLOR_BGR2HSV)
    # image_print(hsv_version)

    # Taken from https://cvexplained.wordpress.com/2020/04/28/color-detection-hsv/#:~:text=The%20HSV%20values%20for%20true,10%20and%20160%20to%20180.
	lower1 = np.array([0, 100, 100])
	upper1 = np.array([10, 255, 255])
	
	# Add a second range for red (red wraps around in HSV)
	lower2 = np.array([160, 100, 100])
	upper2 = np.array([180, 255, 255])
	
	# Create masks for both red ranges
	mask1 = cv2.inRange(hsv_version, lower1, upper1)
	mask2 = cv2.inRange(hsv_version, lower2, upper2)
	
	# Combine the masks
	full_mask = cv2.add(mask1, mask2)
	
	# Apply morphological operations to clean up the mask
	kernel = np.ones((5,5), np.uint8)
	full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)
	full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)

	# image_print(orange_mask)
	masked_img = cv2.bitwise_and(img, img, dst=None, mask=full_mask)
	# image_print(masked_img)

	# Lower the detection threshold to detect smaller red areas
	DETECTION_BOUND = 3000

	contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# image_print(cv2.drawContours(masked_img, contours, -1, (255, 0, 0), 2))

	for i in range(len(contours)):
		this_contour = contours[i]
		contour_area = cv2.contourArea(this_contour)
		print(f"Contour {i} area: {contour_area}")
		if contour_area > DETECTION_BOUND:
			# x_coord, y_coord, width, height = cv2.boundingRect(this_contour)
			# image_print(cv2.rectangle(masked_img, (x_coord, y_coord), (x_coord + width, y_coord + height), (0, 255, 0), 2))
			print("Red light detected")
			return True

	print("No red light")
	return False

# img = cv2.imread("media/traffic_light.png")
# cd_color_segmentation(img)

# img = cv2.imread("media/red_light_off.png")
# cd_color_segmentation(img)
