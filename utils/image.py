import cv2
import numpy as np
from math import sqrt, floor
import imutils

def warp_image(
  im, # Input image.
  debug=False, # If debug is True, show windows of image of each steps. Else, 
  show=False, # If show is True, show image in the window.
  thres=160, # Threshold value.
):
# > Return warped image of the input image.
  s = 'warp_image: '
  try:
  # Show input image.
    if show and debug: cv2.imshow(f"{s}Input Image", im)

  # Step 1: Convert Image to Grayscale.
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if show and debug: cv2.imshow(f"{s}Grayscale Image", imgray)

  # Step 2: Threshold the Image.
    estimated_threshold, imthres = cv2.threshold(imgray, thres, 255, cv2.THRESH_BINARY)
    if show and debug: cv2.imshow(f"{s}Threshold Image", imthres)

  # Step 3: Contours the Image.
    contours, _ = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    imcontour = im.copy()
    imcontour = cv2.drawContours(imcontour, contours, -1, (255, 0, 255), 4)
    if show and debug: cv2.imshow(f"{s}Contour Image", imcontour)

  # Step 4: Add corner to the Image.
    imcorner = im.copy()

  # Step 4.1: Finding biggest area in the image.
    max_area = 0
    biggest = []
    for contour in contours:
    # Check each contour on the Image.
      area = cv2.contourArea(contour)
      if area > 400:
      # Check if area of the contour big enough?
        perimeter = cv2.arcLength(contour, True)
        edges = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if area > max_area and len(edges) == 4:
        # Replace current area if previous area is smaller.
          biggest = edges
          max_area = area
    if len(biggest) == 0: raise # Raise if cannot find a rectriangle in the Image.

  # Step 4.2: Reshape biggest are to be retriangle.
    biggest = biggest.reshape((4,2))
    points = np.zeros((4,1,2), dtype=np.int32)
    add = biggest.sum(1)
    diff = np.diff(biggest, axis=1)
    points[0] = biggest[np.argmin(add)]
    points[1] = biggest[np.argmin(diff)]
    points[2] = biggest[np.argmax(diff)]
    points[3] = biggest[np.argmax(add)]

  # Step 4.3: Draw lines and points on the image.
    cv2.line(imcorner, (points[0][0][0], points[0][0][1]), (points[1][0][0], points[1][0][1]), (0, 255, 0), 2)
    cv2.line(imcorner, (points[0][0][0], points[0][0][1]), (points[2][0][0], points[2][0][1]), (0, 255, 0), 2)
    cv2.line(imcorner, (points[3][0][0], points[3][0][1]), (points[2][0][0], points[2][0][1]), (0, 255, 0), 2)
    cv2.line(imcorner, (points[3][0][0], points[3][0][1]), (points[1][0][0], points[1][0][1]), (0, 255, 0), 2)
    imcorner = cv2.drawContours(imcorner, points, -1, (255, 0, 255), 15)
    if show and debug: cv2.imshow(f"{s}Corner Image", imcorner)

  # Step 5: Warps the Image.
  # Calculate new width.
    w_x1, w_x2, w_y1, w_y2 = points[0][0][0], points[1][0][0], points[0][0][1], points[1][0][1]
    width = floor(sqrt(((w_x2 - w_x1) ** 2) + ((w_y2 - w_y1) ** 2)))
  # Calculate new height.
    h_x1, h_x2, h_y1, h_y2 = points[0][0][0], points[2][0][0], points[0][0][1], points[2][0][1]
    height = floor(sqrt(((h_x2 - h_x1) ** 2) + ((h_y2 - h_y1) ** 2)))
  # Warp the image.
    warp_point1 = np.float32(points)
    warp_point2 = np.float32(((0, 0), (width, 0), (0, height), (width, height)))
    perspective_transaform = cv2.getPerspectiveTransform(warp_point1, warp_point2)
    imw = cv2.warpPerspective(imthres, perspective_transaform, (width,height))
    if show: cv2.imshow(f"{s}Warp Image", imw)
    if show: cv2.waitKey(1)
    return imw
  except:
    return im

def mask_image(
  im, # Input image.
  debug=False, # If debug is True, show windows of image of each steps.
  show=False, # If show is True, show image in the window.
):
# > return masked image of input image.
  s = 'mask_image: '
  try:
  # Show input image.
    if show and debug: cv2.imshow(f"{s}Input Image", im)

  # Step 1: Convert image to grayscale.
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if show and debug: cv2.imshow(f"{s}Grayscale Image", imgray)

  # Step 2: Apply filter and find edges for localization.
    bfilter = cv2.bilateralFilter(imgray, 11, 17, 17) # Noise reduction
    edged = cv2.Canny(bfilter, 30, 200) # edge detection
    if show and debug: cv2.imshow(f"{s}Edged Image", edged)

  # Step 3: Find contours and apply mask.
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
  # Find bigest location
    location = None
    for contour in contours:
      approx = cv2.approxPolyDP(contour, 10, True)
      if len(approx) == 4:
        location = approx
        break
  # Step 5: Masking the image
    mask = np.zeros(imgray.shape, np.uint8)
    imn = cv2.drawContours(mask, [location], 0, 255, -1)
    imn = cv2.bitwise_and(im, im, mask=mask)
    if show and debug: cv2.imshow("Masked Image", imn)

  # Step 5: Crop masking part from gray image.
    (x,y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = imgray[x1:x2+1, y1:y2+1]
    return cropped_image
  except:
    return im