import cv2
import numpy as np
from math import sqrt, floor

def calculate_dimension(points):
  # This function return width and height from the points.
  # Width
  w_x1 = points[0][0][0]
  w_x2 = points[1][0][0]
  w_y1 = points[0][0][1]
  w_y2 = points[1][0][1]
  width = floor(sqrt(((w_x2 - w_x1) ** 2) + ((w_y2 - w_y1) ** 2)))
  # Height
  h_x1 = points[0][0][0]
  h_x2 = points[2][0][0]
  h_y1 = points[0][0][1]
  h_y2 = points[2][0][1]
  height = floor(sqrt(((h_x2 - h_x1) ** 2) + ((h_y2 - h_y1) ** 2)))
  return width, height

def draw_rectriangle(image, points):
  # This function draws rectriangle on the image.
  cv2.line(image, (points[0][0][0], points[0][0][1]), (points[1][0][0], points[1][0][1]), (0, 255, 0), 2)
  cv2.line(image, (points[0][0][0], points[0][0][1]), (points[2][0][0], points[2][0][1]), (0, 255, 0), 2)
  cv2.line(image, (points[3][0][0], points[3][0][1]), (points[2][0][0], points[2][0][1]), (0, 255, 0), 2)
  cv2.line(image, (points[3][0][0], points[3][0][1]), (points[1][0][0], points[1][0][1]), (0, 255, 0), 2)

def warp_image(input_image, debug=False):
  # This function return warp_image of the input_image
  # It recieves 2 inputs: input_image, and debug (False)
  # If debug is True, show windows of image of each steps.

  # Step 0: Show input_image.
  if debug: cv2.imshow("Input Image", input_image)

  # Step 1: Convert Image to Grayscale.
  grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
  if debug: cv2.imshow("Grayscale Image", grayscale_image)

  # Step 2: Threshold the Image.
  estimated_threshold, threshold_image = cv2.threshold(grayscale_image, 160, 255, cv2.THRESH_BINARY)
  if debug: cv2.imshow("Threshold Image", threshold_image)

  # Step 3: Contours the Image.
  contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  contour_image = input_image.copy()
  contour_image = cv2.drawContours(contour_image, contours, -1, (255, 0, 255), 4)
  if debug: cv2.imshow("Contour Image", contour_image)

  # Step 4: Add corner to the Image.
  corner_image = input_image.copy()
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
  
  if len(biggest) == 0: return None # Return None if cannot find a rectriangle in the Image.

  biggest = biggest.reshape((4,2))
  points = np.zeros((4,1,2), dtype=np.int32)
  add = biggest.sum(1)
  diff = np.diff(biggest, axis=1)
  points[0] = biggest[np.argmin(add)]
  points[1] = biggest[np.argmin(diff)]
  points[2] = biggest[np.argmax(diff)]
  points[3] = biggest[np.argmax(add)]
  draw_rectriangle(corner_image, points)
  corner_image = cv2.drawContours(corner_image, points, -1, (255, 0, 255), 15)
  if debug: cv2.imshow("Corner Image", corner_image)

  # Step 5: Warps the Image.
  dimension = calculate_dimension(points)
  warp_point1 = np.float32(points)
  warp_point2 = np.float32(((0, 0), (dimension[0], 0), (0, dimension[1]), dimension))
  perspective_transaform = cv2.getPerspectiveTransform(warp_point1, warp_point2)
  warp_image = cv2.warpPerspective(threshold_image, perspective_transaform, dimension)
  if debug: cv2.imshow("Warp Image", warp_image)

  if debug: cv2.waitKey(1)
  return warp_image