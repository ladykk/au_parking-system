import cv2
import numpy as np
from math import sqrt, floor


def warp_image(
    im,  # Input image.
):
    # > Return warped image of the input image.
    s = 'warp_image: '
    imgray, imthres, imcontour, imcorner, imwarp = None, None, None, None, None
    try:
        # Step 1: Convert Image to Grayscale.
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Step 2: Threshold the Image.
        imthres = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 51, 2)
        # imthres = cv2.adaptiveThreshold(
        #     imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 311, 1)
    # Step 3: Contours the Image.
        contours, _ = cv2.findContours(
            imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        imcontour = im.copy()
        imcontour = cv2.drawContours(imcontour, contours, -1, (255, 0, 255), 4)
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
        if len(biggest) == 0:
            raise  # Raise if cannot find a rectriangle in the Image.
    # Step 4.2: Reshape biggest are to be retriangle.
        biggest = biggest.reshape((4, 2))
        points = np.zeros((4, 1, 2), dtype=np.int32)
        add = biggest.sum(1)
        diff = np.diff(biggest, axis=1)
        points[0] = biggest[np.argmin(add)]
        points[1] = biggest[np.argmin(diff)]
        points[2] = biggest[np.argmax(diff)]
        points[3] = biggest[np.argmax(add)]

    # Step 4.3: Draw lines and points on the image.
        cv2.line(imcorner, (points[0][0][0], points[0][0][1]),
                 (points[1][0][0], points[1][0][1]), (0, 255, 0), 2)
        cv2.line(imcorner, (points[0][0][0], points[0][0][1]),
                 (points[2][0][0], points[2][0][1]), (0, 255, 0), 2)
        cv2.line(imcorner, (points[3][0][0], points[3][0][1]),
                 (points[2][0][0], points[2][0][1]), (0, 255, 0), 2)
        cv2.line(imcorner, (points[3][0][0], points[3][0][1]),
                 (points[1][0][0], points[1][0][1]), (0, 255, 0), 2)
        imcorner = cv2.drawContours(imcorner, points, -1, (255, 0, 255), 15)
    # Step 5: Warps the Image.
    # Calculate new width.
        w_x1, w_x2, w_y1, w_y2 = points[0][0][0], points[1][0][0], points[0][0][1], points[1][0][1]
        width = floor(sqrt(((w_x2 - w_x1) ** 2) + ((w_y2 - w_y1) ** 2)))
    # Calculate new height.
        h_x1, h_x2, h_y1, h_y2 = points[0][0][0], points[2][0][0], points[0][0][1], points[2][0][1]
        height = floor(sqrt(((h_x2 - h_x1) ** 2) + ((h_y2 - h_y1) ** 2)))
    # Warp the image.
        warp_point1 = np.float32(points)
        warp_point2 = np.float32(
            ((0, 0), (width, 0), (0, height), (width, height)))
        perspective_transaform = cv2.getPerspectiveTransform(
            warp_point1, warp_point2)
        imwarp = cv2.warpPerspective(
            imthres, perspective_transaform, (width, height))
    finally:
        return im, imgray, imthres, imcontour, imcorner, imwarp
