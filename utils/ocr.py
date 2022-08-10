import pytesseract
import cv2

def image_to_license_id(input_image, debug=False):
  text = ''
  try:
    if debug: cv2.imshow("Input Image", input_image)
    rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    if debug: cv2.imshow("RGB Image", rgb_image)
    text = pytesseract.image_to_string(rgb_image, lang="tha-license-plate")
    text = text.split('\n')[0]
    if debug: cv2.waitKey(1)
  finally:
    return text if len(text) > 0 else None