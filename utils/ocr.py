from operator import contains
import pytesseract
import cv2

from constants.license_plate import LICENSE_ID_CHARS
from utils.image import mask_image, warp_image


def image_to_license_id(im, debug=False, show=False, thres=160):
# > This function return license ID that detected from the image.
  s = 'Image to license ID: '
  license_id = ''
  try:
  # Show input image.
    if show and debug: cv2.imshow(f"{s}Input Image", im)

  # Step 1: Optimize image for OCR.
    imo = im.copy() # Copy input image.
    # imo = mask_image(imo, debug, show) # Masking image.
    imo = warp_image(imo, debug, show, thres) # Warping image.
    if show and debug: cv2.imshow(f"{s}Optimized Image", imo)

  # Step 2: Convert image to RGB.
    imrgb = cv2.cvtColor(imo, cv2.COLOR_BGR2RGB)
    if show: cv2.imshow(f"{s}OCR Image", imrgb)
    if show: cv2.waitKey(1)
  
  # Step 3: Retreive the first line of text from OCR.
    text = pytesseract.image_to_string(imrgb, lang="tha-license-plate")
    text = text.split('\n')[0] # Get the first line.
  
  # Step 4: Check is a valid License ID.
    if text is None or len(text) == 0: raise # Raise if no string.
    
    is_contain_digit = False # Check if it is License ID not the province. (Has number)
    
    for char in text:
      if char.isdigit():
        is_contain_digit = True
        break
    if not is_contain_digit: raise

  # Step 5: Format only valid characters.
    for char in text:
      if contains(LICENSE_ID_CHARS, char):
        license_id += char
  finally:
  # Return result.
    if len(license_id) == 0:
      return False, None
    else:
      return True, license_id