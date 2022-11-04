from operator import contains
import pytesseract
import cv2

from constants.license_plate import LICENSE_NUMBER_CHARS
from utils.image import warp_image


def image_to_license_number(im):
    # > This function return license ID that detected from the image.
    s = 'Image to license ID: '
    license_number = ''
    try:
        # Step 1: Convert image to RGB.
        imrgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # Step 2: Retreive the first line of text from OCR.
        text = pytesseract.image_to_string(imrgb, lang="tha-license-plate")
        text = text.split('\n')[0]  # Get the first line.

    # Step 4: Check is a valid License ID.
        if text is None or len(text) == 0:
            raise  # Raise if no string.

        # Check if it is License ID not the province. (Has number)
        is_contain_digit = False

        for char in text:
            if char.isdigit():
                is_contain_digit = True
                break
        if not is_contain_digit:
            raise

    # Step 5: Format only valid characters.
        for char in text:
            if contains(LICENSE_NUMBER_CHARS, char):
                license_number += char
    finally:
        # Return result.
        if len(license_number) == 0:
            return False, None
        else:
            return True, license_number
