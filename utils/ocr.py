from operator import contains
import pytesseract
import easyocr
import cv2

from constants.license_plate import LICENSE_NUMBER_CHARS

reader = easyocr.Reader(['th'])


def image_to_license_number(im):
    # > This function return license ID that detected from the image.
    s = 'Image to license ID: '
    license_number = ''
    try:
        # Step 1: Convert image to RGB.
        imrgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # Step 2: Retreive text from OCR.
        texts = reader.readtext(imrgb, detail=0, add_margin=0.5)

    # Step 3: Filter out province.
        filtered_texts = []
        for text in texts:
            if len(filtered_texts) > 2:
                break  # if more than 2 texts.
            # has more than 2 characters -> filter out provice.
            elif len(text) > 7:
                continue
            elif len(text) > 2:
                is_contain_number = False
                for char in text:
                    if char.isdigit():
                        is_contain_number = True
                        break
                if is_contain_number:
                    filtered_texts.append(text)
            else:
                filtered_texts.append(text)

    # Step 4: Connect string.
        if len(filtered_texts) == 2:
            i_0 = False
            for char in filtered_texts[0]:
                if not char.isdigit():
                    i_0 = True
            filter_text = f'{filtered_texts[0]}{filtered_texts[1]}' if i_0 else f'{filtered_texts[1]}{filtered_texts[0]}'
        elif len(filtered_texts) == 1:
            filter_text = filtered_texts[0]

    # Step 5: Check is a valid License ID.
        if filter_text is None or len(filter_text) == 0:
            raise  # Raise if no string.

        # Check if it is License ID not the province. (Has number)
        is_contain_digit = False

        for char in filter_text:
            if char.isdigit():
                is_contain_digit = True
                break
        if not is_contain_digit:
            raise

    # Step 6: Format only valid characters.
        for char in filter_text:
            if contains(LICENSE_NUMBER_CHARS, char):
                license_number += char
    finally:
        # Return result.
        if len(license_number) == 0:
            return False, None
        else:
            return True, license_number
