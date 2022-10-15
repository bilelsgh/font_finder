import cv2
from pytesseract import pytesseract
from dotenv import load_dotenv
import os

load_dotenv("../.env")
TESSERACT = os.getenv("tesseract")
HEIGHT = os.getenv("height")
WIDTH = os.getenv("width")


def to_binary(image):
    """
    Transform a colored image in a binary image (black and white)
    :param image: Path to the image
    :return: A binary image
    """
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binary image
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    return thresh1


