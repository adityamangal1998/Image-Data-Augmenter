import os
import numpy as np
import cv2
import time
import glob  # to read files
import json  # to write the classes text file for yolo
import xml.etree.ElementTree as ET  # to read the xml data from annotations files of images
from tqdm import tqdm  # to check the progress
import matplotlib.pyplot as plt
from scipy import ndimage
import shutil
import ntpath
import os
from pathlib import Path
import glob

def brightness():
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Adjust the brightness (increase or decrease)
    brightness_factor = 1.5  # Increase brightness by 50%; use values less than 1 to decrease

    # Multiply the brightness channel by the brightness factor
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * brightness_factor, 0, 255)

    # Convert the image back to the BGR color space
    brightened_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return brightened_image

def sharpen(image):
    """
    To sharpen the effects of the object in the image
    1. Define Kernel to sharpen (https://setosa.io/ev/image-kernels/)
    2. Using filter2D() for sharpening
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image.copy(), -1, kernel)


def contrast(image):
    """
    To imporve the contrast effect of the images.
    1. Convert image to gray scale
    2. Perform Equlization Hist, to improve the contrast of the image
    """
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def blur(image):
    """
    To remove noise from image.
    As most of the Image data have low light intensity, Gaussain Blur is used, to remove noise without affecting
    the image

    kernel size = 15,15 is used as the image is large. This value can be increased or decreased based on the
    desired output.
    """
    return cv2.GaussianBlur(image.copy(), (15, 15), 0)


def rotate(image, angle=135):
    """
    This function focuses on rotating the image to a particular angle.
    """
    height, width = image.shape[:2]
    img_c = (width / 2, height / 2)  # Image Center Coordinates

    rotation_matrix = cv2.getRotationMatrix2D(img_c, angle, 1.)  # Rotating Image along the actual center

    abs_cos = abs(rotation_matrix[0, 0])  # Cos(angle)
    abs_sin = abs(rotation_matrix[0, 1])  # sin(angle)

    # New Width and Height of Image after rotation
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract the old image center and add the new center coordinates
    rotation_matrix[0, 2] += bound_w / 2 - img_c[0]
    rotation_matrix[1, 2] += bound_h / 2 - img_c[1]

    # rotating image with transformed matrix and new center coordinates
    rotated_matrix = cv2.warpAffine(image.copy(), rotation_matrix, (bound_w, bound_h), borderValue=(255, 255, 255))

    return rotated_matrix


def crop(image=None, is_center_crop=True, crop_width=10, crop_height=10, x_start=None, x_end=None, y_start=None,
         y_end=None):
    # Get the dimensions of the image
    height, width = image.shape[:2]
    x_start = 0
    y_start = 0
    y_end = height // 2
    x_end = width // 2
    if is_center_crop:
        # Calculate the center of the image
        center_x, center_y = width // 2, height // 2
        # Calculate the region of interest (ROI) for center cropping
        x_start = center_x - crop_width // 2
        y_start = center_y - crop_height // 2
        x_end = center_x + crop_width // 2
        y_end = center_y + crop_height // 2

        # Crop the image using array slicing
        center_cropped_image = image[y_start:y_end, x_start:x_end]
    else:
        # Crop the image using array slicing
        center_cropped_image = image[y_start:y_end, x_start:x_end]
    return center_cropped_image


# Scaling
def scale(image=None, new_width=300, new_height=300, scale_factor=None):
    try:
        height, width = image[:2]
        if scale_factor is not None:
            new_width = width * scale_factor
            new_height = height * scale_factor
        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image
    except Exception as e:
        print(f"Error : {e}")


# Flipping
def flipping(image=None, is_horizontal=True, is_vertical=False):
    if is_horizontal:
        # Flip the image horizontally
        flipped_image = cv2.flip(image, 1)
    elif is_vertical:
        # Flip the image vertically
        flipped_image = cv2.flip(image, 0)
    else:
        # Flip the image both horizontally and vertically
        flipped_image = cv2.flip(image, -1)
    return flipped_image

# Translation
def translation():
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Define the translation distances along the x and y axes
    tx = 50  # Translation in the x-axis
    ty = 30  # Translation in the y-axis

    # Create separate translation matrices for x and y axes
    translation_matrix_x = np.float32([[1, 0, tx], [0, 1, 0]])
    translation_matrix_y = np.float32([[1, 0, 0], [0, 1, ty]])

    # Apply the translations using cv2.warpAffine() for x and y axes
    translated_image_x = cv2.warpAffine(image, translation_matrix_x, (width, height))
    translated_image_y = cv2.warpAffine(image, translation_matrix_y, (width, height))