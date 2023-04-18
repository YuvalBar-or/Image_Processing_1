"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import cv2
import matplotlib.pyplot as plt
import numpy as np
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 214329633


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    # read the file
    path = cv2.imread(filename)

    #if representation is grayscale(1)
    if representation == 1:
        path = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY)
    # if representation is rgb(2)
    if representation == 2:
        path = cv2.cvtColor(path, cv2.COLOR_BGR2RGB)
    #normalize the pics:
    # first, convert the image to floating point values
    img_float = np.float32(path)
    #normalize the pixels
    img_normalized = cv2.normalize(img_float, None, 0.0, 1.0, cv2.NORM_MINMAX)

    return img_normalized

def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    # load the pics from the previous function
    img= imReadAndConvert(filename , representation)
    #display the image
    # if grayscale
    if representation == 1:
        plt.imshow(img, cmap='gray')
    # if rgb
    else:
        plt.imshow(img)
    plt.figure()
    plt.show()

def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    # define the YIQ matrix
    YIQ_matrix = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.274, -0.322],
                           [0.211, -0.523, 0.312]])
    # Matrix multiplication
    YIQ_img = np.dot(imgRGB, YIQ_matrix.transpose())
    return YIQ_img

def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    RGB_matrix = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.274, -0.322],
                           [0.211, -0.523, 0.312]])
    # Matrix multiplication
    inverse_matrix = np.linalg.inv(RGB_matrix)
    RGB_img = np.dot(imgYIQ, inverse_matrix.transpose())
    return RGB_img

def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    # check if the pic is rgb
    is_rgb =False
    if len(imgOrig.shape) == 3 and imgOrig.shape[2] == 3:
        is_rgb = True
        imgOrig_RGB = transformRGB2YIQ(imgOrig)
        imgOrig = imgOrig_RGB[:, :, 0]

    # normalize from 0-1 to 0-255
    imgOrig = cv2.normalize(imgOrig, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #look up table
    imgOrig_hist = np.bincount(imgOrig.flatten(), minlength=256)
    num_pixels = np.sum(imgOrig_hist)
    histogram_array = imgOrig_hist / num_pixels
    histogram_array_cumsum = np.cumsum(histogram_array)

    transform_map = np.floor(255 * histogram_array_cumsum).astype(np.uint8)
    img_list = list(imgOrig.flatten())
    # transform pixel values to equalize
    eq_img_list = [transform_map[i] for i in img_list]
    # reshape and write back into img_array
    equalized_img_array = np.reshape(np.asarray(eq_img_list), imgOrig.shape)
    returnHistogram = np.bincount(equalized_img_array.flatten(), minlength=256)

    #if the pic was rgb-
    if is_rgb:
        imgOrig_RGB[:, :, 0] = equalized_img_array / 255
        equalized_img_array = transformYIQ2RGB(imgOrig_RGB)

    return equalized_img_array, imgOrig_hist, returnHistogram


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    # check if the pic is rgb
    shape = imOrig.shape
    is_RGB = False
    img_cpy = imOrig
    if len(shape) > 2:
        is_RGB = True
        imOrig_rgb = transformRGB2YIQ(img_cpy)
        img_cpy = imOrig_rgb[:, :, 0]

    # initialize the lists we will return
    images = []
    error = []

    # normalize from 0-1 to 0-255 and hist
    img_cpy = cv2.normalize(img_cpy, None, 0, 255, cv2.NORM_MINMAX)
    imOrig_flat = img_cpy.ravel().astype(int)
    hist = np.zeros(256)
    for pix in imOrig_flat:
        hist[pix] += 1

    # borders
    borders = np.zeros(nQuant + 1, dtype = int)
    borders_counter = 0
    for i in range(nQuant + 1):
        borders[i] = i * (255.0 / nQuant)
    borders[-1] = 256

    # runnig the main loop nIter times
    for i in range(nIter):
        array_q = np.zeros(nQuant, dtype=int)
        # calculate each q with the formula
        for j in range(nQuant):
            q = hist[borders[j]:borders[j + 1]]
            rng = np.arange(int(borders[j]), int(borders[j + 1]))
            array_q[j] = (rng * q).sum() / (q.sum()).astype(int)
        # borders again
        z_first = borders[0]
        z_last = borders[-1]
        borders = np.zeros_like(borders)
        for k in range(1, nQuant):
            borders[k] = (array_q[k - 1] + array_q[k]) / 2
        borders[0] = z_first
        borders[-1] = z_last

        # recolor image
        temp_img = np.zeros_like(img_cpy)
        for h in range(nQuant):
            z_temp = borders[h]
            temp_img[img_cpy > z_temp] = array_q[h]
        images.append(temp_img)

        # MSE -error
        error.append(((img_cpy - temp_img) ** 2).mean())
        if len(error) > 1 and abs(error[-2] - error[-1]) < 0.001:
            break

    # if picture is RGB return it from YIQ to RGB before adding it to the list of images.
    if is_RGB is True:
        for i in range(len(images)):
            imOrig_rgb[:, :, 0] = images[i] / 255
            images[i] = transformYIQ2RGB(imOrig_rgb)

            images[i][images[i] > 1] = 1
            images[i][images[i] < 0] = 0
    return images, error






