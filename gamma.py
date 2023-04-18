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
from ex1_utils import LOAD_GRAY_SCALE
import cv2
import numpy as np

def gammaDisplay(img_path: str, rep: int) -> None:
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    # read the file
    path = cv2.imread(img_path)

    #if representation is grayscale(1)
    if rep == 1:
        path = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY)
    # if representation is rgb(2)
    else:
        path = cv2.cvtColor(path, cv2.COLOR_BGR2RGB)

    def tool_bar(val):
        Gamma = (0.02 + val / 100.0)
        new_img = pow(path/255, Gamma)
        cv2.imshow('Gamma bar', new_img)

    cv2.namedWindow('Gamma bar')
    # values should be from 0 to 2 with resolution 0.01 but i'm using int so the presentation will be 0-100
    cv2.createTrackbar('Gamma', 'Gamma bar', 100, 200, tool_bar)
    cv2.imshow('Gamma bar', path)
    cv2.waitKey(0)

def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
