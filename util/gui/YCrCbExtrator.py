import cv2
import numpy as np

def call_back(x):
    pass

def create(name):
    image = cv2.namedWindow(name)
    cv2.moveWindow(name, 1000,0)
    cv2.createTrackbar('Y_low', name, 0, 255, call_back)
    cv2.createTrackbar('Y_high', name, 255, 255, call_back)
    cv2.createTrackbar('Cr_low', name, 0, 255, call_back)
    cv2.createTrackbar('Cr_high', name, 255, 255, call_back)
    cv2.createTrackbar('Cb_low', name, 0, 255, call_back)
    cv2.createTrackbar('Cb_high', name, 255, 255, call_back)

def extract(name):
    Y_low = cv2.getTrackbarPos('Y_low', name)
    Y_high = cv2.getTrackbarPos('Y_high', name) 
    Cr_low = cv2.getTrackbarPos('Cr_low', name)
    Cr_high = cv2.getTrackbarPos('Cr_high', name) 
    Cb_low = cv2.getTrackbarPos('Cb_low', name)
    Cb_high = cv2.getTrackbarPos('Cb_high', name) 

    return (
        (Y_low, Cr_low, Cb_low),
        (Y_high, Cr_high, Cb_high) )
