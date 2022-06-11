import cv2 as cv
import numpy as np
import cmake
import dlib

frontal_face_detector=dlib.get_frontal_face_detector()
frontal_shape_detector=dlib.shape_predictor("dataset/shape_predictor_68_face_landmarks")
imu=cv.imread("./image/imu.jpg",0)
cv.imshow("Imran Khan",imu)