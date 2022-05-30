# VITA, EPFL

# import cv2
import socket
import sys
import numpy
import struct
import binascii

from PIL import Image
from detector import Detector
import argparse

counter = 0

while counter < 3:
    if True:
        testimage = Image.open("/home/group17/DLAV-2022/projects/loomo/darknet/data/testimage2.jpeg")

        #######################
        # Detect
        #######################
        bbox, bbox_label = Detector(testimage)

        if bbox_label:
            print("BBOX: {}".format(bbox))
            print("BBOX_label: {}".format(bbox_label))
        else:
            print("False")
            print("BBOX: {}".format(bbox))
            print("BBOX_label: {}".format(bbox_label))
        
        counter += 1

print("Check completed")