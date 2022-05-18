# import dependencies
from IPython.display import display, Javascript, Image
from base64 import b64decode, b64encode
import cv2
import numpy as np
import PIL
import io
import html
import time
import matplotlib.pyplot as plt

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# import darknet functions to perform object detections
from darknet import *
# load in our YOLOv4 architecture network
network, class_names, class_colors = load_network("cfg/yolov4-csp.cfg", "cfg/coco.data", "yolov4-csp.weights")
width = network_width(network)
height = network_height(network)

# darknet helper function to run detection on image
def darknet_helper(img, width, height):
  darknet_image = make_image(width, height, 3)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_resized = cv2.resize(img_rgb, (width, height),
                              interpolation=cv2.INTER_LINEAR)

  # get image ratios to convert bounding boxes to proper size
  img_height, img_width, _ = img.shape
  width_ratio = img_width/width
  height_ratio = img_height/height

  # run model on darknet style image to get detections
  copy_image_from_bytes(darknet_image, img_resized.tobytes())
  detections = detect_image(network, class_names, darknet_image)
  free_image(darknet_image)
  return detections, width_ratio, height_ratio


doMatches = 0
bf = cv2.BFMatcher()
person_detected_before = None
init = 0

def imageProcess(frame):
	global width, height, doMatches, bf, person_detected_before, init
	detections, width_ratio, height_ratio = darknet_helper(frame, width, height)
  
    # loop through detections and draw them on transparent overlay image
    list_person = []
    frame_cropped = []
    nbrMatches = []
    for label, confidence, bbox in detections:
      if label == 'person' and float(confidence) > 0.75:
        points = np.asarray(bbox2points(bbox))
        # to correct the negative values of bbox pixels
        points[points < 0] = 0
        left, top, right, bottom = points
        left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(bottom * height_ratio)
        cropped_frame = frame[top:bottom, left:right]
        
        if doMatches == 0:   #-------POSE ESTIMATION--------
          
          with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.5) as pose:
              # Recolor image to RGB
              cropped_image_shape = np.array(cropped_frame.shape)
              image = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
              image.flags.writeable = False

              # Make detection
              results = pose.process(image)

              # Recolor back to BGR
              image.flags.writeable = True
              image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
              
              if results.pose_landmarks != None:
                left_shoulder_height = 1-results.pose_landmarks.landmark[11].y
                left_shoulder_visibility = results.pose_landmarks.landmark[11].visibility
                left_elbow_height = 1-results.pose_landmarks.landmark[13].y
                left_elbow_visibility = results.pose_landmarks.landmark[13].visibility
              else:
                left_shoulder_height, left_elbow_height = 0, 0
                left_shoulder_visibility, left_elbow_visibility = 0, 0

          if left_shoulder_visibility > 0.75 and left_elbow_visibility > 0.75 and left_elbow_height > left_shoulder_height:
            color = class_colors[label]
            detected = True
            person_detected_before = cropped_frame.copy()
            doMatches = 1
            print('Person Detected!')
          
        else:           #-------TRACKING--------
          init = 1 #initialization was done
          person_detected_before_gray = cv2.cvtColor(person_detected_before, cv2.COLOR_RGB2GRAY)
          cropped_frame_gray =  cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2GRAY)

          feature_extractor = cv2.xfeatures2d.SIFT_create() 

          # find the keypoints and descriptors with chosen feature_extractor
          kp_l, desc_l = feature_extractor.detectAndCompute(person_detected_before_gray, None)
          kp_r, desc_r = feature_extractor.detectAndCompute(cropped_frame_gray, None)

          matches = bf.knnMatch(desc_l, desc_r, k=2)  #compute matches between person detected and reference image 

          # Apply a filter to select the best matches only
          good_matches = []
          for m,n in matches:
            if m.distance < 0.7*n.distance:
                good_matches.append([m])

          #if the person detected has more than 15 matches in common with the reference person 
          #then store the frame, the number of matches and the bounding box
          if len(good_matches) > 15:
            list_person.append([left, top, right, bottom, confidence])
            nbrMatches.append(len(good_matches))
            frame_cropped.append(cropped_frame)

    if doMatches == 1 and init == 1:

        #If only one person is detected, its bounding box become the reference frame 
        if len(list_person) == 1:
          person_detected_before = frame_cropped[0].copy()

        #If several persons had more than 15 matches in common with reference frame,
        #diplay green bounding box around the person that have the more matches 
        if len(nbrMatches) > 0:
          color = [0,200,0]
          good_person = np.argmax(nbrMatches)
          person = list_person[good_person]    
        else:
        	person = [0,0,0,0,0]
    return person
          

