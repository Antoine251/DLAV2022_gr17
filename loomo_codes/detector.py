import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import os
from PIL import Image

#New imports
import cv2
import time
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# from darknet import *
from darknet.darknet import *
# load in our YOLOv4 architecture network
network, class_names, class_colors = load_network("/home/group17/DLAV-2022/projects/loomo/darknet/cfg/yolov4-csp.cfg", 
                                                    "/home/group17/DLAV-2022/projects/loomo/darknet/cfg/coco.data", 
                                                    "/home/group17/DLAV-2022/projects/loomo/darknet/yolov4-csp.weights")
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
debug = False
lossTimerStart = 0

def Detector(image):
  # convert image to a numpy array
  image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
  if debug: cv2.imshow('Test window', image)

  return imageProcess(image)

#Function performing person detection and tracking
def imageProcess(frame):
    global width, height, doMatches, bf, person_detected_before, init, lossTimerStart
    if debug: print("doMatches: ", doMatches, "init: ", init)
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
            if debug: print('Person Detected!')
            print('Person Detected!')
          
        else:           #-------TRACKING--------
          init = 1 #initialization was done
          person_detected_before_gray = cv2.cvtColor(person_detected_before, cv2.COLOR_RGB2GRAY)
          cropped_frame_gray =  cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2GRAY)

          #feature_extractor = cv2.xfeatures2d.SIFT_create() 
          feature_extractor = cv2.SIFT_create()

          # find the keypoints and descriptors with chosen feature_extractor
          kp_l, desc_l = feature_extractor.detectAndCompute(person_detected_before_gray, None)
          kp_r, desc_r = feature_extractor.detectAndCompute(cropped_frame_gray, None)
          
          if desc_r is not None and desc_l is not None:
            #print(len(desc_r),len(desc_r))
            matches = bf.knnMatch(desc_l, desc_r, k=2)  #compute matches between person detected and reference image 
            
            if np.shape(matches)[1] == 2:
              good_matches = []
              for m,n in matches:
                if m.distance < 0.9*n.distance:
                    good_matches.append([m])
            else:
              good_matches = []
          else:
            good_matches = []
          # Apply a filter to select the best matches only
          

          #if the person detected has more than 15 matches in common with the reference person 
          #then store the frame, the number of matches and the bounding box
          if len(good_matches) > 5:
            print("good matches ", len(good_matches))
            list_person.append([left, top, right, bottom, confidence])
            nbrMatches.append(len(good_matches))
            frame_cropped.append(cropped_frame)

    if doMatches == 1 and init == 1:

        #If only one person is detected, its bounding box become the reference frame 
        if len(list_person) == 1:
          person_detected_before = frame_cropped[0].copy()
          print(np.shape(person_detected_before))

        #If several persons had more than 15 matches in common with reference frame,
        #diplay green bounding box around the person that have the more matches 
        if len(nbrMatches) > 0:
          color = [0,200,0]
          good_person = np.argmax(nbrMatches)
          bbox_coord = list_person[good_person]
          bbox_label = 1  #'Person' 
          lossTimerStart = 0
        else:   
            bbox_coord = [0,0,0,0,0]
            bbox_label = 0  #'No person'
            if lossTimerStart == 0:
              lossTimerStart = time.time()

            print(time.time() - lossTimerStart)
            print(lossTimerStart)
            if time.time() - lossTimerStart > 5:
              print("Person lost !!  Initializing restart")
              doMatches = 0
              init = 0
              lossTimerStart = 0
              person_detected_before = None

        
        
    else:
        bbox_coord = [0,0,0,0,0]
        bbox_label = 0  #'No person'
    

    

    center_x = (bbox_coord[0] + bbox_coord[2])/2
    center_y = (bbox_coord[1] + bbox_coord[3])/2
    width_ = bbox_coord[2] - bbox_coord[0]
    height_ = bbox_coord[3] - bbox_coord[1]
    return [center_x, center_y, width_, height_], bbox_label