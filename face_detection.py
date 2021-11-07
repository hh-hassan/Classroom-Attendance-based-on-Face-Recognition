import cv2
import sys
import numpy as np
import sys
import os
import dlib
import time
from skimage import io
from skimage.viewer import ImageViewer

# video object
video = cv2.VideoCapture(0)

# Enter the directory in which you want to store the files (end the location with a '/')
# directory = '<Directory Path>'

# My directory
directory = 'E:/'

flag = int(input('Enter the purpose:\n1 - Storing your image\n2 - Checking for the attendance\n'))

# create a directory, if it doesnot exits

if flag == 1:
    
    sub_code = input('Enter your subject code:')
    roll = input('Enter your roll number:')
    d_path = directory + sub_code + '/' + roll
    
    if not os.path.exists(d_path):
        os.makedirs(d_path)


# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()

# count for total number of images to be saved
cnt = 0

# Generate the images
while True:

    # turn the webcam on and capture the first frame
    check, frame = video.read()

    if check:
        
        # convert the frames into gray scale images
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        win = dlib.image_window()

        # open a window showing the image
        win.set_image(gray)

        # Run the HOG face detector on the image data.
        # The result will be the bounding boxes of the faces in our image.
        detected_faces = face_detector(gray, 0)


        for i, face_rect in enumerate(detected_faces):

            # Draw boxes around the faces in the image
            win.add_overlay(face_rect)
            
            cnt = cnt + 1
            
            img_box = gray[face_rect.top() : face_rect.bottom(), face_rect.left(): face_rect.right()]
                
            # Show the captured face of the image in a new window
            cv2.imshow('Image', frame)

            # Saving the images
            
            if flag == 1:
                cv2.imwrite(directory + sub_code + '/{}/images{}.jpg'.format(roll,cnt), img_box)

            if flag == 2:
                cv2.imwrite(directory + 'test/images{}.jpg'.format(cnt), img_box) 


        #dlib.hit_enter_to_continue()       

        key = cv2.waitKey(5)
            
        # stop when user enters 'q' or when 100 images are taken
        
        if key == ord('q'):
           break
        
        if flag == 1:
            if cnt >= 100:
                break
    
#close the video object (it closes the webcam)      
video.release()
cv2.destroyAllWindows()

## run face_detection.py