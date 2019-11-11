# importing libraries 
import cv2 
import numpy as np 
import os
import time

video_path = '/Volumes/Data_HD/Users/Carles/codes/malnascudes/'
video_name = '8.Logo2.mov'
save_path = '/Volumes/Data_HD/Users/Carles/codes/malnascudes/last_frame'


number_of_frames_in_folder = len([f for f in os.listdir(save_path) if f.endswith('.png')])
print(str(number_of_frames_in_folder) +  ' frames in folder')
save_name = 'last_frame_' + str(number_of_frames_in_folder) + '.png'



# Create a VideoCapture object and read from input file 
video = cv2.VideoCapture(os.path.join(video_path,video_name)) 
fps = video.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
number_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

frames_done = 0  


video.set(cv2.CAP_PROP_POS_FRAMES, number_frames-10)
print('Frame number:', int(number_frames))
print('Position:', int(video.get(cv2.CAP_PROP_POS_FRAMES)))
_, frame = video.read()
cv2.imwrite(os.path.join(save_path,save_name),np.uint8(frame))

'''
# Read until video is completed 
while(video.isOpened()): 
      
  # Capture frame-by-frame 
  ret, frame = video.read() 
  if frames_done == number_frames-1: 

    # Display the resulting frame 
    cv2.imwrite(save_path,frame)
    
  frames_done+=1
  
'''
# When everything done, release  
# the video capture object 
video.release() 
   
