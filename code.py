import cv2
import numpy as np
import  os
import time
import random

path = 'cremeta' #fodler name


image_interval = 1
time_step = 0.05
seconds = time.time()

image_scale = 1/5 #scale factor that tells how big each image will look compared to the final image 


final_res = (1080,720)
final_image = np.dstack((np.zeros((final_res[1],final_res[0])),np.ones((final_res[1],final_res[0])),np.zeros((final_res[1],final_res[0]))))

final_image = cv2.imread(os.path.join(os.getcwd(),'background1.png'))/255
final_res = (final_image.shape[1],final_image.shape[0])
print(final_res)


number_of_videos_in_folder = len([f for f in os.listdir(os.path.join(os.getcwd(),'outputs')) if f.endswith('.avi')])
print(str(number_of_videos_in_folder) +  ' videos in folder')
output_video_name = 'output_' + str(number_of_videos_in_folder) + '.avi'
output_video = cv2.VideoWriter(os.path.join(os.getcwd(),'outputs',output_video_name),cv2.VideoWriter_fourcc('M','J','P','G'), 25, final_res)




archius_carpeta = os.listdir(os.path.join(os.getcwd(),path))
fotos_carpeta = [i for i in archius_carpeta if i.endswith('.png') or i.endswith('.mov')]
random.shuffle(archius_carpeta)

numero_fotos = len(fotos_carpeta)
image_count = 0

video_frames_displayed = 50
start_video_at = 0.5 #start video at half of it
end_video_at = 1


while(True):
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break    
    if key == ord('+'):
        image_interval += time_step 
    if key == ord('-'):
        image_interval -= time_step     
    if archius_carpeta[image_count].endswith('.png'):
        
        img = cv2.imread(os.path.join(os.getcwd(),path,archius_carpeta[image_count]))
        #exit loop if 'q' pressed
        og_width_height_ratio = img.shape[0] / img.shape[1]
        
        # resize to fit it into final image
        new_width = int(final_res[0]*image_scale)
        new_height = int(new_width*og_width_height_ratio)
        


        resized_img = cv2.resize(img, (new_width,new_height), interpolation = cv2.INTER_AREA)
        random_angle = np.random.uniform(0,360)
        random_displacement = (np.random.uniform(0,final_res[0]-new_width),np.random.uniform(0,final_res[1]-new_height))
        #random_angle = 90
        #random_displacement = (final_res[0]/2-new_height/2,final_res[1]/2-new_width/2)
        #compute rotation and translation matrix
        
        #rot_matrix= np.float32([[np.cos(random_angle), np.sin(random_angle), 0], [-np.sin(random_angle), np.cos(random_angle), 0]])
        rotation_matrix = cv2.getRotationMatrix2D((0,0), random_angle, 1)
        translation_matrix = np.float32([[1, 0, random_displacement[0]], [0, 1, random_displacement[1]]]) 

        final_matrix = translation_matrix

        #apply transofrmations and show image
        prov_image = cv2.warpAffine(resized_img,final_matrix, final_res)
        #final_image += prov_image/255
        #show_image = cv2.normalize(final_image, 0, 255)
        #final_image = show_image
        
        prov_image_position = np.where(prov_image!=0)
        final_image[prov_image_position[0],prov_image_position[1],:] = prov_image[prov_image_position[0],prov_image_position[1],:]/255
        cv2.imshow('Rotation', final_image)
        output_video.write(np.uint8(final_image*255))
        
        while time.time() - seconds < image_interval:
            #print(time.time() - seconds)
            output_video.write(np.uint8(final_image*255))
        seconds = time.time()
    elif archius_carpeta[image_count].endswith('.mov'):
                    
        video = cv2.VideoCapture(os.path.join(os.getcwd(),path,archius_carpeta[image_count]))
        fps = video.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
        number_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = number_frames/fps
        _, first_frame = video.read()
        og_width_height_ratio = first_frame.shape[0] / first_frame.shape[1]
        new_width = int(final_res[0]*image_scale)
        new_height = int(new_width*og_width_height_ratio)

        random_angle = np.random.uniform(0,360)
        random_displacement = (np.random.uniform(0,final_res[0]-new_width),np.random.uniform(0,final_res[1]-new_height))
        #random_angle = 90
        #random_displacement = (final_res[0]/2-new_height/2,final_res[1]/2-new_width/2)
        #compute rotation and translation matrix
        
        #rot_matrix= np.float32([[np.cos(random_angle), np.sin(random_angle), 0], [-np.sin(random_angle), np.cos(random_angle), 0]])
        rotation_matrix = cv2.getRotationMatrix2D((0,0), random_angle, 1)
        translation_matrix = np.float32([[1, 0, random_displacement[0]], [0, 1, random_displacement[1]]])

        final_matrix = translation_matrix
        print(archius_carpeta[image_count])
        frame_count = 0
        while(video.isOpened()):
            # Capture frame-by-frame
            ret, frame = video.read()
            if ret == True:
                if frame_count>int(start_video_at*number_frames) and frame_count<start_video_at*number_frames + video_frames_displayed:
                    _, frame = video.read()
                    # resize to fit it into final image
                    resized_frame = cv2.resize(frame, (new_width,new_height), interpolation = cv2.INTER_AREA)

                    #apply transofrmations and show image
                    prov_frame = cv2.warpAffine(resized_frame,final_matrix, final_res)
                    #final_image += prov_image/255
                    #show_image = cv2.normalize(final_image, 0, 255)
                    #final_image = show_image
                    
                    prov_frame_position = np.where(prov_frame!=0)
                    final_image[prov_frame_position[0],prov_frame_position[1],:] = prov_frame[prov_frame_position[0],prov_frame_position[1],:]/255

                    cv2.imshow('Rotation', final_image)
                    output_video.write(np.uint8(final_image*255))



                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                elif frame_count>start_video_at*number_frames + video_frames_displayed:
                    break
            # Break the loop
            frame_count+=1
        video.release()

    image_count+=1
    if image_count==numero_fotos:
        image_count = 0
        random.shuffle(archius_carpeta)

        
    

    
# When everything done, release the capture
output_video.release()
cv2.destroyAllWindows()
