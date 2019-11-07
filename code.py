import cv2
import numpy as np
import  os
import time

path = 'images'
final_res = (500,500)
final_image_center = tuple(np.asarray(final_res)/2)

image_interval = 1
time_step = 0.05
seconds = time.time()

image_scale = 1/5 #scale factor that tells how big each image will look compared to the final image 

final_image = np.dstack((np.zeros((final_res[0],final_res[1])),np.ones((final_res[0],final_res[1])),np.zeros((final_res[0],final_res[1]))))


archius_carpeta = [cv2.imread(os.path.join(os.getcwd(),path,i)) for i in os.listdir(os.path.join(os.getcwd(),path))]
fotos_carpeta = [i for i in archius_carpeta if type(i) == type(final_image)]
numero_fotos = len(fotos_carpeta)
image_count = 0



while(True):
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break    
    if key == ord('+'):
        image_interval += time_step 
    if key == ord('-'):
        image_interval -= time_step 
    print(image_interval)
    if time.time() - seconds > image_interval:
        img = fotos_carpeta[image_count]
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
        
        '''
        final_y_pos = np.asarray(prov_image_position[0]+random_displacement[0],int)
        final_x_pos = np.asarray(prov_image_position[1]+random_displacement[1],int)
        for c in range(len(final_y_pos)):
            if final_x_pos[c]< final_res[1] and final_y_pos[c]< final_res[0]:
                final_image[final_y_pos[c], final_x_pos[c], :] = prov_image[prov_image_position[0][c],prov_image_position[1][c],:]/255
        #final_image[final_y_pos,final_x_pos,:] = prov_image[np.where(prov_image!=0)]/255
        '''

        cv2.imshow('Rotation', final_image)  

        image_count+=1
        if image_count==numero_fotos:
            image_count = 0

        seconds = time.time()

        
    
# When everything done, release the capture
cv2.destroyAllWindows()

'''
def aply_affine_transf(image, destiny, rotation_angle, dispalcement_tuple,scale):
    matrix= np.float32([[np.cos(rotation_angle), np.sin(rotation_angle), dispalcement_tuple[0]], [np.-sin(rotation_angle), np.cos(rotation_angle), dispalcement_tuple[1]]])
    im_index_x = range(image.shape[1])
    im_index_y = range(image.shape[0])

    for x in im_index_x:
        for y in im_index_y:

'''