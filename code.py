import cv2
import numpy as np
import  os

path = '/home/u124223/Desktop/images'
final_res = (100,200)
final_image_center = tuple(np.asarray(final_res)/2)

image_scale = 1/5 #scale factor that tells how big each image will look compared to the final image 

final_image = np.zeros(final_res)

for i in os.listdir(path):
    
    #read corresponding image
    img = cv2.imread(os.path.join(path,i))
    
    if type(img) == 'numpy.ndarray':
    
        og_width_height_ratio = img.shape[0] / img.shape[1]
        print(i)
        print(type(img))
        
        
        # resize to fit it into final image
        new_width = int(final_res[0]*image_scale)
        new_height = int(new_width/og_width_height_ratio)
        resized_img = cv2.resize(img, (new_width,new_height), interpolation = cv2.INTER_AREA)
        
        #compute rotation and translation matrix
        rotation_matrix = cv2.getRotationMatrix2D(final_image_center, 0, image_scale)
        
        #apply transofrmations and show image
        final_image = cv2.warpAffine(resized_img, rotation_matrix, final_res)
        
cv2.imshow('Rotation', final_image)    
    
cv2.waitKey()
