import cv2
import os
import numpy as np
from settings import WIDTH, HEIGHT, FOLDER_NAME

def read_images(foldername=FOLDER_NAME):
    image_list = []
    image_data = []
    for file in os.listdir("./{}".format(foldername)):
       if file.endswith(('.jpg','.jpeg','.png')):
           image_list.append(file)

    for i in range(0,len(image_list)):
        image = cv2.imread(os.getcwd() + "\\{}\\{}".format(foldername, image_list[i]))
        image = cv2.resize(image, (WIDTH, HEIGHT))
        image_data.append(image)

    return np.array(image_data)