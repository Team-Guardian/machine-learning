import cv2
import os
import numpy as np

import xml.etree.ElementTree as ET
from settings import WIDTH, HEIGHT, RAW_FOLDER_DATA, PROC_FOLDER_DATA


def process_data(raw_folder_name=RAW_FOLDER_DATA, proc_folder_name=PROC_FOLDER_DATA):
    image_list = []
    for file in os.listdir(raw_folder_name):
       if file.endswith(('.jpg','.jpeg','.png')):
           image_list.append(file)

    for i in range(0,len(image_list)):
        image = cv2.imread(os.path.join(os.getcwd(), raw_folder_name, image_list[i]))
        image = cv2.resize(image, (WIDTH, HEIGHT))
        cv2.imwrite(os.path.join(proc_folder_name, image_list[i]), image)
    return True


def read_data(proc_folder_name=PROC_FOLDER_DATA):
    image_list = []
    image_data = []
    image_metadata = []

    for file in os.listdir(proc_folder_name):
        if file.endswith(('.jpg','.jpeg','.png')):
            image_list.append(file)

    for i in range(0,len(image_list)):
        image = cv2.imread(os.path.join(os.getcwd(), proc_folder_name, image_list[i]))
        image_data.append(image)

    results_images = np.array(image_data)

    for file in image_list:
        filename = file.split('.')[0] + '.xml'
        path = os.path.join(os.getcwd(),proc_folder_name, filename)

        if os.path.exists(path) == False:
            raise Exception('Error, missing file {}, path is {}'.format(filename, path))

        tree = ET.parse(path)
        root = tree.getroot()


    return results_images