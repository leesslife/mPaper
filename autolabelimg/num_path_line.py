import cv2
import os
def read_directory(directory_name):
    num_line=0
    for filename in os.listdir(directory_name):
        print(filename)
        img=cv2.imread(directory_name+"/"+filename)
        cv2.imshow("filename",img)
        num_line+=1
        print(num_line)
        if(cv2.waitKey(30)&0xFF==ord('q')):
            break
            
    print(num_line)

read_directory("C:\\lees_project\\fire-dataset-dunnings\\images-224x224\\train\\fire")
