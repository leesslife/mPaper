#import sys
import os,shutil
import xml.etree.ElementTree as ET
#xmlPath2="C:\\lees_project\\mEPaper_project\\VOCdevkit\\mVOC2019\\Annotations_bk"
#xmlToPath2="C:\\lees_project\\mEPaper_project\\VOCdevkit\\mVOC2019\\Annotations"
#ImagePath2="C:\\lees_project\\mEPaper_project\\VOCdevkit\\mVOC2019\\JPEGImages_bk"
#ImageToPath2="C:\\lees_project\\mEPaper_project\\VOCdevkit\\mVOC2019\\JPEGImages"
xmlPath1="C:\\lees_project\\fire-dataset-dunnings\\fire_result_train"
xmlToPath1="C:\\lees_project\\mEPaper_project\\VOCdevkit\\mVOC2019\\Annotations"
ImagePath1="C:\\lees_project\\fire-dataset-dunnings\\images-224x224\\train\\fire"
ImageToPath1="C:\\lees_project\\mEPaper_project\\VOCdevkit\\mVOC2019\\JPEGImages"
def MoveVoc(xmlPath1,xmlToPath1,ImagePath1,ImageToPath1):

    i=0    
    xmlStr="从"+xmlPath1+"读取xml文件并写入"+xmlToPath1
    print(xmlStr)
    ImageStr="从"+ImagePath1+"读取图片文件并写入"+ImageToPath1
    print(ImageStr)
    for filename in os.listdir(xmlPath1):
        i+=1
        subImageName=filename.split('.')[0]
        ext=filename.split('.')[1]
        ImageName=subImageName+'.png'
        filename_dir=xmlPath1+'\\'+filename
        Imagename_dir=ImagePath1+'\\'+ImageName
        if ext=="xml":
            tree=ET.parse(filename_dir)
            root=tree.getroot()
            obj=root.find("object")
            if obj is None:
                print("本图片内没有对象，可以跳过")
            else:
                shutil.copy(filename_dir,xmlToPath1+'\\fire'+str(i)+'.xml')
                shutil.copy(Imagename_dir,ImageToPath1+'\\fire'+str(i)+'.png')
        #print(ImageName)


if __name__=='__main__':
    MoveVoc(xmlPath1,xmlToPath1,ImagePath1,ImageToPath1)
