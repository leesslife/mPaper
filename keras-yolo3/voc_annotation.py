import xml.etree.ElementTree as ET
from os import getcwd
#这段代码解析voc代码
sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert_annotation(year, image_id, list_file):          #转化voc数据
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))  #根据固定格式的路径打开对应的文件
    tree=ET.parse(in_file)                                                 #利用xml.etree.ElementTree模块打开xml文件
    root = tree.getroot()
    #获取xml的根路径

    for obj in root.iter('object'):
        #在根路径下解析object对象
        difficult = obj.find('difficult').text
        #在object对象下找到difficult对象的text内容
        cls = obj.find('name').text
        #在object对象下找到name对象的text内容
        if cls not in classes or int(difficult)==1:
            continue
        #如果cls不是在类名中 或者difficult==1那么跳过本次循环，difficult是一个忽略参数
        cls_id = classes.index(cls)
        #获取类名在classes列表中的位置
        xmlbox = obj.find('bndbox')
        #在object对象中找到‘bndbox’对象
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        #根据bndbox对象生成(xmin,ymin,xmax,ymax)对象
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        #以固定格式写入list_file xmin ymin xmax ymax 以及cls_id

wd = getcwd()
#返回当前的工作目录

for year, image_set in sets:
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    #以固定格式打开txt对象，此txt文档对象里记录着图片的编号id，获取到后去除头尾的空格和回车，然后再用空格和回车分割字符串，形成数组
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    #以只写并创建的方式打开文件
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
        #将%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg格式的字符串写入到文件中
        convert_annotation(year, image_id, list_file)
        #在同一行中继续写入此图片的 xmin ymin xmax ymax 以及cls_id
        list_file.write('\n')
        #在末尾加入换行符
    list_file.close()

