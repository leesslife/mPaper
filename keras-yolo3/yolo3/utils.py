"""Miscellaneous utility functions."""

from functools import reduce
#reduce(function,sequential[,inital])
#from functools import reduce 
#reduce(lambda x,y:x+y,[1,2,3])
#reduce(lambda x,y:x+y,[1,2,3],9)
#reduce(lambda x,y:x+y,[1,2,3],7)
#意思就是对sequence连续使用function, 如果不给出initial, 
#则第一次调用传递sequence的两个元素, 以后把前一次调用的结果和sequence的下一个元素传递给function. 
#如果给出initial, 则第一次传递initial和sequence的第一个元素给function.
#test_git
from PIL import Image
#PIL是图像处理库 详情请见练习
import numpy as np
#numpy 是矩阵运算库 详情请见练习
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
#是python绘图库与numpy一起使用才行
def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
        #函数拼接过程中存在的问题reduce(lambda f,g:function(lamdba),funcs)
        #*a代表元组传入，**kw代表字典，字典不定长通常放在后面
        #funcs=[conv1,conv2,conv3] 会成为conv3(conv2(conv1))x,越后面的越外层运算在后面，因此运算顺序和定义顺序是一样的
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    #改变图片的大小，保持图片的纵横比不变，使用padding(这里表示图片的纵横比保持不变，是的外边框正好能够展示图片，多于部分用0来填充)
    iw, ih = image.size                 #图片的真实大小
    w, h = size                         #所需尺寸呢的大小
    scale = min(w/iw, h/ih)             #尺度比值，取小的值，以iw，ih中大的为主
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)      #改变图片的大小，BICUBIC时一种差值算法具体
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))    #居中图像在画布中间
    return new_image

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a                #产生随机数标准差b-a 均值a

def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation''' 
    #随机数预处理
    line = annotation_line.split()    #对注释行进行分割，分割符号为空格
    image = Image.open(line[0])       #打开对应的路径的图片
    iw, ih = image.size               #图片大小
    h, w = input_shape                #输入类型，对应应该时416
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
    #对注释中line[1:]的每一行用逗号进行分割，生成一个数组，最终形成一个二位数组 分割结果应该时对应的x1,y1,x2,y2,o
    if not random:                    #如果random时false进行如下操作
        # resize image
        scale = min(w/iw, h/ih)       #
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))                      #让图片整体展示，并且居中 以不改变像素 来padding
            image_data = np.array(new_image)/255.                 #255归一化

        # correct boxes
        box_data = np.zeros((max_boxes,5))                        #生成box使用的(20,5)的box数据x1,y1,x2,y2,o
        if len(box)>0:                                            #如果图片存在对象box则进行如下操作                                
            np.random.shuffle(box)                                #对box(x1,y1,x2,y2,o)进行随机排序
            if len(box)>max_boxes: box = box[:max_boxes]          #如果box数量超过20个，则取前20个就可以了
            box[:, [0,2]] = box[:, [0,2]]*scale + dx              #为了使得图片中的box符合 input_box，缩放和移动位置
            box[:, [1,3]] = box[:, [1,3]]*scale + dy              #为了使得图片中的box符合 input_box, 缩放和移动位置
            box_data[:len(box)] = box                             #将box赋值给box_data

        return image_data, box_data                               #放回image数据和box数据(boxesNUm,5)box_data(max_boxes,5)但剩下的都是0

    #根据以上关系，如果random==false 则直接把图片塞进input_shape即可，如果random==True 则需要进行以下变更，
    # 1）图片随机大小，翻转，位置 2）HSV级别上的随机变更 3）根据图片的随机大小，翻转，位置纠正box
    # resize image
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter) #rand(i,j) 返回i到j之间的一个数 属于均匀分别，默认时0-1之间
    scale = rand(.25, 2)                                           
    if new_ar < 1:                                               
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)                   #图片大小改变，图片w/h比有可能改变，图片一条边变大

    # place image
    dx = int(rand(0, w-nw))                                        #如果图片变大，有可能为负定位，超出画布部分去除
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand()<.5                                            #翻转产生一个随机数(0-1),看看这个数是否大于.5，如何大于则左右翻转
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)                                       #如片HSV 中的色调(变更随机数)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)         #饱和度s(变更随机数)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)         #明度V(变更随机数)

    x = rgb_to_hsv(np.array(image)/255.)                        #把图片从rgb转化成HSV模型，但转化之前进行255的归一化
    x[..., 0] += hue                                            #色调增加
    x[..., 0][x[..., 0]>1] -= 1                                 #把x[...,0]大于1的数-1
    x[..., 0][x[..., 0]<0] += 1                                 #把x[...,0]小于0的数+1
    x[..., 1] *= sat                                            #饱和度改变
    x[..., 2] *= val                                            #明度改变
    x[x>1] = 1                                                  #大于1的数等于1
    x[x<0] = 0                                                  #小于0的数等于0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1            #转换回rgb格式
    #对图像数据进行变更

    # correct boxes
    # 根据图片的随机大小与翻转对box进行纠正
    box_data = np.zeros((max_boxes,5))                          #
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx                #根据大小与位置变更x1，x2
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy                #根据大小与位置变更y1, y2
        if flip: box[:, [0,2]] = w - box[:, [2,0]]              #如果翻转x1-x2 y1->y2翻转
        box[:, 0:2][box[:, 0:2]<0] = 0                          #最终如果x1，y1小于0则归0
        box[:, 2][box[:, 2]>w] = w                              #如果x2，y2大于w ，h 则归于w ，h
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]                           #计算处理后box_w
        box_h = box[:, 3] - box[:, 1]                           #计算处理后的box_h
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box  
        #以上执行逻辑与，如果box_w与box_h都大于1则返回1 box对应的对象box有效，否则无效
        if len(box)>max_boxes: box = box[:max_boxes]            #排除大于20的box
        box_data[:len(box)] = box                               #box赋值给box[:len(box)]多余的还是为0

    return image_data, box_data
