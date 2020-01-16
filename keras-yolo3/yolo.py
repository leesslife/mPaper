# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""
#这里的代码负责侦测

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

class YOLO(object):  #船舰yolo类，这里object负责为传入参数
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }
    #score负责在对象识别计算时，类别得分C x 对象得分O获取到响应的类别置信度，在分类为20类的情况下，20x3x13x13+20x3x26x26+20x3x52x52个得分中，将得分小于0.3的归0，并且所有参数都归0
    #在非极大化运算中大iou值的直接归0
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
    #此函数用于获取YOLO类自带的默认值

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        #def __init__(self,_obj):
        #    self.name = _obj['name']
        #    self.age = _obj['age']
        #    self.energy = _obj['energy']
        #    self.gender = _obj['gender']
        #    self.email = _obj['email']
        #    self.phone = _obj['phone']
        #    self.country = _obj['country']
        #利用下面的方法可以大大节省代码量
        #class Person:
        #    def __init__(self,_obj):
        #    self.__dict__.update(_obj)
        self.__dict__.update(kwargs) # and update with user overrides
        #    表明使用者也可以自己写入配置参数
        self.class_names = self._get_class()
        #获取类名的一维列表
        self.anchors = self._get_anchors()
        #获取二维的anchor李彪
        self.sess = K.get_session()
        #获取新会话
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        #根据环境获取类文件路径
        with open(classes_path) as f:
            class_names = f.readlines()
        #根据行读取类文件路径 然后形成list
        class_names = [c.strip() for c in class_names]
        #去除首位空字符串与换行符返回list
        return class_names
        #最终返回带类名的列表

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        #获取anchors文件所在完整的路径
        with open(anchors_path) as f:
            anchors = f.readline()
        #这里只读取一行数据即可
        anchors = [float(x) for x in anchors.split(',')]
        #将所有数据用逗号分割 
        return np.array(anchors).reshape(-1, 2)
        #两个一组 形成二维数组

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        #获取权重模型文件路径
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        #如果模型文件路径的结尾不是.h5 则输出错误警告
        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        #anchors的长度
        num_classes = len(self.class_names)
        #类的数量
        is_tiny_version = num_anchors==6 # default setting
        #是否是tiny模型
        try:
            self.yolo_model = load_model(model_path, compile=False)
            #如果load_model加载model_path对应的模型文件不出错，说明模型文件是完整的(不是单单只有权重参数，也包含结构参数)
            #如果出错说明模型文件只有权重参数，所以必须先加载模型
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            #如果is_tiny_version存在 则调用tiny模型否则 调用完整模型
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
            #加载权重文件
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'
                #assert expression [, arguments]
                #等价于
                #if not expression:
                #   raise AssertionError(arguments)
                #如果不满足条件 则输出错误，并将arguments一并输出

        print('{} model, anchors, and classes loaded.'.format(model_path))
        #输出model文件路径

        # Generate colors for drawing bounding boxes.
        # 这里为了画边框而形成不同的颜色，写的太复杂有点故意装逼的气味
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        #HSV元组？[(?/len,1.,1.),(?/len,1.,1.),(?/len,1.,1.),...] 列表长度等于类数量
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        #对hsv_tuples 以元组作为列表 利用colorsys.hsv_to_rgb做rgb转化
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        #反归一化
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        #产生伪随机数种子
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        #打乱元素次序
        np.random.seed(None)  # Reset seed to default.
        #重设随机数种子
        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        # 产生输入图片的shape容器 如(416,416)
        if self.gpu_num>=2:            #如果gpu的数量大于2，则使用多GPU模型进行训练
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
            #启动多gpu模型
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        #返回boxes信息，scores信息以及classes信息(一副图片对应类别的对应对象有多少，这些对象的得分和boxes信息分别是什么)
        return boxes, scores, classes

    def detect_image(self, image):                                 #开始侦测图像
        start = timer()                                            #确定开始监测的时间
        
        if self.model_image_size != (None, None):                #如果image_size存在，则执行以下操作，通常model_image_size=416
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'          #需要32的倍数，否则输出错误提示
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'          #同上
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
            #letterbox_image
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

