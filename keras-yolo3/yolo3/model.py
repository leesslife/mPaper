"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K 
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization 
from keras.models import Model
from keras.regularizers import l2

from yolo3.utils import compose


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)} #内核正则化 L2正则化参数（5e-4）
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same' #设定填充方案如果 ‘Same’的填充方案在于 滑块为了确保所有的数据都被覆盖而在周边填充0，‘Valid’则不会填充，它会在数据列或者行不够是停止滑动。
    darknet_conv_kwargs.update(kwargs) #更行字典在末尾添加kwargs
    return Conv2D(*args, **darknet_conv_kwargs)  
    #keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
    #Conv2D用于卷积计算

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}  #在卷积运算时禁止偏移量b（b=0）
    no_bias_kwargs.update(kwargs) #更新字典 在末尾添加kwargs
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1)) #组合各个成 成为一个层 卷积成后面添加BatchNormalization和LeakyReLU激励函数

def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)   #进行0填充，在行维度上前加一行 后面不加，在列维度上 前加一行 后不加
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x) #进行卷积计算包括L2正则化，BN计算以及LeakReLu激励函数
    for i in range(num_blocks): #这里进行残差计算，首先进行1x1，stride=1的卷积计算，然后  滤子数量增加1倍  进行3x3的计算stride=1 的卷积计算
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])         #这里进行残差计算的最后一步，将x与y进行元素相加
    return x
    #这里注明残差模块如下：
    #3x3 stride=（2，2）fliters=N
    #1x1 stride=（1，1）fliters=N//2
    #3x3 stride=（1，1）fliters=N
    #最后将1，3进行元素相加
    
def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)  #输入为416x416
    x = resblock_body(x, 64, 1)               #第一个残差块 残差部分为1 滤子数量为64
    x = resblock_body(x, 128, 2)              #第二个残差块 残差部分为2 滤子数量为128
    x = resblock_body(x, 256, 8)              #第三个残差块 残差部分为8 滤子数量为256
    x = resblock_body(x, 512, 8)              #第四个残差块 残差部分为8 滤子数量为512
    x = resblock_body(x, 1024, 4)             #第五个残差块 残差部分为4 滤子数量为1024
    return x

def make_last_layers(x, num_filters, out_filters):      #这个部分形成最后卷积层
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(                                        
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),            #卷积层1x1 stride=1 滤子数量为N
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),          #卷积层3x3 stride=1 滤子数量为2N
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),            #卷积层1x1 stride=1 滤子数量为N
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),          #卷积层3x3 stride=1 滤子数量为2N
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)         #卷积层1x1 stride=1 滤子数量为N
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),          #卷积层3x3 stride=1 滤子数量为2N
            DarknetConv2D(out_filters, (1,1)))(x)                  #卷积层1x1 stride=1 滤子数量为N
    return x, y                                                    #保持输出点


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    #此地涉及两种Keras两种模型Sequential()模型为顺序模型，没有分支。Model为函数模型提供函数API来实现
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))
    #此处使用make_last_layers()残生两个分支输出 y1为预测点1
    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)), 
            UpSampling2D(2))(x)                                
    x = Concatenate()([x,darknet.layers[152].output])  #与add_19进行代码拼接 
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))    
    #此处生成两个分支，x1参与接下去的卷积计算 y2为预测点2

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),  #同上
            UpSampling2D(2))(x)                  #同上
    x = Concatenate()([x,darknet.layers[92].output])  #同上
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5)) #同上

    return Model(inputs, [y1,y2,y3])   #放Model模型，输入还是那个输入，输出给位y1，y2，y3/y1,y2,y3

def tiny_yolo_body(inputs, num_anchors, num_classes):  #此处定义tiny_yolo_body模型
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),          #卷积3x3 stride=1 滤子n=16
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),  #MaxPooling stride=2
            DarknetConv2D_BN_Leaky(32, (3,3)),          #卷积3x3 stride=1 滤子n=32
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),  #同上
            DarknetConv2D_BN_Leaky(64, (3,3)),          #卷积3x3 stride=1 滤子n=64   
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),  #同上
            DarknetConv2D_BN_Leaky(128, (3,3)),         #卷积3x3 stride=1 滤子n=128
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'), #卷积2x2 stride=2 
            DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)  #卷积3x3 stride=1 滤子n=256
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'), #MaxPooling "Same"填充
            DarknetConv2D_BN_Leaky(512, (3,3)),                           #卷积3x3 stride=1 
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'), #MaxPooling ”Same“填充
            DarknetConv2D_BN_Leaky(1024, (3,3)),                          #卷积成滤子n=1024 其他参考上方
            DarknetConv2D_BN_Leaky(256, (1,1)))(x1)                       #卷积1x1 stride=1 其他参考上方
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),                           #3x3 卷积 stride=1 滤子n=512
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)        #1X1 卷积 stride=1 滤子n=3x25=75

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),                           #1x1 卷积 stride=1 滤子n=128
            UpSampling2D(2))(x2)                                          #进行一次上采样
    y2 = compose(
            Concatenate(),                                                #将x1与x2进行拼接
            DarknetConv2D_BN_Leaky(256, (3,3)),                           #3x3卷积 滤子n=256
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])   #1x1卷积 滤子n=125

    return Model(inputs, [y1,y2])                                         #确定模型输入与输出


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)                                            #确定anchors 数量，标准为9个 mini——yolov3为6个
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])         #再定义anchors 矩阵类型 根据 num_anchors,在

    grid_shape = K.shape(feats)[1:3] # height, width                      切片打开特征的长宽，yolov3的三种尺度 分别时8x8 16x16以及32x32
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),         #
        [1, grid_shape[1], 1, 1])
        #先生成[0,1,2,3,4,5,6，7]
        #然后再本数列后面再添加三个维度
        #数列变成[[[[0]]],[[[1]]],[[[2]]],[[[3]]],[[[4]]],[[[5]]],[[[6]]],[[[7]]]]
        #在数列中的第二个维度进行扩展
        #变成数列如下[[[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]]],
        #            [[[1]],[[1]],[[1]],[[1]],[[1]],[[1]],[[1]],[[1]]],
        #            [[[2]],[[2]],[[2]],[[2]],[[2]],[[2]],[[2]],[[2]]],
        #            [[[3]],[[3]],[[3]],[[3]],[[3]],[[3]],[[3]],[[3]]],
        #            [[[4]],[[4]],[[4]],[[4]],[[4]],[[4]],[[4]],[[4]]],
        #            [[[5]],[[5]],[[5]],[[5]],[[5]],[[5]],[[5]],[[5]]],
        #            [[[6]],[[6]],[[6]],[[6]],[[6]],[[6]],[[6]],[[6]]],
        #            [[[7]],[[7]],[[7]],[[7]],[[7]],[[7]],[[7]],[[7]]]]
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
        #同上最终结果如下(前外后里 里后先分)
        #[0],[1],[2],[3],[4],[5],[6],[7]
        #[[0]],[[1]],[[2]],[[3]],[[4]],[[5]],[[6]],[[7]]
        #[[[[0]],[[1]],[[2]],[[3]],[[4]],[[5]],[[6]],[[7]]]]


        #[[[[0]],[[1]],[[2]],[[3]],[[4]],[[5]],[[6]],[[7]]]，
        # [[[0]],[[1]],[[2]],[[3]],[[4]],[[5]],[[6]],[[7]]]，
        # [[[0]],[[1]],[[2]],[[3]],[[4]],[[5]],[[6]],[[7]]]，
        # [[[0]],[[1]],[[2]],[[3]],[[4]],[[5]],[[6]],[[7]]]，
        # [[[0]],[[1]],[[2]],[[3]],[[4]],[[5]],[[6]],[[7]]]，
        # [[[0]],[[1]],[[2]],[[3]],[[4]],[[5]],[[6]],[[7]]]，
        # [[[0]],[[1]],[[2]],[[3]],[[4]],[[5]],[[6]],[[7]]]，
        # [[[0]],[[1]],[[2]],[[3]],[[4]],[[5]],[[6]],[[7]]]]
    grid = K.concatenate([grid_x, grid_y])
        #在第一个维度上进行拼接
        #[[[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]]],
        #[[[1]],[[1]],[[1]],[[1]],[[1]],[[1]],[[1]],[[1]]],
        #[[[2]],[[2]],[[2]],[[2]],[[2]],[[2]],[[2]],[[2]]],
        #[[[3]],[[3]],[[3]],[[3]],[[3]],[[3]],[[3]],[[3]]],
        #[[[4]],[[4]],[[4]],[[4]],[[4]],[[4]],[[4]],[[4]]],
        #[[[5]],[[5]],[[5]],[[5]],[[5]],[[5]],[[5]],[[5]]],
        #[[[6]],[[6]],[[6]],[[6]],[[6]],[[6]],[[6]],[[6]]],
        #[[[7]],[[7]],[[7]],[[7]],[[7]],[[7]],[[7]],[[7]]],
        # [[[0]],[[1]],[[2]],[[3]],[[4]],[[5]],[[6]],[[7]]]，
        # [[[0]],[[1]],[[2]],[[3]],[[4]],[[5]],[[6]],[[7]]]，
        # [[[0]],[[1]],[[2]],[[3]],[[4]],[[5]],[[6]],[[7]]]，
        # [[[0]],[[1]],[[2]],[[3]],[[4]],[[5]],[[6]],[[7]]]，
        # [[[0]],[[1]],[[2]],[[3]],[[4]],[[5]],[[6]],[[7]]]，
        # [[[0]],[[1]],[[2]],[[3]],[[4]],[[5]],[[6]],[[7]]]，
        # [[[0]],[[1]],[[2]],[[3]],[[4]],[[5]],[[6]],[[7]]]，
        # [[[0]],[[1]],[[2]],[[3]],[[4]],[[5]],[[6]],[[7]]]]
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))               #实现损失函数中的XY计算
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))         #实现损失函数中的wh计算 
    box_confidence = K.sigmoid(feats[..., 4:5])                                                          #实现损失函数中的置信度计算
    box_class_probs = K.sigmoid(feats[..., 5:])                                                          #实现损失中的类别信息计算
    #以上运算都经过了归一化操作(针对于整个图像)

    if calc_loss == True:                          
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]                                   #倒置xy
    box_hw = box_wh[..., ::-1]                                   #倒置wh                   
    input_shape = K.cast(input_shape, K.dtype(box_yx))           #重新定义数据类型
    image_shape = K.cast(image_shape, K.dtype(box_yx))           #重新定义数据类型
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))     
    #计算input类型与输入图像类型的比值，最后取最小化，然后乘以图像类型，然后round四舍五入取整，将image_put重定义为与input相似的类型
    # 这里理解可以举例）使得图像能够全部显示，但比例保持不变
    offset = (input_shape-new_shape)/2./input_shape               #计算偏差(由于图像可以刚刚好显示，大边比例偏差为0，小边比列除以2，可以想象图像在input_shape里面居中，小边离上下边的差值与input的比例)
    scale = input_shape/new_shape                                 #计算缩放比例，大边为1，小边自己想象
    box_yx = (box_yx - offset) * scale                            #根据偏差重新计算box_yx
    box_hw *= scale                                               #根据缩放比例重新计算box_hw

    box_mins = box_yx - (box_hw / 2.)                            #计算左上角坐标
    box_maxes = box_yx + (box_hw / 2.)                           #计算右下角坐标
    boxes =  K.concatenate([                                     #在最后一个维度上进行拼接，使得最后最后一个维度变成四维
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])       #乘以图像原来尺寸，使之变回来
    return boxes                                             #返回真实预测值


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):         #box得分计算
    '''Process Conv layer output'''                                  
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,                    
        anchors, num_classes, input_shape)
    #用于预测(box_xy,box_wh,box_confidence,box_class_probs,都是五维的tensor,其中(-1,1,1,1,1))
    #第一维度为batch 通常为1
    #第二维度为高度
    #第三维度为宽度
    #第四维度为num_anchor
    #第五维度为具体预测值
    #预测出的结果都是经过归一化(相对于整张图片来说)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape) #根据图片的实际大小，计算实际左上与右下坐标点
    boxes = K.reshape(boxes, [-1, 4])                                    #保持第四维度不变，重新排序（对于每一张图片而言）
    box_scores = box_confidence * box_class_probs                        #置信度x类别信息，得出类别概论
    box_scores = K.reshape(box_scores, [-1, num_classes])                #保持第四维度不变，重新排序（对于每一张图片而言）与坐标信息一直
    return boxes, box_scores                                             #返回排序好的坐标信息与维度信息


def yolo_eval(yolo_outputs,                                             
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)                                                      #具体计算评估预测值
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32                                   # 最小一层13x32=416，回归图片输入大小
    boxes = []                                                                         # 定义boxes_list
    box_scores = []                                                                    # 定义boxes_scores_list
    for l in range(num_layers):                                                        # 根据每一层定义类别信息
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],                   # 计算得分函数与概率值(排序好的)
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)            # 这里需要注意的是13x13对应的是最大的anchor box尺寸
        boxes.append(_boxes)                                                           
        box_scores.append(_box_scores)                                                 # 这里根据排序将boxes 坐标与概率得分各层合并
    boxes = K.concatenate(boxes, axis=0)                                               # 进行最外层合并，根据上面操作需要
    box_scores = K.concatenate(box_scores, axis=0)                                     # 进行最外层合并，根据上面操作需要 

    mask = box_scores >= score_threshold                                               # 过滤掉得分小于.6的点 其他都是1
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')                            # 重定义max_box的值的类型
    boxes_ = []                                                                        # 定义boxes_的list
    scores_ = []                                                                       # 定义list 
    classes_ = []                                                                      # 定义类别信息
    for c in range(num_classes):                                                       # 
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])               #过滤出针对每一类的box与box_scores
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold) #极大化计算，排序加iou计算 输出最终的index_num
        class_boxes = K.gather(class_boxes, nms_index)                                    #根据索引取出index box
        class_box_scores = K.gather(class_box_scores, nms_index)                          #根据索引取出index_box_scores
        classes = K.ones_like(class_box_scores, 'int32') * c                              #生成类别信息矩阵
        boxes_.append(class_boxes)                                                        #
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)                                            #合并各类信息

    return boxes_, scores_, classes_                                                      #返回


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format
    为了训练，对true boxes进行预处理
    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32  
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes, dtype='float32')      #重新定义true_boxes类型
    input_shape = np.array(input_shape, dtype='int32')      #重新定义input_shape类型
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2  #计算true_xy 实际值
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]         #计算true_wh 实际值
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]       #根据输入尺度进行归一化
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]       #根据输入尺度进行归一化
    #生成true_boxes x ,y,w,h,c 类型[m,t,5]

    m = true_boxes.shape[0]                                 #本batch里有多少张图片
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]      #根据不同的层取32，16，8的缩放比列，最小是416/32=13 之后有26与52
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),dtype='float32') for l in range(num_layers)]
    #先生成zeros矩阵，类型如下[batch数量,高度，宽度，anchor_num,预测值] 是个数组
    #这里预备好了y_true 是个数组 素组中的每个矩阵如上

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)                   #在低级维度上加上一维(也就是在最外层在套一层变成(1,N,2))
    anchor_maxes = anchors / 2.                            #wh/2 anchors
    anchor_mins = -anchor_maxes                            #-wh/2
    valid_mask = boxes_wh[..., 0]>0                        #去除不必要的0或负值根据w来实现
    #这里扩充一个维度为下面进一步的计算做准备 anchors扩充一维

    for b in range(m):                                    
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]                    #根据以上掩码去除0的行

        if len(wh)==0: continue                            #如何最终wh长度为0说明根本没有有效的对象，跳过本张图片
        # Expand dim to apply broadcasting.                #
        wh = np.expand_dims(wh, -2)                        #在最后第二个维度上加一层
        box_maxes = wh / 2.                                #true box wh/2
        box_mins = -box_maxes                              #true_box -wh/2
        #每个true_box对象的+/1wh/2 根据维度分成N分与N个Anchor进行计算
        #对象1
        #wh/2->Anchor
        #wh/2->Anchor
        #..................................
        #..................................
        #对象2
        #wh->Anchor
        #wh->Anchor
        #..................................
        #..................................
        

        intersect_mins = np.maximum(box_mins, anchor_mins)  #左上交叉点
        intersect_maxes = np.minimum(box_maxes, anchor_maxes) #右下交叉点
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.) #交集计算，小于哦则归0
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1] #计算交集面积
        box_area = wh[..., 0] * wh[..., 1]                           #1   
        anchor_area = anchors[..., 0] * anchors[..., 1]              #2   
        iou = intersect_area / (box_area + anchor_area - intersect_area) #3计算并集面积
        #这里的iou维度格式[对象，anchor_num，iou值]
        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)  #最后一个维度去掉，在最后一个维度上得出最大值，返回一个三维的索引                
        #np.argmax多维度操作时种非常复杂的操作，我们可以这么想先确定维度，维度的组，维度的维数以及维数中的元素，每一个维数中的元素在组中的对应位置，我们对它们的对应位置进行比较
        for t, n in enumerate(best_anchor):              #先将一张图片中每个对象的最大化索引取出来，
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')     #计算在对应的y_true中的x的位置
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')     #计算在对应的x_true中的y的位置
                    k = anchor_mask[l].index(n)                                           #计算所在anchor的位置，这是本计算的核心需要
                    c = true_boxes[b,t, 4].astype('int32')                                #类别信息
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]                     #往y_true中加入x,y,w,h相对于整张图片进行归一化过的顺序为x,y,w,h
                    y_true[l][b, j, i, k, 4] = 1                                          #对应的置信度值1
                    y_true[l][b, j, i, k, 5+c] = 1                                        #对应的类别信息为1  

    return y_true


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)                         #在b1最后第二行多加一个维度[i1,....,iN,1,4]
    b1_xy = b1[..., :2]                                #在b1进行切片分离出x,y
    b1_wh = b1[..., 2:4]                               #在b1进行切片分离出w,h
    b1_wh_half = b1_wh/2.                              #计算wh/2，对b1_wh进行1/2计算
    b1_mins = b1_xy - b1_wh_half                       #计算左上角点
    b1_maxes = b1_xy + b1_wh_half                      #计算右下角点

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)                          #在b1第一行增加一个维度维度变为[1,j,4]
    b2_xy = b2[..., :2]                                #同上
    b2_wh = b2[..., 2:4]                               #同上
    b2_wh_half = b2_wh/2.                              #同上
    b2_mins = b2_xy - b2_wh_half                       #同上
    b2_maxes = b2_xy + b2_wh_half                      #同上 

    intersect_mins = K.maximum(b1_mins, b2_mins)       #进行左上交叉点计算 维度变为[i1,i2,i3,...iN,j,2]
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)    #进行右下交叉点计算 维度变为[i1,i2,i3....iN,j,2]
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)  #进行wh 维度[i1,i2,i3....iN,j,2]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]    #进行交集计算[i1,i2,i3....iN,j] 这里需要注意的时tensor运算，如果切边结果保留多个数值，运算结果末尾保留维度，如果切边结果保留一个数值，运算结果删除维度
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]                         #进行面积计算[i1,i2,i3....iN,j]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]                         #进行面积计算[i1,i2,i3....iN,j]
    iou = intersect_area / (b1_area + b2_area - intersect_area)      #进行iou计算[i1,i2,i3....iN,j]

    return iou  #return


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):   
    #这部分时损失函数计算 args前三层是输出层数据，后三层是true_box处理后的数据
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]   #取出神经网络输出
    y_true = args[num_layers:]         #去除true_box处理后的值
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    #input_shape为13x32=416 最终为416x416
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    #grid_shapes=[[13,13],[26,26],[52,52]]
    loss = 0                                 #损失初始化为0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))   #定义参数类型

    for l in range(num_layers):                     #分层计算loss
        object_mask = y_true[l][..., 4:5]           #取出对象true置信度来进行掩码
        true_class_probs = y_true[l][..., 5:]       #取出对象true类别信息，将来用来计算

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        #输入yolo_outputs[l] 类数量，输入尺度以及是否需要计算损失
        #返回grid, feats, box_xy, box_wh后两者经过归一化
        pred_box = K.concatenate([pred_xy, pred_wh])
        #在最后一层中对pred_xy与pred_wh进行拼接

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid  #对xy进行还原 ，本来是针对整个图片归一化的现在变成了针对某个cell，在计算loss是feats[...,0:2]应该先sigmod然后再求损失
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1]) #直接与feat相减求损失函数
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf 通过swith函数将-inf直接变成0
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]  #box损失函数权重计算2-wh

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)           #创建一个动态的TensorArray
        object_mask_bool = K.cast(object_mask, 'bool')                                        #将object_mask,转换层bool型的
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])       #使用object_mask_bool进行掩码
            iou = box_iou(pred_box[b], true_box)                                              #计算iou 生成维度为生成第b张图片的[j,i,achor_n,true_box_object_num]
            best_iou = K.max(iou, axis=-1)                                                    #在最后一个维度上取最大值生成维度为[j,i,achor_n]
            #tf.keras.backend.max(x,axis=None,keepdims=False)  keepdims表示是否保留最后一个维度 False表示不保留
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            #在tensor指定位置b写入tensor，写入的tensor根据best_iou与ignore_thresh来，如果best_iou的值小于ignore_thresh则写1，否则写0
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        #这里执行循环操作b *args作为形参 一方面接受loop_body的返回值，另外一方面作为loop_body的参数在循环中反复的传入
        ignore_mask = ignore_mask.stack()
        #叠堆ignore_mask.stack()使之成为[b,j,i,achor]
        ignore_mask = K.expand_dims(ignore_mask, -1)
        #在ignore_mask的最后加一个维度使之成为[b,j,i,achor,1]这可以使得它更加方便后期的计算
        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)
        #损失函数这一块需要注意 binary_crossentropy在计算时是否需要注意(raw_true_xy-sigmod(raw_pred[...,0:2])^2 然后计算。
        #损失函数的计算这块需要了解
        xy_loss = K.sum(xy_loss) / mf
        #计算平均xy_loss
        wh_loss = K.sum(wh_loss) / mf
        #计算平均wh_loss
        confidence_loss = K.sum(confidence_loss) / mf
        #计算平均confidence_loss
        class_loss = K.sum(class_loss) / mf
        #计算平均class_loss
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        #loss相加
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
        #打印损失值，为训练时实时观察
    return loss
