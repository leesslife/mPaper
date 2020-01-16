"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data


def _main():
    annotation_path = 'train.txt'                               #训练数据 每一行代表着一个数据对象的输出
    log_dir = 'logs/000/'                                       #日志文件保存地址
    classes_path = 'model_data/voc_classes.txt'                 #类信息保存位置
    anchors_path = 'model_data/yolo_anchors.txt'                #模型anchors信息保存位置
    class_names = get_classes(classes_path)                     #获取类名生成list
    num_classes = len(class_names)                              #获取类的数量
    anchors = get_anchors(anchors_path)                         #同前

    input_shape = (416,416) # multiple of 32, hw                #输入416

    is_tiny_version = len(anchors)==6 # default setting         #是否训练tiny版本模型是6
    if is_tiny_version:                                         #如果is_tiny_version存在并且不等于0
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
        #创建冻结模型，并冻结最后两层 但创建的是create_tiny_model tiny模型加载的也是tiny_yolo_weights权重文件
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/yolo_weights.h5') # make sure you know what you freeze
        #创建冻结模型，并冻结最后两层，但创建的是create_model模型，加载的也是yolo_weights权重文件

    logging = TensorBoard(log_dir=log_dir)
    #设置日志位置
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    #设置训练检查点 monitor为='val_loss'
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    #设置学习速率衰变
    #verbose=1 用于是否展示
    #每次衰变为x0.1
    #当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    #触发早停
    #监测值为val_loss 
    #min_delta代表增大或减小的阈值，只有大于这个部分才算作improvement
    #patience=10 当loss停止改善10个EPOCH之后就开始早停
    #verbose信息展示模式
    val_split = 0.1
    #训练值与测试值 的分值线
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    #开始时随机打乱行数
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    #验证集数量
    num_train = len(lines) - num_val
    #训练集数量
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})
        #编译并定义损失函数
        batch_size = 32
        #batch_size定义
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
        #训练与验证
        #第一个data_generator_wrapper产生训练集合
        #第二个data_generator_wrapper产生验证集合
        #steps_per_epoch一个周期产生的步数
        #validation_steps验证集步数
        #周期数量50
        #初始化周期
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')
        #保存权重
    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    #如果训练周期不够好，解冻模型
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
            #解冻模型
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 32 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=100,
            initial_epoch=50,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')
        #同上
    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names
    #同前 生成类名的list

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)
    #获取anchor生成np array[n,2] n是anchor数量


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    #创建正常yolov3模型，load_pretrained=True 为加载与训练，与训练的权重文件再weights_paths的位置
    K.clear_session() # get a new session
    #清楚过去的会话
    image_input = Input(shape=(None, None, 3))
    #定于image_input的(416,416,3)应该数据类型就是这个
    h, w = input_shape
    #获取h,w的值yolov3应该是416
    num_anchors = len(anchors)
    #获取num_anchors==9 或者==6
    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]
    #生成y_true列表[list][shape0(13,13,n),shape1(26,26,n),shape2(52,52,n)]
    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    #生成完整的yolov3模型
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    #打印格式 anchors数量，num_classes,数量
    if load_pretrained:
        #是否加载与训练模型
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        #加载预训练模型，设定通过命名，对于不匹配的地方扫描并合并
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            #获取冻结模型的方法 这里冻结模型有两种方法
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            #第一种冻结方法设定为freeze_body[0]==185 第二种冻结设定方法为freeze_body[1]=max_layer-3
            for i in range(num): model_body.layers[i].trainable = False
            #设置冻结层
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    #将损失函数加入到模型中
    model = Model([model_body.input, *y_true], model_loss)
    #生成最终模型

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    #清理会话，同上
    image_input = Input(shape=(None, None, 3))
    #设置图像输入shape
    h, w = input_shape
    #同上
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]
    #tiny的y_true的shape为[shape0[13x13xnum_classes+5],shape1[26x26xnum_classes+5]]
    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    #创建tiny模型
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        #加载权重
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
            #冻结
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
        #损失函数加入莫能行
    model = Model([model_body.input, *y_true], model_loss)
    #完成最终模型
    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    #fit_generator数据生成器
    n = len(annotation_lines)
    #图片总数
    i = 0
    while True:
        image_data = []
        #图像数据容器
        box_data = []
        #图像数据容器
        for b in range(batch_size):
            #生成batch_size数据
            if i==0:
                np.random.shuffle(annotation_lines)
            #如果所有数据图像数据都已经完成了一边，则重新打乱顺序
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            #利用随机方法重新改变图像数据，包括image图像实际数据以及图片对应的box(x1,x2,y1,y2,c)
            image_data.append(image)
            #放入image_data容器
            box_data.append(box)
            #放入box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        #将image_data转化成np容器
        box_data = np.array(box_data)
        #将box_data转化成np容器
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        #改变[batch_sizex13x13x(num_classes+5),batch_sizex26x26x(num_classes+5),batch_sizex52x52x(num_classes+5)]
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    #排除n==0以及batch_size<=0的两种无效情况
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main()
