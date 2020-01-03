"""
Retrain the YOLO model for your own dataset.
"""
#这里是带瓶颈的训练
import os
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data


def _main():
    annotation_path = 'train.txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/coco_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)                    #根据classes_path获取class_names数组
    num_classes = len(class_names)                             #总共的类的数量，这里的类的数量会影响到后续的卷积计算
    anchors = get_anchors(anchors_path)                        #获取anchors的二维数组[n,2]

    input_shape = (416,416) # multiple of 32, hw               #定义输入类型

    model, bottleneck_model, last_layer_model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/yolo_weights.h5') # make sure you know what you freeze
    #以上形成了标准model模型，冻结后的瓶颈模型，以及瓶颈模型下最后一层的模型
    logging = TensorBoard(log_dir=log_dir)
    #确定日志路径？
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    #filename：字符串，保存模型的路径
    #monitor：需要监视的值
    #verbose：信息展示模式，0或1(checkpoint的保存信息，类似Epoch 00001: saving model to ...)
    #save_best_only：当设置为True时，监测值有改进时才会保存当前的模型（ the latest best model according to the quantity monitored will not be overwritten）
    #mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，例如，当监测值为val_acc时，模式应为max，当监测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
    #save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
    #period：CheckPoint之间的间隔的epoch数
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    #ReduceLROnPlateau()
    #monitor='val_loss'监测值
    #factor=0.1 降低比率
    #patience 如果3个周期内 比例不在改变，降低学习率变为原来的0.1
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    #设定早停
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        # perform bottleneck training
        if not os.path.isfile("bottlenecks.npz"):
            print("calculating bottlenecks")
            batch_size=8
            bottlenecks=bottleneck_model.predict_generator(data_generator_wrapper(lines, batch_size, input_shape, anchors, num_classes, random=False, verbose=True),
             steps=(len(lines)//batch_size)+1, max_queue_size=1)
            np.savez("bottlenecks.npz", bot0=bottlenecks[0], bot1=bottlenecks[1], bot2=bottlenecks[2])
    
        # load bottleneck features from file
        dict_bot=np.load("bottlenecks.npz")
        bottlenecks_train=[dict_bot["bot0"][:num_train], dict_bot["bot1"][:num_train], dict_bot["bot2"][:num_train]]
        bottlenecks_val=[dict_bot["bot0"][num_train:], dict_bot["bot1"][num_train:], dict_bot["bot2"][num_train:]]

        # train last layers with fixed bottleneck features
        batch_size=8
        print("Training last layers with bottleneck features")
        print('with {} samples, val on {} samples and batch size {}.'.format(num_train, num_val, batch_size))
        last_layer_model.compile(optimizer='adam', loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        last_layer_model.fit_generator(bottleneck_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, bottlenecks_train),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=bottleneck_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes, bottlenecks_val),
                validation_steps=max(1, num_val//batch_size),
                epochs=30,
                initial_epoch=0, max_queue_size=1)
        model.save_weights(log_dir + 'trained_weights_stage_0.h5')
        
        # train last layers with random augmented data
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})
        batch_size = 16
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 4 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=100,
            initial_epoch=50,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]       #读入字符串，去掉首位，形成二维数组
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:                         
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]              #用逗号进行分割，形成一维数组
    return np.array(anchors).reshape(-1, 2)                       #在定义成二维[n,2]的二维数组


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))    #建立一个input的三通道容器
    h, w = input_shape                            #h,w=input_shape 416
    num_anchors = len(anchors)                    #anchor的数量

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]
    #生成一个输入容器[13,13,9//3,n+5]当然还有26x26 52x52 y_true是一个数组
    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    #yolo_生成yolov3 body模型 Model(inputs, [y1,y2,y3])
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    #输出num_anchors 和num_class
    if load_pretrained:   #允许登录预训练
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        #登录权重，加载name 并且对不匹配的内容进行匹配 by_name 和skip_mismatch 什么意思？
        print('Load weights {}.'.format(weights_path))
        #打印权重
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            # 当freeze_body=1时 num=185
            # 当freeze_body=2时 num=len(model_body.layer)-3
            for i in range(num): model_body.layers[i].trainable = False
            # freeze_body=1 冻结前面185层
            # frezze_body=2 冻结前面model_body.layer-3
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
            #打印冻结层

    # get output of second last layers and create bottleneck model of it
    out1=model_body.layers[246].output
    out2=model_body.layers[247].output
    out3=model_body.layers[248].output
    bottleneck_model = Model([model_body.input, *y_true], [out1, out2, out3])

 
    # create last layer model of last layers from yolo model
    in0 = Input(shape=bottleneck_model.output[0].shape[1:].as_list()) 
    #获取维度信息列表 output0
    in1 = Input(shape=bottleneck_model.output[1].shape[1:].as_list())
    #获取维度信息列表 output1
    in2 = Input(shape=bottleneck_model.output[2].shape[1:].as_list())
    #获取维度信息列表 output2
    last_out0=model_body.layers[249](in0)
    last_out1=model_body.layers[250](in1)
    last_out2=model_body.layers[251](in2)
    model_last=Model(inputs=[in0, in1, in2], outputs=[last_out0, last_out1, last_out2])
    #创建最后一层模型
    model_loss_last =Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_last.output, *y_true])
    #yolo_loss中arguments是提前填入参数
    #[*model_last.output,*y_true]等于args
    last_layer_model = Model([in0,in1,in2, *y_true], model_loss_last)
    #建立基于瓶颈模型的最后一层
    
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    #此处建立非瓶颈模型的最后一层

    return model, bottleneck_model, last_layer_model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, random=True, verbose=False):
    '''data generator for fit_generator'''
    n = len(annotation_lines)           #建立需要解析的行数(annotation 行数) 一张图片的对象annotation 行数
    i = 0                               
    while True:                     
        image_data = []                 #image_data 放置图片的容器
        box_data = []                   #box_data 放置trux box的容器
        for b in range(batch_size):    #根据batch_size 生成 一捆数据
            if i==0 and random:                    
                np.random.shuffle(annotation_lines)              #打乱annotation_lines                            
            image, box = get_random_data(annotation_lines[i], input_shape, random=random)  
            #形成image,数据，根据图片尺度，居中归一化之后的数据，box是纠正好后的box数据，n个对象对应n行，一行对应的参数应该是x1,x2,y1,y2,c
            image_data.append(image)
            #往数据容器中注入图像数据
            box_data.append(box)
            #往数据容器中注入box数据
            i = (i+1) % n
            #如果图片数量不够则利用循环补充
        image_data = np.array(image_data)
        #将数据分装成np.array类型数据
        if verbose:
            print("Progress: ",i,"/",n)
        box_data = np.array(box_data)
        #将数据封装成np.array数据
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        #对y_true拥有三个对象的数组
        #y_true对象13x13x75
        #y_true对象26x26x75
        #y_true对象52x52x75
        yield [image_data, *y_true], np.zeros(batch_size)
        #根据数据生成器，生成数据容器(根据需要来生成)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes, random=True, verbose=False):
    n = len(annotation_lines)                                  #当前数据中的图片数量
    if n==0 or batch_size<=0: return None                      #图片数量n==0,或者batch_size<=0 直接返回 不做处理
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, random, verbose)  #利用数据生成器来生产所需要的数据

def bottleneck_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, bottlenecks):
    n = len(annotation_lines)
    i = 0
    while True:
        box_data = []
        b0=np.zeros((batch_size,bottlenecks[0].shape[1],bottlenecks[0].shape[2],bottlenecks[0].shape[3]))
        b1=np.zeros((batch_size,bottlenecks[1].shape[1],bottlenecks[1].shape[2],bottlenecks[1].shape[3]))
        b2=np.zeros((batch_size,bottlenecks[2].shape[1],bottlenecks[2].shape[2],bottlenecks[2].shape[3]))
        for b in range(batch_size):
            _, box = get_random_data(annotation_lines[i], input_shape, random=False, proc_img=False)
            box_data.append(box)
            b0[b]=bottlenecks[0][i]
            b1[b]=bottlenecks[1][i]
            b2[b]=bottlenecks[2][i]
            i = (i+1) % n
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [b0, b1, b2, *y_true], np.zeros(batch_size)

if __name__ == '__main__':
    _main()
