import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image

def detect_img(yolo):          #开始侦测图片
    while True:
        img = input('Input image filename:')  #每个循环时需要输入图片路径
        try:
            image = Image.open(img)           #打开图片并获取image，失败则捕获异常
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)          #侦测图片并标出对象位置
            r_image.show()                              #展示图片
    yolo.close_session()                                #关闭会话

FLAGS = None                                            #FLAGS=None？

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)          #获取参数解析器
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )
    #用model来指定模型权重文件路径，如果不指定这个参数，则返回默认路径
    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )
    #指定anchors文件路径，如果没有则返回anchors默认路径
    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )
    #指定class路径 同上
    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )
    #指定gpu数量，没有则返回默认数量
    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    #指定image路径
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )
    #指定video路径
    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )
    #video指定输出路径
    #解析所有参数
    FLAGS = parser.parse_args()

    if FLAGS.image: #如果image路径存在，则使用图片识别模型
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        #如果image路径存在，则直接忽略video参数中的input 与output
        detect_img(YOLO(**vars(FLAGS)))
        #参数导入到YOLO对象中，开始侦测图像
    elif "input" in FLAGS:  #确定侦测视频
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
        #参数导入到侦测视频接口，开始侦测视频并可以保留视频保存位置，带有对象标签
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
