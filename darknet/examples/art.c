#include "darknet.h"

#include <sys/time.h>

void demo_art(char *cfgfile, char *weightfile, int cam_index)
{
#ifdef OPENCV
    //如果opencv被定义了话，执行以下操作
    network *net = load_network(cfgfile, weightfile, 0);
    //加载配置文件和权重文件
    set_batch_network(net, 1);
    //设定batch为1
    srand(2222222);
    //随机数种子

    void * cap = open_video_stream(0, cam_index, 0,0,0);
    //根据cam_index获取摄像头句柄
    char *window = "ArtJudgementBot9000!!!";
    //获取窗口名
    if(!cap) error("Couldn't connect to webcam.\n");
    int i;
    int idx[] = {37, 401, 434};
    //整数数组[37,401,434]
    int n = sizeof(idx)/sizeof(idx[0]);
    //整数数组的长度

    while(1){
        image in = get_image_from_stream(cap);
        //根据数据流cap获取图像数据
        image in_s = resize_image(in, net->w, net->h);
        //重新定义图像大小，根据net->w net->h

        float *p = network_predict(net, in_s.data);
        //将图像数据输入到网络中，并获取最终的预测数据
        printf("\033[2J");
        printf("\033[1;1H");

        float score = 0;
        //初始化得分为0
        for(i = 0; i < n; ++i){
            float s = p[idx[i]];
            //获取p[37],p[401],p[434]中的得分，通过if语句获取到三个得分中的最大值
            if (s > score) score = s;
        }
        score = score;
        //再次赋值得分
        printf("I APPRECIATE THIS ARTWORK: %10.7f%%\n", score*100);
        //
        printf("[");

	int upper = 30;
        for(i = 0; i < upper; ++i){
            printf("%c", ((i+.5) < score*upper) ? 219 : ' ');
        }
        //最大得分score乘以upper  如果大于i+0.5 则输入219 否则输入''
        printf("]\n");

        show_image(in, window, 1);
        //展示每一帧图片
        free_image(in_s);
        //释放图片数据
        free_image(in);
        //释放图片数据
    }
#endif
}


void run_art(int argc, char **argv)
{
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    //根据-c 获取摄像头索引
    char *cfg = argv[2];
    //获取配置文件
    char *weights = argv[3];
    //回去权重文件
    demo_art(cfg, weights, cam_index);
    //根据配置文件，权重文件，索引摄像头索引
}

