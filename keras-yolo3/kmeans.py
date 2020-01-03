import numpy as np

#此处YOLO_Kmeans算法
class YOLO_Kmeans:

    def __init__(self, cluster_number, filename):
        self.cluster_number = cluster_number                      #团簇数量
        self.filename = "2012_train.txt"                          #训练对应的文件名

    def iou(self, boxes, clusters):  # 1 box -> k clusters        #iou计算
        n = boxes.shape[0]                                        #取出box的数量
        k = self.cluster_number                                   #确定的box团簇的数量

        box_area = boxes[:, 0] * boxes[:, 1]                      #获取box的面积
        box_area = box_area.repeat(k)                             #在axis=None 将数组变化成为一个维度，然后扩展k倍，k为团簇数量
        box_area = np.reshape(box_area, (n, k))                   #将其分成k列和n行，没一行对应一个box，但是box的面积复制了k份

        cluster_area = clusters[:, 0] * clusters[:, 1]            #cluster团簇面积 第一维度为团簇数量，第二维度表示每个团簇当前的值
        cluster_area = np.tile(cluster_area, [1, n])              #在第一个维度上保持不变，在第二个维度上，扩展n列
        cluster_area = np.reshape(cluster_area, (n, k))           #在定义cluster类型 n行 k列

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))  #再定义类型n行k列，这里的坐标是宽度w
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))  #在定义类型n行k列，k列代表团簇数量
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix) #生成n行 k列，在对比之后将cluster_w与box_w中较小的取出 生成n行k列

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))  #同上 [[1,2],[3,4]]axis=none [1,2,3,4] [1,1,2,2,3,3,4,4]
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k)) #同上
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix) #同上
        inter_area = np.multiply(min_w_matrix, min_h_matrix)      #计算交集面积

        result = inter_area / (box_area + cluster_area - inter_area) 
        #计算iou 行数等于当前图片对象数量，列数为k，列中每个元素等于 对象与K个团簇的iou
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        #np.mean用户压缩各列 当axis=1时
        #np.mean 当axis=0时表示压缩各行
        #最后生成n行 每一行代表一个对象，它们数值代表平均值
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):   #kmean算法主体
        box_number = boxes.shape[0]               #box-shape[n,2] ,n代表行数 就是对象的数量
        distances = np.empty((box_number, k))     #np.empty建立一个空的数组 (n,k)
        last_nearest = np.zeros((box_number,))    #建立一个[n]维度的数组
        np.random.seed()                          #产生随机数种子，后面第一个产生的随机数完全一样
        clusters = boxes[np.random.choice(box_number, k, replace=False)]  # init k clusters 
        #这里np.random.choice[a,b,replace,p]
        #a指定arange(a) 数组中取数
        #b指定取数的数量
        #replace 确定取数是否可重复
        #p指定概率分别，None代表均匀分布
        #这里是从box-number中随机取出k个数，对应boxes[]生成clusters
        while True:

            distances = 1 - self.iou(boxes, clusters)
            #生成(n,k)的数组，每个内容等于1-iou
            current_nearest = np.argmin(distances, axis=1)
            #生成(n,)的数组,每个内容等于对应对象对于团簇的每个对象的1-iou距离的最小值的位置
            if (last_nearest == current_nearest).all(): #当上下两轮的current_nearest不在变化是break
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)
            #以上对boxes中的对象分类，将同一团簇的w，h求均值，重新赋值给clusters，然后形成新的团簇

            last_nearest = current_nearest #把当前的nearest保存下来，然后和下一轮mean进行比较

        return clusters

    def result2txt(self, data):                        #将结果写道文件中去
        f = open("yolo_anchors.txt", 'w')              #只读形式打开yolo_anchors.txt
        row = np.shape(data)[0]                        #确定数据的行数写入anchors.txt
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):                               #重文件中读入指定的
        f = open(self.filename, 'r')
        dataSet = []
        for line in f:
            infos = line.split(" ")
            length = len(infos)
            for i in range(1, length):
                width = int(infos[i].split(",")[2]) - \           
                    int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - \
                    int(infos[i].split(",")[1])
                #x1-x2=width  y1-y2=height
                dataSet.append([width, height])
        result = np.array(dataSet)   #生成numpy对象
        f.close()                    #文件关闭
        return result

    def txt2clusters(self):                                              #读入团簇信息
        all_boxes = self.txt2boxes()                                     #从txt读入boxes信息
        result = self.kmeans(all_boxes, k=self.cluster_number)           #根据 allbox信息与cluster_numbber计算result结果
        result = result[np.lexsort(result.T[0, None])]                   #转置后根据第一行进行排序，之后返回排序索引，result根据索引重新排列
        self.result2txt(result)                                          #写入文件
        print("K anchors:\n {}".format(result))                          #打印result
        print("Accuracy: {:.2f}%".format(                                #答应平均iou
            self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    cluster_number = 9
    filename = "2012_train.txt"
    kmeans = YOLO_Kmeans(cluster_number, filename)
    kmeans.txt2clusters()
