# coding=utf-8
"""读取数据集数据
@author: hehaoming
@time: 2019/10/27 21:00
"""
import os
import numpy as np
import logging
logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.WARNING)

DATA_PATH = "../../../resource/data"


def read_data_from_resource(dataset="dataset1"):
    """读取数据集数据
    Args: dataset is a string, dataset = "dataset1" or "dataset2"
    输入"dataset1"得到第一个数据集的数据，输入"dataset2"得到第二个数据集数据
    """
    if dataset == "dataset1":
        x_path = "ex4x.dat"
        y_path = "ex4y.dat"
        x1_index = 3
        x2_index = 6
        y_index = 3
    elif dataset == "dataset2":
        x_path = "iris_x.dat"
        y_path = "iris_y.dat"
        x1_index = 0
        x2_index = 1
        y_index = 0
    else:
        raise Exception("不存在这个数据集")
    logging.debug("%s %s %s", x1_index, x2_index, y_index)
    
    # 读取训练数据
    train_data = []
    for index, line in enumerate(open(os.path.join(DATA_PATH, x_path), 'r')):
        item = line.replace("\n", "").split(" ")
        logging.debug("%d %s %s", index, float(item[x1_index]), float(item[x2_index]))
        train_data.append([1., float(item[x1_index]), float(item[x2_index])])
    logging.debug("%s %s", "train_data", train_data)

    # 读取训练标签
    train_labels = []
    for index, line in enumerate(open(os.path.join(DATA_PATH, y_path), 'r')):
        item = line.split(" ")
        logging.debug("%d %s", index, int(float(item[y_index])))
        train_labels.append(int(float(item[y_index])))
    logging.debug("%s %s", "train_labels", train_labels)
    return np.array(train_data, dtype=np.float), np.array(train_labels, dtype=np.int)


if __name__ == "__main__":
    logging.debug("\n%s\n%s", *read_data_from_resource("dataset1"))
