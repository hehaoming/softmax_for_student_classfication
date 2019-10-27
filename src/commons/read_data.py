"""读取数据集数据
author: hehaoming
time: 2019/10/27 21:00
"""
import os

import numpy as np
import logging

logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)

DATA_PATH = "../../resource/data/ex4Data"
X_PATH = "ex4x.dat"
Y_PATH = "ex4y.dat"


def read_data_from_resource():
    # 读取训练数据
    train_data = []
    for index, line in enumerate(open(os.path.join(DATA_PATH, X_PATH), 'r')):
        item = line.split(" ")
        logging.debug("%d %s %s", index, float(item[3]), float(item[6]))
        train_data.append([float(item[3]), float(item[6])])
    logging.debug("%s %s", "train_data", train_data)

    # 读取训练标签
    train_labels = []
    for index, line in enumerate(open(os.path.join(DATA_PATH, Y_PATH), 'r')):
        item = line.split(" ")
        logging.debug("%d %s", index, item[3])
        train_labels.append(int(float(item[3])))
    logging.debug("%s %s", "train_labels", train_labels)
    return np.array(train_data, dtype=np.float), np.array(train_labels, dtype=np.int)


if __name__ == "__main__":
    logging.debug("%s %s", *read_data_from_resource())
