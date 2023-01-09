import pandas as pd
import numpy as np
import glob


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    # nums = [-3, 10, 50, -30, 2]
    # sigmoid_list = []
    # for num in nums:
    #     sigmoid_list.append(sigmoid(num))
    # print(sum(sigmoid_list) / 5)
    # print(sigmoid_list)
    #
    # nums_mean = np.mean(nums)
    # print(sigmoid(nums_mean))

    print(len(glob.glob("data/tmp_raw/test_images/*/*.dcm")))
