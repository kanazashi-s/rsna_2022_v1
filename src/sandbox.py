import pandas as pd
import numpy as np


if __name__ == '__main__':
    a = np.array([0, 1, 0, 0, 1])
    print(np.isin(a, [0, 1]).all())
    pass
