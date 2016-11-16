# -*- coding: UTF-8 -*-
#加载数据的说明文件 

# labels 1~6对应六种情感
# 1   angry
# 2   fear
# 3   happy
# 4   neutral
# 5   sad
# 6   surprise

import numpy as np

data_matrix = np.loadtxt('emo_fea_unscaled.csv',delimiter=',')
data_labels = np.loadtxt('emo_tag.csv',delimiter=',',dtype=int)
