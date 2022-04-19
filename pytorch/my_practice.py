import torch
import os
import pandas as pd
import csv
# import pandas.util.testing
import random
# import matplotlib
from d2l import torch as d2l


# df = pd.util.testing.makeMixedDataFrame()


from matplotlib import pyplot as plt

plt.figure(figsize = (10,8))

labels =[u'Quantitative', u'Qualitative', u'Theory', u'Mixed']
sizes = [38,33,13,9]
colors = ['#80AFBF','#608595','#DFC286','#C07A92']
distance =0
explode = (distance,distance,distance,distance)
patches,text1,text2 = plt.pie(sizes,
                             explode=explode,
                             colors=colors,
                              labels=None,
                              labeldistance = 2.2,
                              autopct = '%.2f%%',
                              shadow = False,
                              startangle = 90,
                              pctdistance =0.8)
plt.axis('equal')
plt.title('Research Methods')
plt.legend(labels,loc = 1,title = 'Method')
plt.show()


import pdb; pdb.set_trace()





def synthetic_data(w, b, num_examples):
    """生成 y = Xw +b + 噪声"""
    X = torch.normal(0,1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1,1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
