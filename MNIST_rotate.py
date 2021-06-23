import numpy as np
import pandas as pd
import time
from scipy import ndimage
start = time.time()

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
for i in range(len(df)):
    picture = np.array(df[i:i+1].drop(['label'], axis=1)).reshape(28,28)

    picture_left = ndimage.interpolation.rotate(picture,10,cval=0.01, reshape =False)
    picture_right = ndimage.interpolation.rotate(picture,-10,cval=0.01, reshape =False)

    picture_left = list(np.array(np.ravel(picture_left)))
    picture_right = list(np.array(np.ravel(picture_right)))
    label = list(np.array(df[i:i+1]['label']))
    picture_l = np.array(label + picture_left)
    picture_r = np.array(label + picture_right)

    df.loc[len(df)] = picture_l
    df.loc[len(df)] = picture_r
    if i % 100 == 99:
        print(i + 1, ':', time.time() - start)

df.to_csv('df_rotate_120000.csv', index=False)


