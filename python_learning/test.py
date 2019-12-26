from keras import backend as k
import tensorflow as tf
import numpy as np

b1=[[11,21,31],
   [21,22,23],
   [31,32,33]]
b2=[[41,42,43],
    [51,52,53],
    [61,62,63]]
b1_np=np.asarray(b1)
b2_np=np.asarray(b2)
b1_tensor=tf.convert_to_tensor(b1_np)
b2_tensor=tf.convert_to_tensor(b2_np)
c=b1_tensor[...,0:1]*b1_tensor[...,1:2]
d=tf.convert_to_tensor(np.zeros((5),dtype=np.int))
print(b1)
print(b2)
print(c)
print(d)