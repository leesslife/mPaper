import tensorflow as tf
import numpy as np

sess=tf.Session()
x=np.arange(20)
input_ta=tf.TensorArray(size=0,dtype=tf.int32,dynamic_size=True)
input_ta=input_ta.unstack(x)
for time in range(len(x)):
    print(sess.run(input_ta.read(time)))

output=input_ta.stack()
print(sess.run(output))

for time in range(5):
    input_ta=input_ta.write(time+len(x),time)
output=input_ta.stack()
print(sess.run(output))