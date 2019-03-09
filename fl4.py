'''
placeholder
与feed_dict= 绑定
'''
import tensorflow as tf

input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)

#乘法
output=tf.multiply(input1,input2)
# 输出
with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))








