'''
session 会话控制
'''
import tensorflow as tf

maxtrix1=tf.constant([[3,3]])
maxtrix2=tf.constant([[2],
                      [2]])

product=tf.matmul(maxtrix1,maxtrix2)

#method1
# sess=tf.Session()
# result=sess.run(product)
# print(result)
# sess.close()


#method2
with tf.Session() as sess:
    result2=sess.run(product)
    print(result2)



