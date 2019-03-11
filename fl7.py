'''
activation function
'''
import tensorflow as tf
import numpy as np

def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    layer_name='layer%s'%n_layer;
    # with tf.name_scope('layer_name'):
        # 随机值
    with tf.name_scope('W'):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]), name="W")
        tf.summary.histogram(layer_name+'/weights',Weights)
        # 0.1
    with tf.name_scope('b'):
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        tf.summary.histogram(layer_name + '/biases',biases)

    with tf.name_scope('Wx_plus_b'):
         Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs

x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise

xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])
#add hidden layer
l1=add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu)
#add output layer
prediction=add_layer(l1,10,1,n_layer=2,activation_function=None)
#the error betwween prediction and real data
with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#初始化变量
sess=tf.Session()
#合并summary
merged=tf.summary.merge_all()

writer = tf.summary.FileWriter("logs/",sess.graph)

#must
sess.run(tf.initialize_all_variables())

# 运行并输出结果
for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%40==0:
        result=sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)





