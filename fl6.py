'''
tensorboard
'''
import tensorflow as tf



def add_layer(inputs,in_size,out_size,activation_function=None):
    with tf.name_scope('layer'):
        #随机值
        with tf.name_scope('W'):
            Weights=tf.Variable(tf.random_normal([in_size,out_size]),name="W")
        #0.1
        with tf.name_scope('b'):
            biases=tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.matmul(inputs,Weights)+biases
        if activation_function is None:
            outputs=Wx_plus_b
        else:
            outputs=activation_function(Wx_plus_b)
        return outputs

#define placeholder for inputs to network
with tf.name_scope('input'):
    xs=tf.placeholder(tf.float32,[None,1],name="x_input")
    ys=tf.placeholder(tf.float32,[None,1],name="y_input")
#add hidden layer
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
#add output layer
prediction=add_layer(l1,10,1,activation_function=None)

#the error betwween prediction and real data
with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#初始化变量

sess=tf.Session()

writer = tf.summary.FileWriter("logs/",sess.graph)

#must
sess.run(tf.initialize_all_variables())







