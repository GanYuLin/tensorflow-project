'''
overfitting 过拟合
'''
import tensorflow as tf
from  sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

#load data
digits=load_digits()
X=digits.data



def add_layer(inputs,in_size,out_size,activation_function=None):
    #随机值
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    #0.1
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)

    Wx_plus_b=tf.matmul(inputs,Weights)+biases

    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs,v_ys):
    global  prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result


#define placeholder for inputs to network
xs=tf.placeholder(tf.float32,[None,784]) #28*28
ys=tf.placeholder(tf.float32,[None,10]) #28*28

#add output layer
prediction=add_layer(xs,784,10,activation_function=tf.nn.softmax)

#the error between prediction and real data
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1])) #loss
#train
train_step=tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)


sess=tf.Session()
#important step
sess.run(tf.initialize_all_variables())

for i in range(1000):

    if i%50==0:
        pass




