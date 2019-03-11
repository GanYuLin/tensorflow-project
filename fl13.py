'''
预处理
归一化：通过原始数据进行变换把数据映射到（默认[0,1]）之间
'''
from sklearn.preprocessing import MinMaxScaler

#归一化
def mm():
    '''
    归一化处理,适合传统精确小数据场景，容易受最大值和最小值的影响
    :return:
    '''
    mm=MinMaxScaler(feature_range=(2,3))
    data=mm.fit_transform([[90,2,10,40],[60,4,15,45],[75,3,13,46]])
    #打印数据
    print(data)





if __name__ == '__main__':
    mm()







