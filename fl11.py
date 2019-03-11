'''
特征抽取
'''
#导入包
from sklearn.feature_extraction import DictVectorizer

def dictvec():
    '''
    字典数据抽取   把字典中一些类别数据，分别转换成特征
    :return:
    '''
    #实例化
    # dict=DictVectorizer() #if sparse=False,则不返回sparse矩阵

    dict=DictVectorizer(sparse=False) #One-hot 编码
    data=dict.fit_transform([{'city':'北京','temperture':100},{'city':'上海','temperture':90},{'city':'深圳','temperture':80}])
    #打印sparse矩阵格式，节约内存，方便读取处理
    print(dict.get_feature_names())
    print(data)

if __name__ == '__main__':
    dictvec()













