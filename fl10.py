'''
特征抽取
'''
#导入包
from sklearn.feature_extraction.text import CountVectorizer
#实例化CountVectorizer
vector=CountVectorizer()
#调用fit_transfrom输入并转换数据
res=vector.fit_transform(["Life is short,i like python","Life is too long,i hate python"])
#打印结果

print(vector.get_feature_names())

print(res.toarray())














