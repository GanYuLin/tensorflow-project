'''
文本特征抽取
'''
#导入包
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import jieba
def countvec():
    '''
    对文本进行特征抽取
    :return:
    '''
    cv=CountVectorizer()
    data=cv.fit_transform(["Life is short,i like python","Life is too long,i hate python"])
    #统计文章中出现的所有词，重复只当做一次 词的列表
    #对每篇文章，在词的列表里面统计每个词出现的次数
    #单个字母不统计
    print(cv.get_feature_names())
    print(data.toarray())

def cutword():
    con1=jieba.cut("我想和女朋友一起去北京故宫博物院参观和闲逛。")
    con2 =jieba.cut("欧阳建国是创新办主任也是欢聚时代公司云计算方面的专家。")
    con3 =jieba.cut("他来到上海交通大学。")
    #转换成列表
    content1=list(con1)
    content2=list(con2)
    content3=list(con3)
    #把列表转换成字符串
    c1=' '.join(content1)
    c2=' '.join(content2)
    c3=' '.join(content3)

    return c1,c2,c3



def hanzivec():
    '''
       对文本进行特征抽取
       :return:
       '''
    c1,c2,c3=cutword()
    print(c1,c2,c3)
    cv = CountVectorizer()
    data = cv.fit_transform([c1,c2,c3])
    print(cv.get_feature_names())
    print(data.toarray())


def tfidvec():
    '''
       对文本进行特征抽取
       :return:
       '''
    c1,c2,c3=cutword()
    print(c1,c2,c3)
    tf = TfidfVectorizer()
    data =tf.fit_transform([c1,c2,c3])
    print(tf.get_feature_names())
    print(data.toarray())

if __name__ == '__main__':
    tfidvec()














