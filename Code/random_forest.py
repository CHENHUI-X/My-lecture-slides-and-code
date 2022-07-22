'''

1  anaconda ( Comes with Python )
2  Python + pycharm

    相关的库:  xlrd pandas numpy scipy sklearn matplotlib 等
    安装方法: 在cmd命令行(win+r) 输入  pip install 库名
    例子 : pip install pandas  即可安装pandas库

    anaconda 一般应该不用另外安装 基本都自带了


3 超实用函数 : help() 万物皆可help


4 项目流程及知识点
    读取数据
    处理数据
    模型选取
    模型建立
    模型预测


'''

#***********************************读取数据********************************

# 1 使用Python自带的数据集
from sklearn.datasets import load_boston

X = load_boston() # 读取数据
print(X['DESCR']) #字典类型
print(X['data'])

X,Y = load_boston(return_X_y=True)

# 2 pandas 读取磁盘文件
import pandas as pd  # 基于xlrd库,需要先安装xlrd  再安装pandas(在xlrd基础做了高级封装)

DATA_path = r'C:\users\Administrator\Desktop\iris.csv'
# r表示字符串不进行转义. 转义 : 比如'\n'表示换行
# 建议养成习惯字符串前边加r

DATA = pd.read_csv(
    filepath_or_buffer = DATA_path
)

# 或者直接 data = pd.read_csv(path)
# pd.read_excel 可用来读取excel表格

Test_set = [5.5,3.9,0.8,0.5] # 需要预测该样本所属类别

# 查看数据 对数据进行简单分析
print(DATA.shape) # 数据大小
print(DATA.head(10)) # 数据太多的话 可以输出前十行
print(DATA.describe()) # 简单的描述性分析


#***********************************数据预处理 ********************************
'''

1 缺失值处理 :
        填充 : 均值填充 中位数 等
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html
         
         
        插值 : 线性插值 多项式插值 拉格朗日插值等等(相当于高级的填充)
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html
        
        
        删除 : 删除空值
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
        
        
2 异常值处理 : 箱线图 绘图观察 3σ原则 描述性统计法

        https://cloud.tencent.com/developer/article/1429994
        描述性统计法 : 概率应该0-1之间 , 性别 : 男 女
        3σ原则 : 依Gaussian分布 ,离均值3倍标准差之外的数据出现的概率小于%1
        箱线图 : 大于1.5倍四分位差的为异常值,上须的计算公式为Q3+1.5(Q3-Q1)


path =  r'C:\Users\Administrator\Desktop\iris_error.xlsx'     
data = pd.read_excel(
    io = path
)
des = data.describe() # 简单的描述性分析
print(des) 

new_data = data[ data['SepalLength'] > 0 ] # 描述性统计法 假设已知某属性大于0


# 通过3σ原则删除 
mu_plus_3sigma = des.loc['mean'] + 3*des.loc['std']

new_data2 =   data[ 
                    (data['SepalLength'].abs() <=  mu_plus_3sigma[0]) \
                    & (data['SepalWidth'].abs() <=  mu_plus_3sigma[1]) \
                    & (data['PetalLength'].abs() <=  mu_plus_3sigma[2]) \
                    & (data['PetalWidth'].abs() <=  mu_plus_3sigma[3])
                ]


# 通过1.5倍四分位差删除
import matplotlib.pyplot as plt
plt.boxplot(data['SepalLength']) # 绘制箱线图

# 手动计算
Q_up =   des.loc['75%'] + 1.5* (  des.loc['75%'] -  des.loc['25%'] )
Q_down =   des.loc['25%'] - 1.5* (  des.loc['75%'] -  des.loc['25%'] )

new_data3 = data[ 
                    (data['SepalLength'] <=  Q_up[0]) &\
                  (data['SepalLength']  >=  Q_down[0])
                ]
  
  
                    
3 降维 or 特征提取 : 比如可以用之前学到的PCA 
    PCA :　可以翻看之前的直播　
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)  # 定义模型
pca.fit(DATA.iloc[:,:-1])

print(pca.explained_variance_) # 各个新轴上的方差,即XX^T各个特征值
print(pca.explained_variance_ratio_) # 方差解释百分比
print(pca.components_) # 各个特征值对应特征向量(标准化的),也叫主成分系数

pca_data = pca.transform(DATA.iloc[:,:-1])

X = pd.DataFrame( pca_data )
X['c'] = DATA.iloc[:,-1]
X.columns = ['x','y','c']
co = 'bkr' # blue 
for i in range(3): 
    plt.scatter(  X[X['c']== i]['x'],
                  X[X['c']== i]['y'],
                  c = co[i]
                  )
                  


4 数据拆分: 将数据集拆分为 训练集和验证集(与测试集不同)   
https://easyaitech.medium.com/%E4%B8%80%E6%96%87%E7%9C%8B%E6%87%82-ai-%E6%95%B0%E6%8D%AE%E9%9B%86-%E8%AE%AD%E7%BB%83%E9%9B%86-%E9%AA%8C%E8%AF%81%E9%9B%86-%E6%B5%8B%E8%AF%95%E9%9B%86-%E9%99%84-%E5%88%86%E5%89%B2%E6%96%B9%E6%B3%95-%E4%BA%A4%E5%8F%89%E9%AA%8C%E8%AF%81-9b3afd37fd58               
        train set : Training model
        validation set : Evaluation model    
        test set : Only used after the model is determined
                    not participating in the model training phase 
    
from sklearn.model_selection import  train_test_split # 拆分数据
x_train,x_test,y_train,y_test = train_test_split(
    DATA.iloc[:,:-1],DATA.iloc[:,-1],test_size = 0.2
            )#习惯写法
                
#事实上 , 这里的 x_test 应该对应的是训练集validation set 
# test_size=0.2指的也是验证集所占数据集的百分比
#一开始提到的 Test_set 才是真正的预测集,或叫测试集



    4.5 先标准化再拆分 VS 拆分之后做标准化
        https://blog.csdn.net/qq_40304090/article/details/90597892
        a : 数据窥探 or 信息泄露
        b : 考虑到测试集(预测集)标准化时,以哪个均值做标准化


5 数据标准化 , 连续值离散化  

from sklearn.preprocessing import StandardScaler #标准化包
stdScale = StandardScaler().fit(x_train) ## 生成规则
x_train = stdScale.transform(x_train) ## 将规则应用于训练集
x_test = stdScale.transform(x_test) ## 将规则应用于验证集 :( 验证集 - 训练集均值) / 训练集方差

'''

#***********************************模型建立与预测--classification  ********************************

# 暂时不实现降维等操作,直接使用原数据

from sklearn.tree import DecisionTreeClassifier #决策树模型
from sklearn.metrics import confusion_matrix  #混淆矩阵
from sklearn.metrics import accuracy_score  #准确率判断
from sklearn.model_selection import train_test_split  # 拆分数据
import pandas as pd
import numpy as np

DATA_path = r'C:\Users\Administrator\Desktop\iris.csv'
DATA = pd.read_csv(
    filepath_or_buffer = DATA_path
)



x_train, x_validation , y_train, y_validation = train_test_split(DATA.iloc[:,:-1],DATA.iloc[:,-1],test_size=0.2)
#建立模型
dtc = DecisionTreeClassifier(
                            criterion = 'gini',
                            random_state = 0,
                            min_samples_split= 10,
                            min_samples_leaf = 4 ,
                            max_features = None,
                            min_impurity_decrease = 0.1,

                            )

dtc.fit(x_train,y_train)#训练模型
result_y = dtc.predict(x_validation)#测试结果

print(confusion_matrix(result_y,y_validation)) # 混淆矩阵

print(accuracy_score(result_y,y_validation)) # 准确率


Test_set = [5.5,3.9,0.8,0.5] # 需要预测该样本所属类别
pre =  dtc.predict(np.array(Test_set).reshape(1,-1)) #预测





from sklearn.ensemble import RandomForestClassifier # 随机森林分类
from sklearn.metrics import confusion_matrix  #混淆矩阵
from sklearn.metrics import accuracy_score  #准确率判断
from sklearn.model_selection import train_test_split  # 拆分数据
import pandas as pd
import numpy as np

DATA_path = r'C:\Users\Administrator\Desktop\iris.csv'
DATA = pd.read_csv(
    filepath_or_buffer = DATA_path
)

Test_set = [5.5,3.9,0.8,0.5] # 需要预测该样本所属类别

x_train, x_validation , y_train, y_validation = train_test_split(DATA.iloc[:,:-1],DATA.iloc[:,-1],test_size=0.2)
#建立模型
rfc = RandomForestClassifier(
                             n_estimators=100,

                             random_state = 0,
                             min_samples_split= 10,
                             min_samples_leaf = 4 ,
                             max_features = None,
                             min_impurity_decrease = 0.1,

                             oob_score = True

                             )
rfc.fit(x_train,y_train)#训练模型
result_y = rfc.predict(x_validation)#测试结果
print(confusion_matrix(result_y,y_validation)) # 混淆矩阵
print(accuracy_score(result_y,y_validation)) # 准确率

print(rfc.oob_score_) # 袋外数据验证分数

pre =  rfc.predict(np.array(Test_set).reshape(1,-1)) #预测


# *****************************summary - classification *********************
# 几个参数的解释
'''

n_estimators=100, # 随机森林 用到 ,树的个数 
random_state = 0, # 随机数种子
min_samples_split= 10, # 拆分时,该节点所含最小样本 : 泛化性和防止过拟合
min_samples_leaf = 4 , # 叶子节点最少为多少个样本 : 泛化性和防止过拟合
max_features = None, # 在划分时考虑多少个属性 : 属性多的时候
min_impurity_decrease = 0.1, # 防止过拟合,设置"不纯度"分裂阈值
oob_score = True # 袋外数据:
当且仅当用在random forest中,用没选中的数据做一个简单验证

'''




#***********************************模型建立与预测--regression  ********************************

# 暂时不实现降维等操作,直接使用原数据
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor #决策树回归
from sklearn.model_selection import train_test_split  # 拆分数据
from sklearn.metrics import mean_squared_error  #回归准确率判断-mse
import pandas as pd
import numpy as np

X,Y = load_boston(return_X_y=True)
x_train, x_validation , y_train, y_validation = \
    train_test_split(X,Y,test_size=0.2)

Test_set = (x_train[1] + x_train[2])/2
# 可以增加数据处理
regressor = DecisionTreeRegressor(

                                  random_state=0,
                                  min_samples_split=10,
                                  min_samples_leaf=4,
                                  max_features=None,
                                  min_impurity_decrease=0.1,

                                  )
regressor.fit(x_train,y_train)#训练模型
result_y = regressor.predict(x_validation)#测试结果
print(mean_squared_error(result_y,y_validation)) # 准确率
pre =  regressor.predict(np.array(Test_set).reshape(1,-1)) #预测




from sklearn.ensemble import RandomForestRegressor # 随机森林回归
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error  #回归准确率判断-mse
from sklearn.model_selection import train_test_split  # 拆分数据
import pandas as pd
import numpy as np

X,Y = load_boston(return_X_y=True)
x_train, x_validation , y_train, y_validation = \
    train_test_split(X,Y,test_size=0.2)
Test_set = (x_train[1] + x_train[2])/2
# 可以增加数据处理
rfr = RandomForestRegressor(
                             n_estimators=200,
                             random_state = 0,
                             min_samples_split= 5,
                             min_samples_leaf = 3 ,
                             max_features = None,
                             min_impurity_decrease = 0.1,
                             oob_score = True
                             )

rfr.fit(x_train,y_train)#训练模型
result_y = rfr.predict(x_validation)#测试结果
print(mean_squared_error(result_y,y_validation)) # 准确率
print(rfr.oob_score_) # 袋外数据验证分数
pre =  rfr.predict(np.array(Test_set).reshape(1,-1)) #预测




