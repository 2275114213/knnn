# day71  kNN

K nearest neighbour

## 0、导引

### 如何进行电影分类

众所周知，电影可以按照题材分类，然而题材本身是如何定义的?由谁来判定某部电影属于哪 个题材?也就是说同一题材的电影具有哪些公共特征?这些都是在进行电影分类时必须要考虑的问 题。没有哪个电影人会说自己制作的电影和以前的某部电影类似，但我们确实知道每部电影在风格 上的确有可能会和同题材的电影相近。那么动作片具有哪些共有特征，使得动作片之间非常类似， 而与爱情片存在着明显的差别呢？动作片中也会存在接吻镜头，爱情片中也会存在打斗场景，我们 不能单纯依靠是否存在打斗或者亲吻来判断影片的类型。但是爱情片中的亲吻镜头更多，动作片中 的打斗场景也更频繁，基于此类场景在某部电影中出现的次数可以用来进行电影分类。

本章介绍第一个机器学习算法：K-近邻算法，它非常有效而且易于掌握。

## 1、k-近邻算法原理

简单地说，K-近邻算法采用测量不同特征值之间的距离方法进行分类。

- 优点：精度高、对异常值不敏感、无数据输入假定。
- 缺点：时间复杂度高、空间复杂度高。
- 适用数据范围：数值型和标称型。

### 工作原理

存在一个样本数据集合，也称作训练样本集，并且样本集中每个数据都存在标签，即我们知道样本集中每一数据 与所属分类的对应关系。输人没有标签的新数据后，将新数据的每个特征与样本集中数据对应的 特征进行比较，然后算法提取样本集中特征最相似数据（最近邻）的分类标签。一般来说，我们 只选择样本数据集中前K个最相似的数据，这就是K-近邻算法中K的出处,通常*K是不大于20的整数。 最后 ，选择K个最相似数据中出现次数最多的分类，作为新数据的分类*。

回到前面电影分类的例子，使用K-近邻算法分类爱情片和动作片。有人曾经统计过很多电影的打斗镜头和接吻镜头，下图显示了6部电影的打斗和接吻次数。假如有一部未看过的电影，如何确定它是爱情片还是动作片呢？我们可以使用K-近邻算法来解决这个问题。



## 2、在scikit-learn库中使用k-近邻算法

- 分类问题：from sklearn.neighbors import KNeighborsClassifier
- 回归问题：from sklearn.neighbors import KNeighborsRegressor

### 0）一个最简单的例子

身高、体重、鞋子尺码数据对应性别

```python
# 导入KNN 分类算法
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
# !!! 样本中，男女比例应该1：1
data = np.array([[175,70,43],[180,75,44],[165,50,38],[163,48,37],[170,68,42],[168,52,40]])
target = np.array(['男','男','女','女','男','女'])
# 声明算法
knn = KNeighborsClassifier(n_neighbors=5)
# 使用算法，进行学习，训练
knn.fit(data,target)
# 使用算法，进行预测数据
data_predict = np.array([[188,90,46],[166,55,38],[169,65,41]])
knn.predict(data_predict)
```

### 1）用于分类

导包，机器学习的算法KNN、数据蓝蝴蝶

```python
# 使用KNN算法，对一种花，进行分类
# 数据源在sklearn中

import sklearn.datasets as datasets
# 使用datasets中的方法，导入数据
# data属性：花萼长度，花萼宽度，花瓣长度，花瓣宽度
# 鸢尾花分三类 ：'target_names': array(['setosa', 'versicolor', 'virginica'], dtype='<U10')}
iris = datasets.load_iris()

data = iris['data']
target = iris['target']
# numpy 将数据打乱顺序
# shuffle 随机打乱顺序，data ，target两个都需要打乱顺序
# 随机种子，每次和每次都不同，所以，随机数的时候，每次和每次都不同
# np.random.shuffle()
# np.random.seed(8)
# np.random.randint(0,150,size = 1)
# 只能使用一次
np.random.seed(11)
np.random.shuffle(data)
np.random.seed(11)
np.random.shuffle(target)
#训练样本
# 150个样本，分成两份，140个（训练数据），10个（预测）

# 获取了140个训练数据
X_train = data[:140]
y_train = target[:140]

# 预测数据
X_test = data[140:]
y_test = target[140:] #真实答案

knn = KNeighborsClassifier(n_neighbors=5)

# 第一步，训练
knn.fit(X_train,y_train)

# 第二步，预测
# 返回自己的“观点”
# 一般情况下，机器学习返回结果 添加：_
y_ = knn.predict(X_test)

print('鸢尾花分类真实情况是：',y_test)

print('鸢尾花机器学习分类情况是： ',y_)

# 通过结果，看到，机器学习，将最后这10未知的数据，全部预测准确

# 计算算法的准确率
score = knn.score(X_test,y_test)
print('算法的准确率： ', score)
```

+ 使用pandas数据类型进行操作

```python
#机器学习的数据可以是numpy也可以是pandas

import pandas as pd
from pandas import Series,DataFrame

#先将训练数据转换为pandans类型数据
X_train_df = DataFrame(X_train,columns=['speal length','speal width','petal length','petal width'])
y_train_s = Series(y_train)

#将测试测试数据转为pandas数据
X_test_df = DataFrame(X_test,columns=['speal length','speal width','petal length','petal width'])
y_test_s = Series(y_test)

knn = KNeighborsClassifier(10)
knn.fit(X_train_df,y_train_s)

y_ = knn.predict(X_test_df)
print(y_test_s.values)
print(y_)
```

+ 训练数字


## 二、KNN手写数字识别

```python
import numpy as np
# bmp 图片后缀
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.neighbors import KNeighborsClassifier

digit = plt.imread('./data/3/3_200.bmp')
# 28 * 28 很小
# 首先将图片放大显示
plt.figure(figsize=(2,2))
plt.imshow(digit,cmap = 'gray')
```

+ 批量获取数据

```python
# 批量获取数据
data = []
target = []

for i in range(10):
    for j in range(1,501):
#         plt.imread('./data/3/3_200.bmp')
#         digit是二维的数组，黑白
        digit = plt.imread('./data/%d/%d_%d.bmp'%(i,i,j))
        data.append(digit)
        
#         目标值
        target.append(i)
# data 和target 5000个数据
len(data)
```

```python
# 将机器学习的数据转换成ndarry，操作起来比较方便
# ndarray 提供了很多方法
data = np.array(data)
target = np.array(target)

print(data.shape,target.shape)
```

```python
#显示正确值及图片，仅供测试
index = np.random.randint(0,5000,size = 1)[0]

print('该索引所对应的目标值： ',target[index])

digit = data[index]
plt.figure(figsize=(2,2))
plt.imshow(digit,cmap = 'gray')
```

+ 打乱数据，生成学习队列

```python
seed = np.random.randint(0,5000,size = 1)[0]

# 指明随机数的种子，打乱顺序
np.random.seed(seed)
np.random.shuffle(data)

# 指明的种子和上面的一样，打乱顺序的规则就和上面一样
np.random.seed(seed)
np.random.shuffle(target)

```

```python
# 验证一下，顺序是否匹配
index = np.random.randint(0,5000,size = 1)[0]

print('该索引所对应的目标值： ',target[index])

digit = data[index]
plt.figure(figsize=(2,2))
plt.imshow(digit,cmap = 'gray')
```

+ 机器学习,分割数据

````python
knn = KNeighborsClassifier(n_neighbors=20)
# 最后保留50个数据作为预测数据集
# 训练数据
X_train,y_train = data[:4950],target[:4950]
# 测试数据
X_test,y_test = data[-50:],target[-50:]

````

+ 因算法只能接受二维数据，故学习和预测的数据都需要转化为二维数据

```python
X_train = X_train.reshape((4950,784))

# 正着数像素784像素点，倒着数-1
X_test = X_test.reshape((50,-1))
```

+ 训练

```python
# 第一步进行训练
knn.fit(X_train,y_train)
```

+ 预测

```python
# 第二步，使用算法，进行预测
y_ = knn.predict(X_test)

print('真实数据：',y_test)
print('预测数据： ',y_)
```

+ 将五十条数据画出来

```python
# 可视化，将50张绘制出来

plt.figure(figsize=(2*5,3*10))

for i in range(50):
    
#     10行5列
#     子视图，在每一个子视图中绘制图片
    subplot = plt.subplot(10,5,i+1)
    
#     最后50张图片在 X_test中
#     !!! 像素点需要reshape成图片形状
    subplot.imshow(X_test[i].reshape((28,28)))
    
#     添加标题，True：0
#               Predict：0
#     y_test ----真实
#     y_  ------预测
    t = y_test[i]
    p = y_[i]
    subplot.set_title('True: %d\nPredict:%d'%(t,p))
```

获取网络上的数字图片进行识别

+ 读取图片转化为灰度

```python
digits = plt.imread('./数字.jpg')
digits = digits.mean(axis = 2)
plt.imshow(digits,cmap = 'gray')
```

+ 将图片中的数字切片切下来

```python
data_pre = digits[175:240,78:143]
plt.imshow(data_pre,cmap = 'gray')
```

+ 将图片转为28*28

```python
import scipy.ndimage as ndimage
data_pre_test = ndimage.zoom(data_pre,zoom = (28/65,28/65))
print(data_pre_test.shape)

plt.figure(figsize=(2,2))
plt.imshow(data_pre_test,cmap = 'gray')
```

+ 机器预测

```python
# 从网络上获取的数据，有时候，因为写数字，和样本不同，误差可能会很大
knn.predict(data_pre_test.reshape((1,-1)))
knn.predict(data_pre_test.reshape((1,784)))
```

## 三、保存模型

```python
# 保存模型
# knn 算法，模型，estimator
# 数学建模，model 算法
from sklearn.externals import joblib
# 保存
joblib.dump(knn,'数字识别.m')
# 提取算法
knn_digits = joblib.load('./数字识别.m')
#使用模型
knn_digits.predict(data_pre_test.reshape((1,-1)))
```

## 四、预测癌症

```python
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from sklearn.neighbors import KNeighborsClassifier
```

+ 读取.csv文件，以\t作为换行符

```python
cancer = pd.read_csv('./data/cancer.csv',sep = '\t')
print(cancer.shape)
cancer
```

+ 将数据进行切片，获取有用数据

```python
# 获取数据
data = cancer.iloc[:,2:]
target = cancer.iloc[:,1]
# target目标值中，M恶性，B良性，没事儿
display(data.head(),target.head())
```

+ 设置邻点个数，将数据分为训练和测试两部分

```python
knn = KNeighborsClassifier(n_neighbors=50)
# 打乱顺序，并且一分为二，训练数据，预测数据
# sklearn 为我们提供了方法
from sklearn.model_selection import train_test_split
# 使用train_test_split进行数据分割
X_train,X_test,y_train,y_test = train_test_split(data,target,test_size = 0.1)
```

+ 训练数据

```python
knn.fit(X_train,y_train)
```

+ 预测数据

```python
#检查预测率
print(knn.score(X_test,y_test))
y_ = knn.predict(X_test)
```

+ 设置交叉表

```python
# y_ 预测值
# y_test 真实值

# 交叉表,可以说明，真实值和预测的哪一些数据不同
pd.crosstab(index = y_,columns=y_test,rownames=['Predict'],colnames=['True'],margins=True)
```

+ 提升准确度

因样本中的数据大小参差不齐，会给预测的准确性带来偏差，故将数据归一化处理后在进行机器学习，可以提高预测的准确性

```python
# ！！！对数据进行处理，清洗
# 归一化
# (num - min)/(max - min)  0 ~ 1
columns = data.columns
columns
```

+ 获取所有的列

```python
columns = data.columns
columns
```

+ 对列的值归一化处理

```python
for col in columns:
    data_min = data[col].min()
    data_max = data[col].max()
    #一次将一整列的数据都处理了
    data[col] = (data[col] - data_min)/(data_max - data_min)
```

+ 将数据分为学习和预测两部分

```python
X_train,X_test,y_train,y_test = train_test_split(data,target,test_size = 0.1)
```

+ 开始学习

```python
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)
# 数据清洗之后，准确率大幅提升，30列属性，单位不同， 归一化数据，准确率很大幅度进行了提升
y_ = knn.predict(X_test)
```

+ 交叉表

```python
# 交叉表

pd.crosstab(index=y_,columns=y_test,rownames=['Predict'],colnames=['True'])
```






















预测男女性别

预测鸢尾花