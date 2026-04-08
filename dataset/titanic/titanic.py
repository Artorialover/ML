import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import model_selection
from sklearn import neighbors
from sklearn import tree

train_data=pd.read_csv('C:/Users/27132/Desktop/大模型/dataset/titanic/train.csv');

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_rows', None)    # 显示所有行
pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', None)       # 不限制宽度
pd.set_option('display.max_colwidth', None)# 不限制列宽

# print(train_data.head())
# 检查空值
# print(train_data.isnull().any().any())
# print(train_data.isnull().sum().sum())

# print(train_data.info())
##特征工程
##数据清洗--填充空数据

###出发港口
# print(train_data['Embarked'].value_counts())
# embarked_ports_name=train_data['Embarked'].value_counts().index
# embarked_ports_count=train_data['Embarked'].value_counts().values

# plt.bar(embarked_ports_name,embarked_ports_count)
# plt.show()

train_data["Embarked"]=train_data['Embarked'].fillna('S')

###船仓
train_data["Cabin"]=train_data['Cabin'].fillna('U')

# print(train_data['Age'].value_counts())

###构建随机森林预测年龄

corr=train_data.corr(numeric_only=True)
# print(corr['Age'])

agePre=train_data[['Age','Pclass','SibSp','Parch']]
agePre=pd.get_dummies(agePre)
# print(agePre.corr()[['Age']])

ageUnknown:pd.DataFrame =agePre[agePre['Age'].isnull()];
ageknown:pd.DataFrame=agePre[agePre['Age'].notnull()]

ageknown_X=ageknown.drop('Age',axis=1)
ageknown_y=ageknown['Age']

ageUnknown_X=ageUnknown.drop('Age',axis=1)

rfr=ensemble.RandomForestRegressor()
rfr.fit(ageknown_X,ageknown_y)
ageUnknown_y=rfr.predict(ageUnknown_X)

train_data.loc[train_data['Age'].isnull(),['Age']]=ageUnknown_y



##相关性系数筛选特征
train_data1=train_data.drop(['Name','Ticket','PassengerId','SibSp','Parch','Age','Cabin'],axis=1)
train_data1=pd.get_dummies(train_data1)
print(train_data1.corr()['Survived'])

train_data=train_data.drop(['Name','Ticket','PassengerId','SibSp','Parch','Age','Cabin'],axis=1)

##多个模型选优分析


train_data=pd.get_dummies(train_data);
train_X=train_data.drop('Survived',axis=1)
train_y=train_data['Survived']


# classcifiers=[]
# classcifiers.append(ensemble.RandomForestClassifier())
# classcifiers.append(neighbors.KNeighborsClassifier())
# classcifiers.append(tree.DecisionTreeClassifier())

# classcifiers_results=[]
# for cl in classcifiers:
#     classcifiers_results.append(model_selection.cross_val_score(cl,train_X,train_y,scoring='accuracy',cv=model_selection.StratifiedKFold(10),n_jobs=-1));

   
# c_means=[]
# c_std=[]

# for cr in classcifiers_results:
#     c_means.append(cr.mean())
#     c_std.append(cr.std())


# print(train_data.info())
# print(c_means)
# print(c_std)


###选择模型






test_data=pd.read_csv('C:/Users/27132/Desktop/大模型/dataset/titanic/test.csv');
test_data1=test_data.drop(['Name','Ticket','PassengerId','SibSp','Parch','Age','Cabin'],axis=1)
test_data1=pd.get_dummies(test_data1)
test_X=test_data1
model=ensemble.RandomForestClassifier()
model.fit(train_X,train_y)
print(f'模型准确率：{model.score(train_X, train_y)}')
test_y=model.predict(test_X)

pdf=pd.DataFrame()
pdf['PassengerId']=test_data['PassengerId']
pdf['Survived']=test_y
pdf.to_csv('C:/Users/27132/Desktop/大模型/dataset/titanic/result.csv',index=False)

# #策略1：删除空值列
# new_train_data=train_data.dropna()
# ##统计取值情况
# print(new_train_data['Pclass'].unique())
# print(new_train_data['Pclass'].value_counts())


# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# fig = plt.figure()
# fig.set(alpha=0.65) # 设置图像透明度
# ax1=fig.add_subplot(131)

# df1=pd.crosstab(new_train_data['Sex'],new_train_data['Survived'])
# # df1.rename()
# df1.rename({0:'未生还',1:'生还'},axis=1,inplace=True)
# df1.rename({'female':'F','male':'M'},inplace=True)
# pct_Sex = df1.div(df1.sum(1).astype(float),axis=0) #归一化
# pct_Sex.plot(kind='bar',stacked=True,title=u'不同性别的生还情况',ax=ax1)
# plt.show()
# print(pct_Sex)
