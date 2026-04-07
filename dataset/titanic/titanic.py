import pandas as pd
import matplotlib.pyplot as plt

train_data=pd.read_csv('C:/Users/27132/Desktop/大模型/dataset/titanic/train.csv');

print(train_data.head())
# 检查空值
# print(train_data.isnull().any().any())
# print(train_data.isnull().sum().sum())


#策略1：删除空值列
new_train_data=train_data.dropna()
##统计取值情况
print(new_train_data['Pclass'].unique())
print(new_train_data['Pclass'].value_counts())


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure()
fig.set(alpha=0.65) # 设置图像透明度
ax1=fig.add_subplot(131)

df1=pd.crosstab(new_train_data['Sex'],new_train_data['Survived'])
# df1.rename()
df1.rename({0:'未生还',1:'生还'},axis=1,inplace=True)
df1.rename({'female':'F','male':'M'},inplace=True)
pct_Sex = df1.div(df1.sum(1).astype(float),axis=0) #归一化
pct_Sex.plot(kind='bar',stacked=True,title=u'不同性别的生还情况',ax=ax1)
plt.show()
print(pct_Sex)
