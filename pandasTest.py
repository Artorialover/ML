import pandas as pd

# 造一个数据
data = {
    '姓名': ['小明', '小红', '小刚'],
    '年龄': [18, 19, 20],
    '成绩': [90, 85, 88]
}
print(pd.__version__)
df=pd.DataFrame(data)
# df.to_csv('C:/Users/27132/Desktop/大模型/pandas.csv',index=False,encoding='utf-8-sig')
rdata=pd.read_csv('C:/Users/27132/Desktop/大模型/pandas.csv',encoding='utf-8-sig')
print(rdata)
print(rdata[['姓名','年龄']])
print(rdata.iloc[:2])
print(rdata[(rdata['年龄']>=19) & (rdata['成绩']>=86)])
# rdata=rdata.fillna(rdata.mean(numeric_only=True))
rdata.sort_values(by='成绩')
rdata.groupby('成绩').sum()
rdata.rename(columns={'成绩':'成绩1'},inplace=True)
print(rdata.value_counts("年龄"))
print(rdata)