import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

X, y=make_regression(n_samples=20,n_features=2,noise=0.1)
# X=[[10,0.5],[20,0.9],[30,1.6]];
# y=[20,30,40]
scaler=MinMaxScaler(feature_range=(10,40))
X=scaler.fit_transform(X)

model=linear_model.LinearRegression()
model.fit(X,y)
current=[40,2.0]
pred_temp=model.predict([current])
model.coef_
print(f"预测温度：{pred_temp[0]:.2f} ℃")
loss=mean_squared_error(model.predict(X),y)


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
x_axis=[row[0] for row in X]
plt.scatter(x_axis, y, label="真实数据")
yy=model.predict(X)
plt.plot(x_axis, yy, "r-", label="预测线")
x_axis=[current[0]]
plt.scatter(x_axis, pred_temp, color="green", s=100, label="新预测点")
plt.xlabel("电流 (特征)")
plt.ylabel("温度 (标签)")
plt.legend()
plt.show()