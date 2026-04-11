import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class StudentDataset(Dataset):
    def __init__(self):
        super().__init__()
        # 随机生成 1000 个学生数据
        # 输入: [年龄, 数学, 英语]
        torch.manual_seed(42)
        self.x = torch.randn(1000, 3)  # (样本数, 3特征)
        # 输出: 语文成绩（简单线性关系+噪声）
        self.y = 0.3 * self.x[:, 0:1] + 0.4 * self.x[:, 1:2] + 0.3 * self.x[:, 2:3] + 0.1 * torch.randn(1000, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 数据集与加载器
dataset = StudentDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class StudentScoreModel(nn.Module):
    def __init__(self):
        super(StudentScoreModel, self).__init__()
        # 构建 20 层全连接层
        layers = []
        
        # 第1层：输入3维 → 64维
        layers.append(nn.Linear(3, 64))
        layers.append(nn.ReLU())
        
        # 中间 18 层（保持维度64，保证深度20层）
        for _ in range(18):
            layers.append(nn.Linear(64, 64))
            layers.append(nn.ReLU())
            
            
        # 第20层：64维 → 输出1维（语文成绩）
        layers.append(nn.Linear(64, 1))
        
        # 把所有层组合成序列模型
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        # 前向传播
        return self.model(x)


# ====================== 3. 训练配置 ======================
device = torch.device('cpu')
model = StudentScoreModel().to(device)


model = StudentScoreModel()
criterion = nn.MSELoss()  # 回归任务
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 50



# 查看模型结构（确认20层 Linear）
print("模型结构：")
print(model)

# 统计总层数（只算Linear层，确认=20）
linear_count = sum(1 for layer in model.modules() if isinstance(layer, nn.Linear))
print(f"\n全连接层总数：{linear_count} 层")



# ====================== 4. 训练循环 ======================
print("开始训练...")
model.train()
for epoch in range(epochs):
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        # 前向
        pred = model(x)
        loss = criterion(pred, y)

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch+1:2d}/{epochs}]  Loss: {avg_loss:.4f}")
       
# ====================== 5. 模型保存 ======================
savePath="C:/Users/27132/Desktop/大模型/model/student_score_model.pth"
torch.save(model.state_dict(), savePath)
print("\n模型已保存为: student_score_model.pth")

model_load = StudentScoreModel().to(device)
model_load.load_state_dict(torch.load(savePath))
model_load.eval()

# ====================== 7. 单样本预测 ======================
with torch.no_grad():
    # 输入: 年龄, 数学, 英语
    test_data = torch.tensor([[16, 88, 92]], dtype=torch.float32).to(device)
    pred_chinese = model_load(test_data)
    print(f"\n预测语文成绩: {pred_chinese.item():.2f}")
        