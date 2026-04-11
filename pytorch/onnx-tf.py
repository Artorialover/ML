import torch
import torch.nn as nn
import onnx
import tensorflow as tf
import onnx2tf

# ===================== 1. 先定义你的20层模型结构（必须和训练时一样） =====================
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

# ===================== 2. 加载训练好的.pth模型 =====================
savePath="C:/Users/27132/Desktop/大模型/model/student_score_model.pth"
ONNXPath="C:/Users/27132/Desktop/大模型/model/student_score_model.onnx"
TFLITEPath="C:/Users/27132/Desktop/大模型/model/student_score_model.tflite"
model = StudentScoreModel()
model.load_state_dict(torch.load(savePath))
model.eval()

# ===================== 3. 导出 ONNX 格式 =====================
dummy_input = torch.randn(1, 3)  # 输入：1个样本，3个特征（年龄、数学、英语）
torch.onnx.export(
    model,
    dummy_input,
    ONNXPath,
    opset_version=10,
    input_names=["input"],
    output_names=["output"]
)
print("✅ 已导出 ONNX 模型")

# ===================== 4. ONNX → TensorFlow =====================
# onnx_model = onnx.load(ONNXPath)
# onnx2tf.convert..convert.from_onnx(
#     onnx_model,
#     output_path="C:/Users/27132/Desktop/大模型/model/student_model_tf"  # 输出TF模型
# )
# print("✅ 已导出 TensorFlow 模型")

# ===================== 5. TensorFlow → TFLite =====================
converter = tf.lite.TFLiteConverter.from_saved_model("C:/Users/27132/Desktop/大模型/model/student_model_tf")
tflite_model = converter.convert()

# 保存 TFLite 模型
with open("C:/Users/27132/Desktop/大模型/model/student_model.tflite", "wb") as f:
    f.write(tflite_model)

print("🎉 成功！最终 TFLite 模型已保存为 student_model.tflite")