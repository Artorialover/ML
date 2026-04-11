

from ai_edge_litert.interpreter import Interpreter

# ai_edge_torch.convert()

model=Interpreter('C:/Users/27132/Desktop/大模型/model/TF/student_score_model_float32.tflite')
model.allocate_tensors()

input_details = model.get_input_details()
output_details = model.get_output_details()

# model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
# model.eval()

# # 2. 准备示例输入
# sample_input = (torch.randn(1, 3, 224, 224),)

# # 3. 转换为 LiteRT 模型
# edge_model = ai_edge_torch.convert(model, sample_input)

# # 4. 导出为 .tflite 文件
# edge_model.export("resnet18.tflite")

