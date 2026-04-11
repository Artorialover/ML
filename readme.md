安装环境：
pip install numpy pandas matplotlib scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install torch onnx onnx-tf tensorflow

# 1. 卸载冲突包
pip uninstall -y onnx tf2onnx tensorflow protobuf tensorboard

pip install tf2onnx onnx tensorflow protobuf --upgrade

pip install onnxscript onnx-ir onnx2tf

pip install sng4onnx ai_edge_litert psutil onnx_graphsurgeon tf_keras

git clone https://gitcode.com/gh_mirrors/on/onnx2tflite

onnx2tf -i xxx.onnx -o xxx/TF

pip install tflite-runtime
ai_edge_litert

pip install ai_edge_torch torchvision

线性回归
```
适合预测连续值
缺点是如果数据不成线性关系则不准确

```