import onnx

model_path = 'models/sentiment_onnx.onnx'

onnx_model = onnx.load(model_path)

# 节点
# for node in onnx_model.graph.node:
#     print(node)

# 输入
print("Inputs：")
for inputs in onnx_model.graph.input:
    print(inputs)

# 输出
# for outputs in onnx_model.graph.output:
#     print(outputs)

# 参数
# for initializer in onnx_model.graph.initializer:
#     print(initializer)

