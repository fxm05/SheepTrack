from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")
# Train the model

train_results = model.train(

    # data="/home/datasets/mycoco.yaml",  # path to dataset YAML

    data="/home/data/datasets/data.yaml",

    epochs=10,  # number of training epochs

    imgsz=640,  # training image size

    device=[0],  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu

)

# Evaluate model performance on the validation setmetrics = model.val()

# Perform object detection on an imageresults = model("path/to image. jpg")results [0] . show()

# Export the model to ONNX format
results = model.val()

print("\n--- 测试集评估结果 ---")
test_metrics = model.val(
    split='test',        # 指定评估测试集（关键参数）
    data="/home/data/datasets/data.yaml",  # 再次指定数据集配置文件（确保读取test路径）
    imgsz=640            # 测试图片尺寸需与训练/验证一致
)

# 5. 输出测试集关键指标（根据需求选择输出）
print(f"测试集 mAP@0.5: {test_metrics.box.map:.4f}")          # 测试集mAP（IoU=0.5）
print(f"测试集 mAP@0.5:0.95: {test_metrics.box.map50-95:.4f}")  # 测试集mAP（IoU=0.5~0.95）
print(f"测试集 Precision: {test_metrics.box.p:.4f}")          # 测试集精确率
print(f"测试集 Recall: {test_metrics.box.r:.4f}")             # 测试集召回率
print(f"测试集 F1-Score: {test_metrics.box.f1:.4f}")          # 测试集F1分数

print(results.box.map)
path = model.export(format="onnx")  # return path to exported model
