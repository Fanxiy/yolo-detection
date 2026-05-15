from ultralytics import YOLO

# 加载你训练好的最佳模型
model = YOLO(r"models\best.pt")  # 路径写对

# 1. 预测单张图片
# model.predict("test.jpg", save=True)  # 结果自动保存

# 2. 预测视频
model.predict("video/test.mp4", save=True) # 此处不带ID

# 3. 打开摄像头实时检测
# model.predict(0, show=True)