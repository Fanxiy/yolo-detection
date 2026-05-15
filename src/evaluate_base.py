import os
import sys
from ultralytics import YOLO


def main():
    # 定义路径
    model_path = "yolov8n.pt"  # 模型路径
    data_path = "data/data_1.yaml"  # 数据集配置文件路径

    # 加载模型
    print(f"正在加载模型: {model_path}")
    model = YOLO(model_path)

    # 评估模型
    print(f"\n正在使用数据集 {data_path} 评估模型...")
    print("=" * 60)
    
    # 运行验证
    results = model.val(
        data=data_path,
        split="val",
        verbose=True,
        plots=True  # 生成评估图
    )

    # 打印关键指标
    print("\n" + "=" * 60)
    print("评估结果总结")
    print("=" * 60)
    
    # 获取mAP指标
    map50 = results.box.map50  # mAP@0.5
    map = results.box.map  # mAP@0.5:0.95
    map75 = results.box.map75  # mAP@0.75
    
    # 打印结果
    print(f"mAP@0.5 (IoU=0.5):      {map50:.4f}")
    print(f"mAP@0.5:0.95 (mAP50-95): {map:.4f}")
    print(f"mAP@0.75 (IoU=0.75):     {map75:.4f}")
    print("=" * 60)
    print(f"\n详细结果已保存到: {os.path.abspath('runs/detect/base')}")


if __name__ == "__main__":
    main()
