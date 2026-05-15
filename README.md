# 场景目标检测与视频多目标跟踪

基于YOLOv8的道路车辆目标检测与多目标跟踪项目，包含模型训练、视频跟踪、遮挡分析和越线计数功能。

## 项目结构

```
project/
├── data/                    # 数据集目录
│   ├── train/              # 训练集（包含images/和labels/）
│   └── valid/              # 验证集（包含images/和labels/）
├── src/                     # 源代码目录
│   ├── prepare_data.py     # 数据准备脚本
│   ├── train.py            # 训练脚本
│   ├── track.py            # 视频跟踪脚本
│   ├── occlusionIDjump.py  # 遮挡分析脚本
│   └── line_count.py       # 越线计数脚本
├── models/                  # 训练好的模型目录
│   ├── best.pt             # 最佳模型
├── video/                   # 测试视频目录
│   └── test.mp4
├── video_outputs/           # 视频结果目录
│   ├── occlusion_analysis/ # 遮挡分析帧
│   └── line_count_output.mp4
├── pic/                     # wandb结果图片
│   ├── metrics_mAP50(B).png
│   ├── metrics_mAP50-95(B).png
│   ├── metrics_precision(B).png
│   ├── metrics_recall(B).png
│   ├── train_box_loss.png
│   ├── train_cls_loss.png
│   ├── train_dfl_loss.png
│   ├── train_total_loss.png
│   ├── val_box_loss.png
│   ├── val_cls_loss.png
│   ├── val_dfl_loss.png
│   └── val_total_loss.png
├── runs/                    # 训练结果目录
├── latex/                   # LaTeX报告
├── data.yaml               # 数据集配置
├── requirements.txt        # 依赖包
└── README.md
```

## 环境配置

### 1. 创建虚拟环境（推荐）

```bash
conda create -n yolo python=3.9
conda activate yolo
```

### 2. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 验证安装

```bash
python -c "import torch; import ultralytics; print('OK')"
```

## 数据准备

### 1. 数据集格式

目录结构：
```
data/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
```

### 2. 修改类别配置

编辑`data.yaml`文件，修改类别数量和名称。

## 模型训练

### 1. 基本训练

```bash
python src/train.py
```

### 2. 完整参数示例

```bash
python src/train.py \
    --model yolov8n.pt \
    --data data.yaml \
    --epochs 100 \
    --batch 64 \
    --imgsz 640 \
    --lr 0.01 \
    --optimizer AdamW \
    --device 0 \
    --project runs/detect \
    --name vehicle_detection \
    --wandb \
    --wandb_project vehicle-detection
```

### 3. 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --model | yolov8n.pt | 预训练模型或yaml配置 |
| --data | data.yaml | 数据集配置文件 |
| --epochs | 100 | 训练轮数 |
| --batch | 64 | 批次大小 |
| --imgsz | 640 | 输入图像尺寸 |
| --lr | 0.01 | 初始学习率 |
| --optimizer | AdamW | 优化器 (SGD/Adam/AdamW/RMSProp) |
| --device | 0 | 设备 (0/GPU, cpu) |
| --project | runs/train | 项目目录 |
| --name | train | 实验名称 |
| --wandb | False | 启用wandb日志 |

### 4. 选择模型规模

- `yolov8n.pt` -  Nano (最快，精度较低)
- `yolov8s.pt` -  Small (推荐)
- `yolov8m.pt` -  Medium
- `yolov8l.pt` -  Large
- `yolov8x.pt` -  Extra Large (最精确，速度最慢)

### 5. 训练可视化

启用wandb:
```bash
wandb login
python src/train.py --wandb --wandb_project vehicle-detection
```

### 6. 获取训练好的模型

训练完成后，最佳模型保存在：
```
runs/detect/runs/train/[experiment_name]/weights/best.pt
```

将其复制到models目录。

## 视频多目标跟踪

### 1. 准备测试视频

将测试视频放入`video/`目录。

### 2. 运行跟踪

```bash
python src/track.py --source video/test.mp4 --save_video
```

### 3. 跟踪参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --model | models/best.pt | 训练好的模型路径 |
| --source | - | 输入视频路径 (必填) |
| --conf | 0.5 | 置信度阈值 |
| --iou | 0.5 | IOU阈值 |
| --output_dir | video_outputs | 输出目录 |
| --save_video | False | 保存输出视频 |
| --save_frames | False | 保存单帧图像 |
| --show | False | 实时显示 |

### 4. 示例

```bash
python src/track.py \
    --model models/best.pt \
    --source video/test.mp4 \
    --conf 0.4 \
    --save_video \
    --show
```

## 遮挡与ID跳变分析

### 1. 选择分析片段

在视频中找到包含遮挡或密集交汇的片段，记下起始帧号。

### 2. 运行分析

```bash
python src/occlusionIDjump.py --source video/test.mp4 --start_frame 0 --num_frames 100
```

### 3. 分析参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --model | models/best.pt | 训练好的模型路径 |
| --source | - | 输入视频路径 (必填) |
| --start_frame | 0 | 起始帧号 |
| --num_frames | None | 分析连续帧数 |
| --conf | 0.5 | 置信度阈值 |
| --output_dir | video_outputs | 输出目录 |

### 4. 分析输出

- 保存标注的帧图像到`video_outputs/occlusion_analysis/`
- 在终端输出ID连续性分析
- 检测ID跳变和目标丢失
- 生成分析报告

## 越线计数

### 1. 运行计数

```bash
python src/line_count.py --source video/test.mp4 --save_video
```

### 2. 计数参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --model | models/best.pt | 训练好的模型路径 |
| --source | - | 输入视频路径 (必填) |
| --line_y | None | 计数线的Y坐标 |
| --conf | 0.5 | 置信度阈值 |
| --output_dir | video_outputs | 输出目录 |
| --save_video | False | 保存输出视频 |
| --show | False | 实时显示 |

### 3. 工作原理

- 跟踪每个目标的中心点
- 检测中心点穿过指定的水平线
- 每个ID只计数一次
- 使用叉积判断点在线的哪一侧

## 实验报告

实验报告LaTeX源文件位于`latex/report.tex`，包含以下内容：
- 模型结构和数据集介绍
- 详细的实验设置（训练/验证划分、batch size、学习率等）
- wandb训练过程可视化截图及分析
- 遮挡与ID跳变分析
- 越线计数功能介绍
- 实验总结与展望

## 模型下载

训练好的模型权重保存在`models/best.pt`，可直接使用。
