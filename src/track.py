import argparse
import cv2
import os
from ultralytics import YOLO
import numpy as np

# 定义21类别的颜色（用于不同类别的可视化）
CLASS_COLORS = [
    (255, 0, 0),      # ambulance - 红
    (0, 255, 0),      # army vehicle - 绿
    (0, 0, 255),      # auto rickshaw - 蓝
    (255, 255, 0),    # bicycle - 黄
    (255, 0, 255),    # bus - 紫
    (0, 255, 255),    # car - 青
    (128, 0, 0),      # garbagevan - 深红
    (0, 128, 0),      # human hauler - 深绿
    (0, 0, 128),      # minibus - 深蓝
    (128, 128, 0),    # minivan - 橄榄
    (128, 0, 128),    # motorbike - 紫灰
    (0, 128, 128),    # pickup - 青灰
    (255, 128, 0),    # policecar - 橙
    (255, 0, 128),    # rickshaw - 粉红
    (128, 255, 0),    # scooter - 草绿
    (0, 128, 255),    # suv - 天蓝
    (255, 128, 128),  # taxi - 浅红
    (128, 255, 128),  # three wheelers -CNG- - 浅绿
    (128, 128, 255),  # truck - 浅蓝
    (192, 192, 192),  # van - 灰
    (255, 192, 128)   # wheelbarrow - 浅橙
]

CLASS_NAMES = [
    'ambulance', 'army vehicle', 'auto rickshaw', 'bicycle', 'bus', 'car', 
    'garbagevan', 'human hauler', 'minibus', 'minivan', 'motorbike', 
    'pickup', 'policecar', 'rickshaw', 'scooter', 'suv', 'taxi', 
    'three wheelers -CNG-', 'truck', 'van', 'wheelbarrow'
]

# 为每个跟踪ID生成唯一颜色
def get_track_color(track_id):
    track_id = int(track_id)
    r = (track_id * 13) % 255
    g = (track_id * 51) % 255
    b = (track_id * 89) % 255
    return (int(r), int(g), int(b))

def track_video(args):
    # 加载模型 - 如果没有训练好的模型，使用预训练模型
    if os.path.exists(args.model):
        model = YOLO(args.model)
        print(f"Loaded custom model from {args.model}")
    else:
        model = YOLO('yolov8n.pt')
        print("Using pretrained YOLOv8n model")
    
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.source}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
    
    if args.save_video:
        output_path = os.path.join(args.output_dir, 'tracked_output.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    tracking_history = {}
    all_track_ids = set()
    
    print("Starting tracking...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 复制帧用于绘制
        annotated_frame = frame.copy()
        
        # 使用YOLO跟踪 - 明确指定跟踪器
        results = model.track(
            annotated_frame, 
            persist=True, 
            conf=args.conf, 
            iou=args.iou,
            tracker='botsort.yaml'  # 使用botsort跟踪器
        )
        
        # 处理跟踪结果
        if results[0].boxes is not None:
            boxes = results[0].boxes
            if boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().astype(int)
                xyxy = boxes.xyxy.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                
                # 遍历每个检测到的目标
                for box, track_id, cls_id, conf in zip(xyxy, track_ids, class_ids, confs):
                    x1, y1, x2, y2 = box
                    track_id = int(track_id)
                    all_track_ids.add(track_id)
                    
                    # 获取颜色
                    track_color = get_track_color(track_id)
                    
                    # 绘制边界框
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), track_color, 2)
                    
                    # 计算中心点
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # 绘制中心点
                    cv2.circle(annotated_frame, (center_x, center_y), 4, track_color, -1)
                    
                    # 获取类别名称
                    cls_name = CLASS_NAMES[cls_id % len(CLASS_NAMES)] if cls_id < len(CLASS_NAMES) else str(cls_id)
                    
                    # 构建标签文本：ID + 类别 + 置信度
                    label = f"ID:{track_id} {cls_name} {conf:.2f}"
                    
                    # 计算标签位置
                    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    
                    # 绘制标签背景
                    cv2.rectangle(
                        annotated_frame, 
                        (int(x1), int(y1) - label_size[1] - 10), 
                        (int(x1) + label_size[0], int(y1)), 
                        track_color, 
                        -1
                    )
                    
                    # 绘制标签文本
                    cv2.putText(
                        annotated_frame, 
                        label, 
                        (int(x1), int(y1) - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 255, 255), 
                        2
                    )
                    
                    # 保存跟踪历史
                    if track_id not in tracking_history:
                        tracking_history[track_id] = []
                    tracking_history[track_id].append((center_x, center_y, frame_count))
        
        # 在左上角添加统计信息
        stats_text = [
            f"Frame: {frame_count}/{total_frames}",
            f"Total Tracks: {len(all_track_ids)}",
            f"Active Tracks: {len(tracking_history)}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(
                annotated_frame, 
                text, 
                (10, 30 + i * 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
        
        if args.save_frames:
            frame_filename = os.path.join(args.output_dir, f'frame_{frame_count:06d}.jpg')
            cv2.imwrite(frame_filename, annotated_frame)
        
        if args.save_video:
            out.write(annotated_frame)
        
        if args.show:
            cv2.imshow('Tracking - Press Q to quit', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames, Active tracks: {len(tracking_history)}")
    
    cap.release()
    if args.save_video:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"\nTracking complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total unique track IDs: {len(all_track_ids)}")
    print(f"Track IDs: {sorted(all_track_ids)}")
    
    # 打印每个跟踪ID的出现帧数
    print("\nTrack statistics:")
    for track_id in sorted(tracking_history.keys()):
        print(f"  ID {track_id}: {len(tracking_history[track_id])} frames")
    
    return tracking_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv8 Multi-Object Tracking with Stable IDs')
    parser.add_argument('--model', type=str, default=r"models\best.pt", help='Trained model path')
    parser.add_argument('--source', type=str, required=True, help='Input video path')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (lower for more detections)')
    parser.add_argument('--iou', type=float, default=0.5, help='IOU threshold')
    parser.add_argument('--output_dir', type=str, default='video_outputs', help='Output directory')
    parser.add_argument('--save_video', action='store_true', help='Save output video')
    parser.add_argument('--save_frames', action='store_true', help='Save individual frames')
    parser.add_argument('--show', action='store_true', help='Show video while processing')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("YOLOv8 Multi-Object Tracking")
    print("="*60)
    print(f"Input: {args.source}")
    print(f"Confidence: {args.conf}")
    print(f"IOU: {args.iou}")
    print("="*60)
    
    tracking_history = track_video(args)
