import argparse
import cv2
import os
from ultralytics import YOLO
import numpy as np


class LineCounter:
    def __init__(self, line_start, line_end):
        self.line_start = np.array(line_start, dtype=np.float32)
        self.line_end = np.array(line_end, dtype=np.float32)
        self.counted_ids = set()
        self.cross_count = 0
        self.previous_positions = {}
    
    def point_position(self, point):
        line_vec = self.line_end - self.line_start
        point_vec = point - self.line_start
        cross = np.cross(line_vec, point_vec)
        return cross
    
    def check_cross(self, track_id, current_point):
        if track_id in self.previous_positions:
            prev_pos = self.previous_positions[track_id]
            prev_side = self.point_position(prev_pos)
            curr_side = self.point_position(current_point)
            
            if prev_side * curr_side < 0:
                if track_id not in self.counted_ids:
                    self.counted_ids.add(track_id)
                    self.cross_count += 1
                    return True
        
        self.previous_positions[track_id] = current_point
        return False


def count_crossing(args):
    model = YOLO(args.model)
    
    # 获取模型的类别名称
    class_names = model.names
    
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.source}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
    
    if args.line_y is None:
        args.line_y = height // 2
    
    line_start = (0, args.line_y)
    line_end = (width, args.line_y)
    
    counter = LineCounter(line_start, line_end)
    
    if args.save_video:
        output_path = os.path.join(args.output_dir, 'line_count_output.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 随机选择视频后半段的一帧进行保存
    import random
    save_frame_num = random.randint(total_frames // 2, total_frames - 1)
    saved_frame = False
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.track(frame, persist=True, conf=args.conf, iou=args.iou, tracker='botsort.yaml')
        
        annotated_frame = frame.copy()
        
        cv2.line(annotated_frame, line_start, line_end, (0, 0, 255), 3)
        
        for result in results:
            if result.boxes and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                
                for box, track_id, cls, conf in zip(boxes, ids, classes, confs):
                    x1, y1, x2, y2 = box
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    crossed = counter.check_cross(track_id, np.array([center_x, center_y]))
                    
                    color = (0, 255, 0) if not crossed else (255, 0, 0)
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.circle(annotated_frame, (center_x, center_y), 5, (255, 0, 255), -1)
                    
                    # 获取类别名称
                    class_name = class_names.get(cls, f'Class_{cls}')
                    # 格式化置信度，保留两位小数
                    conf_str = f'{conf:.2f}'
                    
                    # 组合标签文本：ID + 类别 + 置信度
                    label_text = f'ID: {track_id} | {class_name} | Conf: {conf_str}'
                    
                    # 绘制标签（调整字体大小和位置，避免超出画面）
                    cv2.putText(annotated_frame, label_text, 
                               (int(x1), int(y1) - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.putText(annotated_frame, f'Count: {counter.cross_count}', 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(annotated_frame, f'Line Y: {args.line_y}', 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 保存随机选择的帧
        if frame_count == save_frame_num and not saved_frame:
            original_save_path = os.path.join(args.output_dir, f'line_count_frame_{save_frame_num:04d}_original.jpg')
            annotated_save_path = os.path.join(args.output_dir, f'line_count_frame_{save_frame_num:04d}_annotated.jpg')
            cv2.imwrite(original_save_path, frame)
            cv2.imwrite(annotated_save_path, annotated_frame)
            print(f"\n保存随机帧 (第 {save_frame_num} 帧):")
            print(f"原始帧: {original_save_path}")
            print(f"标注帧: {annotated_save_path}")
            saved_frame = True
        
        if args.save_video:
            out.write(annotated_frame)
        
        if args.show:
            cv2.imshow('Line Counting', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames, Count: {counter.cross_count}")
    
    cap.release()
    if args.save_video:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"\nCounting complete!")
    print(f"Total objects crossed the line: {counter.cross_count}")
    print(f"Counted track IDs: {sorted(counter.counted_ids)}")
    
    if saved_frame:
        return counter.cross_count, counter.counted_ids, save_frame_num
    else:
        return counter.cross_count, counter.counted_ids, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv8 Line Crossing Counter')
    parser.add_argument('--model', type=str, default=r"models\best.pt", help='Trained model path')
    parser.add_argument('--source', type=str, required=True, help='Input video path')
    parser.add_argument('--line_y', type=int, default=None, help='Y coordinate of counting line (default: middle)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='IOU threshold')
    parser.add_argument('--output_dir', type=str, default='video_outputs', help='Output directory')
    parser.add_argument('--save_video', action='store_true', help='Save output video')
    parser.add_argument('--show', action='store_true', help='Show video while processing')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    result = count_crossing(args)
    if len(result) == 3:
        count, counted_ids, save_frame = result
        if save_frame:
            print(f"\n随机帧已保存: 第 {save_frame} 帧")
    else:
        count, counted_ids = result