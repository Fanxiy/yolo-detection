import argparse
import cv2
import os
from ultralytics import YOLO
import numpy as np

def analyze_occlusion(args):
    model = YOLO(args.model)
    
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.source}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 自动设置帧数为视频总帧数
    if args.num_frames is None or args.num_frames <= 0:
        args.num_frames = total_frames - args.start_frame
        if args.num_frames <= 0:
            args.num_frames = total_frames
    
    start_frame = args.start_frame
    num_frames = args.num_frames
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    output_dir = os.path.join(args.output_dir, 'occlusion_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # ===================== 新建报告文件 =====================
    report_path = os.path.join(output_dir, 'occlusion_report.txt')
    report_file = open(report_path, 'w', encoding='utf-8')
    # =======================================================
    
    frames_data = []
    track_ids_history = []
    
    print(f"Analyzing frames {start_frame} to {start_frame + num_frames - 1}...")
    print(f"Total video frames: {total_frames}")
    print(f"Frames to process: {num_frames}")
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {start_frame + i}")
            break
        
        results = model.track(frame, persist=True, conf=args.conf, iou=args.iou, tracker='botsort.yaml')
        annotated_frame = results[0].plot()
        
        current_ids = []
        if results[0].boxes and results[0].boxes.id is not None:
            current_ids = results[0].boxes.id.cpu().numpy().astype(int).tolist()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
        
        track_ids_history.append(current_ids)
        frames_data.append((frame.copy(), annotated_frame.copy(), current_ids))
        
        # 保存图片
        frame_filename = os.path.join(output_dir, f'frame_{start_frame + i:04d}_annotated.jpg')
        cv2.imwrite(frame_filename, annotated_frame)
        original_filename = os.path.join(output_dir, f'frame_{start_frame + i:04d}_original.jpg')
        cv2.imwrite(original_filename, frame)
        
        print(f"Frame {start_frame + i}: Track IDs = {current_ids}")
    
    cap.release()

    # ===================== 输出报告（同时打印 + 保存） =====================
    def print_and_save(text):
        print(text)
        report_file.write(text + '\n')

    print_and_save("\n=== Occlusion Analysis Report ===")
    print_and_save(f"Analyzed {len(frames_data)} consecutive frames")
    
    all_ids = set()
    for ids in track_ids_history:
        all_ids.update(ids)
    
    print_and_save(f"Total unique track IDs: {sorted(all_ids)}")
    
    print_and_save("\nID continuity analysis:")
    for track_id in sorted(all_ids):
        presence = [track_id in ids for ids in track_ids_history]
        if all(presence):
            print_and_save(f"  ID {track_id}: PRESENT in all frames (tracking maintained)")
        elif any(presence):
            print_and_save(f"  ID {track_id}: INTERMITTENT (possible occlusion or ID switch)")
        else:
            print_and_save(f"  ID {track_id}: ABSENT")
    
    print_and_save("\nChecking for ID jumps:")
    for i in range(1, len(track_ids_history)):
        prev_ids = set(track_ids_history[i-1])
        curr_ids = set(track_ids_history[i])
        new_ids = curr_ids - prev_ids
        lost_ids = prev_ids - curr_ids
        
        if new_ids or lost_ids:
            print_and_save(f"  Frame {start_frame + i}:")
            if new_ids:
                print_and_save(f"    New IDs appeared: {new_ids}")
            if lost_ids:
                print_and_save(f"    IDs disappeared: {lost_ids}")
    
    print_and_save("\nAnalysis complete!")
    print_and_save(f"Annotated frames saved to: {output_dir}")
    print_and_save(f"Report saved to: {report_path}")
    
    # 关闭文件
    report_file.close()
    # ======================================================================
    
    return frames_data, track_ids_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze occlusion and ID jumps in video tracking')
    parser.add_argument('--model', type=str, default=r"models\best.pt", help='Trained model path')
    parser.add_argument('--start_frame', type=int, default=0, help='Start frame number')
    parser.add_argument('--source', type=str, required=True, help='Input video path')
    parser.add_argument('--num_frames', type=int, default=None, help='Number of consecutive frames to analyze')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='IOU threshold')
    parser.add_argument('--output_dir', type=str, default='video_outputs', help='Output directory')
    
    args = parser.parse_args()
    analyze_occlusion(args)