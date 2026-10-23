import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def get_frames(video_path, n=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()
    return frames

def bench_model(model_path, frames, warmup=5, repeat=10, imgsz=640, conf=0.25):
    model = YOLO(model_path)
    for f in frames[:warmup]:
        model.predict(f, imgsz=imgsz, conf=conf, verbose=False)
    
    times, det_counts = [], []
    for _ in range(repeat):
        for f in frames:
            t0 = time.perf_counter()
            res = model.predict(f, imgsz=imgsz, conf=conf, verbose=False)
            times.append((time.perf_counter() - t0) * 1000)
            det_counts.append(len(res[0].boxes) if res[0].boxes else 0)
            
    size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    return {
        "模型": model_path,
        "大小(MB)": round(size_mb, 2),
        "平均耗时(ms)": round(float(np.mean(times)), 2),
        "中位耗时(ms)": round(float(np.median(times)), 2),
        "平均检测数": round(float(np.mean(det_counts)), 2)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--pt", type=str, default="models/best.pt")
    parser.add_argument("--onnx", type=str, default="models/best.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    frames = get_frames(args.video, n=20)
    print("\n正在测试 原始 .pt 模型...")
    print(bench_model(args.pt, frames, imgsz=args.imgsz, conf=args.conf))
    print("\n正在测试 导出 .onnx 模型...")
    print(bench_model(args.onnx, frames, imgsz=args.imgsz, conf=args.conf))
