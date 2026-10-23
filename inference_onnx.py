import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best.onnx")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/armor_onnx.mp4")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.source)
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print("开始 ONNX 视频推理...")
    while True:
        ok, frame = cap.read()
        if not ok: break
        
        results = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
        annotated_frame = results[0].plot()
        writer.write(annotated_frame)

    cap.release()
    writer.release()
    print(f"ONNX 推理结果保存至: {out_path}")

if __name__ == "__main__":
    main()
