import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    model = YOLO(args.model)
    out = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=True)
    print(f"导出成功，模型保存在: {out}")

if __name__ == "__main__":
    main()
