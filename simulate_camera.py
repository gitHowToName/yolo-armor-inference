import cv2
import time
from ultralytics import YOLO

def main():
    # 直接使用你的实际路径
    video_path = "data/raw_video.mp4"  
    model_path = "models/best.pt" 

    print("[INFO] 正在加载 YOLO 模型...")
    model = YOLO(model_path)

    print(f"[INFO] 正在打开视频源: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    # 异常处理：模拟取流超时或打开失败（作业第4点要求）
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频源 {video_path} 或取流超时！")
        print("[INFO] 正在清理资源...")
        return

    frame_count = 0
    start_time = time.time()
    fps = 0

    try:
        while True:
            ret, frame = cap.read()
            
            # 异常处理：模拟取流超时（视频读完或读取失败）
            if not ret:
                print("[WARNING] 视频读取结束或取流超时。")
                break

            # --- 核心：YOLO 实时推理 ---
            results = model.predict(frame, conf=0.25, imgsz=480, verbose=False)
            
            # 获取绘制了检测框、类别、置信度的画面
            annotated_frame = results[0].plot()

            # --- 进阶要求：计算 FPS ---
            frame_count += 1
            if time.time() - start_time >= 1.0:
                fps = frame_count / (time.time() - start_time)
                frame_count = 0
                start_time = time.time()

            # 在画面左上角显示 FPS
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 显示画面
            cv2.imshow("Armor Detection (Simulated)", annotated_frame)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] 用户手动中断。")
                break

    except KeyboardInterrupt:
        print("\n[INFO] 接收到 Ctrl+C 中断信号。")
    
    except Exception as e:
        print(f"[ERROR] 发生异常: {e}")
        
    finally:
        # --- 作业第4点重点：资源释放 ---
        print("[INFO] 正在释放资源 (模拟归还相机缓冲区/句柄)...")
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] 所有资源已安全释放。")

if __name__ == "__main__":
    main()
