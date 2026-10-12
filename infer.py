import cv2
from ultralytics import YOLO

# 1. 加载权重
model = YOLO("best.pt")

# 2. 读取视频
video_path = "raw_video.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频属性
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 3. 保存视频设置 (MP4格式，使用mp4v编码)
out = cv2.VideoWriter("result_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print("开始推理... (CPU推理可能较慢，请耐心等待)")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 4. 对视频逐帧推理
    results = model(frame)
    
    # 5. 可视化：绘制检测框、类别、置信度以及4个关键点
    # plot() 方法会自动画好框、标签、置信度，以及关键点（如果模型有keypoints）
    annotated_frame = results[0].plot()
    
    # 写入输出视频
    out.write(annotated_frame)
    cv2.imshow("YOLO Inference", annotated_frame)
    
    # 按 q 键退出预览
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("推理完成，结果已保存为 result_video.mp4")
