import sys
import cv2
import numpy as np
import time
from ultralytics import YOLO

# 引入 MVS SDK
sys.path.append("./MvImport")
from MvCameraControl_class import *

def main():
    # ================= 1. 初始化 YOLO =================
    print("[INFO] 正在加载 YOLO 模型...")
    model = YOLO("models/best.pt")

    # ================= 2. 初始化相机 =================
    print("[INFO] 正在枚举相机设备...")
    MvCamera.MV_CC_Initialize()
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        print("[ERROR] 未找到任何相机设备！")
        return

    print(f"[INFO] 找到 {deviceList.nDeviceNum} 个设备，正在连接第一个...")
    cam = MvCamera()
    stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    cam.MV_CC_CreateHandle(stDeviceList)
    
    # 独占模式打开设备
    # 修改为非独占的控制模式
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Control, 0)
    if ret != 0:
        print("[ERROR] 打开相机失败！")
        return

    try:
        # ================= 3. 设置实验参数 =================
        # 【作业要求】：测试不同曝光和增益，在这里修改数值并记录
        exposure_time = 208015.0  # 曝光时间 (us)
        gain = 0.0             # 增益 (dB)
        
        cam.MV_CC_SetFloatValue("ExposureTime", exposure_time)
        cam.MV_CC_SetFloatValue("Gain", gain)
        
        # 设置像素格式为 BayerRG8（如果你的相机不支持，请改为 PixelType_Gvsp_BGR8_Packed）
        cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_BayerRG8)
        
        # 开始取流
        cam.MV_CC_StartGrabbing()
        print(f"[INFO] 取流开始 (曝光: {exposure_time}us, 增益: {gain}dB)")

        # 获取图像缓冲区大小
        stParam = MVCC_INTVALUE()
        cam.MV_CC_GetIntValue("PayloadSize", stParam)
        data_buf = (c_ubyte * stParam.nCurValue)()

        frame_count = 0
        start_time = time.time()
        fps = 0

        print("[INFO] 按 'q' 键退出画面...")
        while True:
            stFrameInfo = MV_FRAME_OUT_INFO_EX()
            # 获取一帧图像（超时时间 1000ms）
            ret = cam.MV_CC_GetOneFrameTimeout(data_buf, stParam.nCurValue, stFrameInfo, 1000)
            
            if ret == 0:
                # 提取图像数据
                image_data = np.frombuffer(data_buf, dtype=np.uint8)[:stFrameInfo.nFrameLen]
                
                # 根据相机的实际分辨率 reshape
                # 【注意】：这里的 Width 和 Height 最好用 cam.MV_CC_GetIntValue("Width", ...) 动态获取，此处假设为常见分辨率
                frame = image_data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                
                # 格式转换：BayerRG8 -> BGR8
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BayerRG2RGB)
                
                # ================= 4. YOLO 推理 =================
                results = model.predict(frame_bgr, conf=0.25, imgsz=480, verbose=False)
                annotated_frame = results[0].plot()

                # 计算 FPS (进阶要求)
                frame_count += 1
                if time.time() - start_time >= 1.0:
                    fps = frame_count / (time.time() - start_time)
                    frame_count = 0
                    start_time = time.time()

                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Real Camera Armor Detection", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("[WARNING] 取流超时！")

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断程序。")
    
    except Exception as e:
        print(f"[ERROR] 程序异常: {e}")
    
    finally:
        # ================= 5. 资源释放 (作业要求) =================
        print("[INFO] 正在停止取流...")
        cam.MV_CC_StopGrabbing()
        print("[INFO] 正在关闭相机...")
        cam.MV_CC_CloseDevice()
        print("[INFO] 正在销毁句柄...")
        cam.MV_CC_DestroyHandle()
        cv2.destroyAllWindows()
        print("[INFO] 相机资源与内存已安全释放！")

if __name__ == "__main__":
    main()
