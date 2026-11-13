import sys
import cv2
import numpy as np
from ultralytics import YOLO

# 导入海康 MVS SDK (确保 MvImport 文件夹在当前目录)
sys.path.append("./MvImport")
from MvCameraControl_class import *

class KalmanFilter3D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 3) # 状态(6): X,Y,Z,Vx,Vy,Vz  测量(3): X,Y,Z
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]
        ], np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]
        ], np.float32)
        
        # 噪声参数（用于平滑）
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1
        self.is_initialized = False

    def update(self, z):
        if not self.is_initialized:
            self.kf.statePost = np.array([[z[0]], [z[1]], [z[2]], [0], [0], [0]], np.float32)
            self.is_initialized = True
            return z
            
        self.kf.predict()
        self.kf.correct(np.array(z, np.float32).reshape(3, 1))
        return [self.kf.statePost[0, 0], self.kf.statePost[1, 0], self.kf.statePost[2, 0]]

def main():
    # 1. 加载标定参数
    try:
        calib = np.load("camera_calib.npz")
        mtx, dist = calib['mtx'], calib['dist']
    except:
        print("未找到 camera_calib.npz，请先运行 calibrate.py")
        return

    # ⚠️⚠️⚠️ 必须修改区域 ⚠️⚠️⚠️
    # 2. 定义装甲板实际物理尺寸 (单位: mm)，请拿尺子量！
    ARMOR_W, ARMOR_H = 224.0,126.0 
    # ⚠️⚠️⚠️ 必须修改区域 ⚠️⚠️⚠️

    # 3D 点顺序假设为：左上，右上，右下，左下（如果算出来的坐标乱飞，稍后可能需要调换这里的顺序）
    obj_pts = np.array([
        [-ARMOR_W/2, -ARMOR_H/2, 0], [ ARMOR_W/2, -ARMOR_H/2, 0],
        [ ARMOR_W/2,  ARMOR_H/2, 0], [-ARMOR_W/2,  ARMOR_H/2, 0]
    ], np.float32)

    print("正在加载 YOLO 模型...")
    model = YOLO("models/best.pt")
    kf = KalmanFilter3D()

    print("正在连接相机...")
    MvCamera.MV_CC_Initialize()
    deviceList = MV_CC_DEVICE_INFO_LIST()
    MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
    if deviceList.nDeviceNum == 0:
        print("未找到相机！")
        return
        
    cam = MvCamera()
    stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    cam.MV_CC_CreateHandle(stDeviceList)
    cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    
    # 设置曝光时间 (使用之前测试好的 5000)
    cam.MV_CC_SetFloatValue("ExposureTime", 40000)
    cam.MV_CC_StartGrabbing()

    stParam = MVCC_INTVALUE()
    cam.MV_CC_GetIntValue("PayloadSize", stParam)
    data_buf = (c_ubyte * stParam.nCurValue)()

    print("开始推理，按 'q' 退出...")
    try:
        while True:
            stFrameInfo = MV_FRAME_OUT_INFO_EX()
            ret = cam.MV_CC_GetOneFrameTimeout(data_buf, stParam.nCurValue, stFrameInfo, 1000)
            
            if ret == 0:
                image_data = np.frombuffer(data_buf, dtype=np.uint8)[:stFrameInfo.nFrameLen]
                frame = image_data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BayerRG2RGB)
                
                # YOLO 推理
                results = model.predict(frame_bgr, conf=0.5, imgsz=640, verbose=False)
                r = results[0]
                annotated_frame = r.plot()

                if r.keypoints is not None and len(r.keypoints.xy) > 0:
                    kpts = r.keypoints.xy[0].cpu().numpy() # shape: (4, 2)
                    
                    # PnP 解算
                    success, rvec, tvec = cv2.solvePnP(obj_pts, kpts, mtx, dist, flags=cv2.SOLVEPNP_IPPE)
                    
                    if success:
                        X, Y, Z = tvec[0][0], tvec[1][0], tvec[2][0]
                        distance = np.linalg.norm(tvec)
                        
                        # 卡尔曼滤波
                        filtered = kf.update([X, Y, Z])
                        
                        # 画面文字绘制
                        cv2.putText(annotated_frame, f"Raw: X{X:.0f} Y{Y:.0f} Z{Z:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(annotated_frame, f"KF:  X{filtered[0]:.0f} Y{filtered[1]:.0f} Z{filtered[2]:.0f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"Dist: {distance:.1f} mm", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                        # 终端打印数据用于写实验记录
                        print(f"Raw: [{X:.1f}, {Y:.1f}, {Z:.1f}] -> KF: [{filtered[0]:.1f}, {filtered[1]:.1f}, {filtered[2]:.1f}]")

                cv2.imshow("Armor PnP & Kalman", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
