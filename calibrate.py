import cv2
import numpy as np
import glob

# ⚠️⚠️⚠️ 必须修改区域 ⚠️⚠️⚠️
pattern_size = (6,9) # 棋盘格内角点数量 (横向交点, 纵向交点)，根据你的实际标定板修改
square_size = 26.0    # 每个格子的实际物理边长(mm)，拿尺子量！
image_path = 'calib_images/*.bmp' # 因为你的图是 .bmp 格式
# ⚠️⚠️⚠️ 必须修改区域 ⚠️⚠️⚠️

objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

objpoints = [] 
imgpoints = [] 

images = glob.glob(image_path)
print(f"找到 {len(images)} 张图片，开始提取角点...")

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        print(f"[{fname.split('/')[-1]}] 找到角点 ✓")
    else:
        print(f"[{fname.split('/')[-1]}] ❌ 未找到角点")

if len(objpoints) > 5:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("\n=========================================")
    print(f"相机内参矩阵 (K):\n{mtx}")
    print(f"畸变系数 (D):\n{dist}")
    print("=========================================")
    
    np.savez("camera_calib.npz", mtx=mtx, dist=dist)
    print("✅ 标定成功！参数已保存为 camera_calib.npz")
else:
    print("❌ 有效图片太少（少于 6 张），标定失败，请重新拍摄！")
