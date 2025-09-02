# coding=utf-8
import os
import platform
from cv2 import waitKey
from HCNetSDK import *
from PlayCtrl import *
import numpy as np
import time
import cv2
import threading
import hk_sdk
from datetime import datetime
import cv2
import time
from ultralytics import YOLO
import math


class HKCam_mul(object):
    def __init__(self, camIPs, usernames, passwords, devport=8000, recorder=False):
        # 登录的设备信息
        self.DEV_IPs = [create_string_buffer(camIP.encode()) for camIP in camIPs]
        self.DEV_PORT = devport  # 所有相机摄像头端口必须要一致
        self.DEV_USER_NAMEs = [create_string_buffer(username.encode()) for username in usernames]
        self.DEV_PASSWORDs = [create_string_buffer(password.encode()) for password in passwords]
        self.WINDOWS_FLAG = False if platform.system() != "Windows" else True
        self.funcRealDataCallBack_V30 = None
        self.last_stamp = None  # 上次时间戳
        self.FuncDecCB = DECCBFUNWIN(self.DecCBFun)  # 公用解码回调函数
        self.lock = threading.RLock()  # 解码回调时需要枷锁否则会出现通道混乱
        self.recent_imgs = {}  # 用于保存每一路视频的时间戳和rgb图像
        # 加载库,先加载依赖库
        if self.WINDOWS_FLAG:
            os.chdir(r'./lib/win')
            self.Objdll = ctypes.CDLL(r'./HCNetSDK.dll')  # 加载网络库
            self.Playctrldll = ctypes.CDLL(r'./PlayCtrl.dll')  # 加载播放库
        else:
            os.chdir(r'./lib/linux')
            self.Objdll = cdll.LoadLibrary(r'./libhcnetsdk.so')
            self.Playctrldll = cdll.LoadLibrary(r'./libPlayCtrl.so')
        # 设置组件库和SSL库加载路径
        self.SetSDKInitCfg()
        # 初始化DLL
        self.Objdll.NET_DVR_Init()
        # 启用SDK写日志
        self.Objdll.NET_DVR_SetLogToFile(3, bytes('./SdkLog_Python/', encoding="utf-8"), False)
        os.chdir(r'../../')
        self.lUserIds = []  # 信号源id
        self.lRealPlayHandles = []  # 信号源句柄
        self.PlayCtrl_Ports = []  # 播放句柄
        self.get_preview_info()
        self.load_cameras()  # 登录设备
        self.funcRealDataCallBack_V30 = REALDATACALLBACK(self.RealDataCallBack_V30)

        # print(self.lUserIds)

        # 获取画面
        for idex, userid in enumerate(self.lUserIds):
            PlayCtrl_Port = c_long(-1)  # 播放句柄
            if not self.Playctrldll.PlayM4_GetPort(byref(PlayCtrl_Port)):
                print(u'获取播放库句柄失败')
            else:
                print('获取播放库句柄成功', PlayCtrl_Port)
                self.PlayCtrl_Ports.append(PlayCtrl_Port)
            for channel in range(1, 3):  # 遍历通道1和通道2
                self.preview_info.lChannel = channel  # 设置通道号
                lRealPlayHandle = self.Objdll.NET_DVR_RealPlay_V40(userid, byref(self.preview_info), None, idex)
                if not lRealPlayHandle:
                    print('lRealPlayHandle 加载失败')
                else:
                    print('lRealPlayHandle', lRealPlayHandle)
                if recorder:  # 视频录像
                    string_buf = f'test{userid}_ch{channel}.mp4'
                    if not self.Objdll.NET_DVR_SaveRealData(lRealPlayHandle, create_string_buffer(string_buf.encode())):
                        print(f'通道{channel}录像失败')
                self.lRealPlayHandles.append(lRealPlayHandle)
        print(self.PlayCtrl_Ports)
        for idd, item in enumerate(self.lRealPlayHandles):
            self.Objdll.NET_DVR_SetRealDataCallBack(item, self.funcRealDataCallBack_V30, idd)
        time.sleep(2)

    def read(self, ):
        return self.recent_imgs

    def Get_Temperature(self, IR_img):
        # 获取温度
        # 测试获取指定点的温度，1280 720 分别是原始图片长宽
        print("IR_img: ", IR_img)
        result, temperature = hk_sdk.get_temperature(IR_img[0], IR_img[1], IR_img[2], IR_img[3], self.lUserIds[1],
                                                     channel_no=2)
        # result, temperature = hk_sdk.get_temperature(640, 360, 1280, 720, self.lUserIds[1], channel_no=2)

        if result:
            print("获取的温度是", temperature)
            return temperature
        else:
            print("获取温度失败 ", temperature, "错误码是: ", hk_sdk.get_last_error())
            return False
        time.sleep(0.01)

    def RealDataCallBack_V30(self, lPlayHandle, dwDataType, pBuffer, dwBufSize, pUser):
        idx = pUser if pUser else 0
        if dwDataType == NET_DVR_SYSHEAD:
            self.Playctrldll.PlayM4_SetStreamOpenMode(self.PlayCtrl_Ports[idx], 0)
            if self.Playctrldll.PlayM4_OpenStream(self.PlayCtrl_Ports[idx], pBuffer, dwBufSize, 1024 * 1024):
                self.Playctrldll.PlayM4_SetDecCallBackExMend(self.PlayCtrl_Ports[idx], self.FuncDecCB, None, 0, None)
                if self.Playctrldll.PlayM4_Play(self.PlayCtrl_Ports[idx], None):
                    print(u'播放库播放成功', self.PlayCtrl_Ports[idx], lPlayHandle)
                else:
                    print(u'播放库播放失败')
            else:
                print(u'播放库打开流失败')
        elif dwDataType == NET_DVR_STREAMDATA:
            # 检查 idx 是否在有效范围内
            if idx < len(self.PlayCtrl_Ports):
                self.Playctrldll.PlayM4_InputData(self.PlayCtrl_Ports[idx], pBuffer, dwBufSize)
            else:
                pass
        else:
            print(u'其他数据,长度:', dwBufSize)

    def load_cameras(self, ):
        device_info = NET_DVR_DEVICEINFO_V30()
        for dev_ip, dev_name, password in zip(self.DEV_IPs, self.DEV_USER_NAMEs, self.DEV_PASSWORDs):
            lUserId = self.Objdll.NET_DVR_Login_V30(dev_ip, self.DEV_PORT, dev_name, password, byref(device_info))
            if lUserId >= 0:
                print(f'摄像头[{dev_ip.raw.decode()}]登录成功!!')
            self.lUserIds.append(lUserId)

    def get_preview_info(self, ):
        self.preview_info = NET_DVR_PREVIEWINFO()
        self.preview_info.hPlayWnd = 0
        self.preview_info.dwStreamType = 0  # 主码流
        self.preview_info.dwLinkMode = 0  # TCP
        self.preview_info.bBlocked = 1  # 阻塞取流

    def SetSDKInitCfg(self, ):
        if self.WINDOWS_FLAG:
            strPath = os.getcwd().encode('gbk')
            sdk_ComPath = NET_DVR_LOCAL_SDK_PATH()
            sdk_ComPath.sPath = strPath
            self.Objdll.NET_DVR_SetSDKInitCfg(2, byref(sdk_ComPath))
            self.Objdll.NET_DVR_SetSDKInitCfg(3, create_string_buffer(strPath + b'\libcrypto-1_1-x64.dll'))
            self.Objdll.NET_DVR_SetSDKInitCfg(4, create_string_buffer(strPath + b'\libssl-1_1-x64.dll'))
        else:
            strPath = os.getcwd().encode('utf-8')
            sdk_ComPath = NET_DVR_LOCAL_SDK_PATH()
            sdk_ComPath.sPath = strPath
            self.Objdll.NET_DVR_SetSDKInitCfg(2, byref(sdk_ComPath))
            self.Objdll.NET_DVR_SetSDKInitCfg(3, create_string_buffer(strPath + b'/libcrypto.so.1.1'))
            self.Objdll.NET_DVR_SetSDKInitCfg(4, create_string_buffer(strPath + b'/libssl.so.1.1'))

    def DecCBFun(self, nPort, pBuf, nSize, pFrameInfo, nUser, nReserved2):
        self.lock.acquire()
        if pFrameInfo.contents.nType == 3:
            nWidth = pFrameInfo.contents.nWidth
            nHeight = pFrameInfo.contents.nHeight
            dwFrameNum = pFrameInfo.contents.dwFrameNum
            nStamp = pFrameInfo.contents.nStamp
            if self.recent_imgs.get(nPort):
                if self.recent_imgs.get(nPort)[0] != nStamp:
                    YUV = np.frombuffer(pBuf[:nSize], dtype=np.uint8)
                    YUV = np.reshape(YUV, [nHeight + nHeight // 2, nWidth])
                    img_rgb = cv2.cvtColor(YUV, cv2.COLOR_YUV2BGR_YV12)
                    self.recent_imgs[nPort] = (nStamp, img_rgb)
            else:
                YUV = np.frombuffer(pBuf[:nSize], dtype=np.uint8)
                YUV = np.reshape(YUV, [nHeight + nHeight // 2, nWidth])
                img_rgb = cv2.cvtColor(YUV, cv2.COLOR_YUV2BGR_YV12)
                self.recent_imgs[nPort] = (nStamp, img_rgb)
        self.lock.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def release(self):
        for item in self.lRealPlayHandles:
            self.Objdll.NET_DVR_StopRealPlay(item)
        for item in self.PlayCtrl_Ports:
            if item.value > -1:
                self.Playctrldll.PlayM4_Stop(item)
                self.Playctrldll.PlayM4_CloseStream(item)
                self.Playctrldll.PlayM4_FreePort(item)
                PlayCtrl_Port = c_long(-1)
        for item in self.lUserIds:
            self.Objdll.NET_DVR_Logout(item)
        self.Objdll.NET_DVR_Cleanup()
        print('释放资源结束')


def calculate_D(H, theta_c, h, fy, d_p):
    """
    计算俯角下的水平距离
    """
    # 计算arctan部分β
    numerator = (h / 2) - d_p
    β = math.atan(numerator / fy)
    # 计算第一个tan部分
    tan_first = math.tan(theta_c + β)
    # 计算最终的D值
    D = H * (tan_first)
    return D


if __name__ == "__main__":
    camIPS = ['192.168.1.173', '192.168.1.173']
    usernames = ['admin'] * 2
    passwords = ['qwezxc135'] * 2

    hkcam_muls = HKCam_mul(camIPS, usernames, passwords)
    prev_time = time.perf_counter()  # 使用更精确的时间计量方法

    frame_count = 0  # 用于计算每秒的帧数
    fps_text = "FPS: 0"  # 初始帧率

    # Load the YOLO model
    model = YOLO(
        '/home/zyj/zyj/HikSdkPython/python_hkcam/1.素材及数据集/yolo_384_Indoor_dataset_v2/weights_train_1classes/yolov11_11n_pt_384_Indoor_v2_300E_4B_lr0-lrf-0.01_/weights/best.pt')

    H = 1.632  # 相机高度，米
    theta_c = math.radians(82.2)  # 相机俯角θc，转换为弧度
    h = 720  # 图片像素高度，y坐标
    fy_rgb = 950  # y方向像素焦距
    fy_ir = 1230  # y方向像素焦距

    # Create a window for combined display
    cv2.namedWindow("Dual Camera View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dual Camera View", 1280, 720)  # Set window size

    while True:
        curr_time = time.perf_counter()
        imgs = hkcam_muls.read()

        # 计算帧率
        frame_count += 1
        frame_time = curr_time - prev_time
        if frame_time >= 1.0:
            fps = frame_count / frame_time
            frame_count = 0
            prev_time = curr_time
            fps_text = f"FPS: {fps:.0f}"

        rgb_img = None
        ir_img = None

        for key, (stamp, img) in list(imgs.items()):  # 复制为列表再遍历
            if img is None:
                print(f"通道 {key} 图像读取失败！")
                continue

            height, width, channels = img.shape
            print(f"Image:{key}： {channels}， {width}x{height}")

            # 定义消失点（vanishing point）
            vanishing_offset = 50  # 消失点距离底部的高度（可调整）
            vanishing_point = (width // 2, height + vanishing_offset)
            # 绘制消失点（红色点）
            cv2.circle(img, vanishing_point, 5, (0, 0, 255), -1)

            # 处理通道0 (RGB)
            if key == 0:
                # 运行YOLOv11推理
                results = model(img, conf=0.65, iou=0.5)
                for result in results:
                    boxes = result.boxes
                    if len(boxes.cls) > 0:
                        classes = boxes.cls.cpu().numpy()
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()

                        for i, (cls, box, conf) in enumerate(zip(classes, xyxy, confs)):
                            if cls in [0]:  # 处理类别
                                x1, y1, x2, y2 = map(int, box)
                                # 计算检测框下边中点
                                bottom_mid_x = (x1 + x2) // 2
                                bottom_mid_y = y2
                                bottom_mid_point = (bottom_mid_x, bottom_mid_y)

                                # 计算距离
                                d_p = bottom_mid_y
                                D = calculate_D(H, theta_c, h, fy_rgb, d_p)
                                actual_distance = D

                                # 绘制检测框（蓝色）
                                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                # 绘制下边中点（绿色）
                                cv2.circle(img, bottom_mid_point, 5, (0, 255, 0), -1)
                                # 绘制测距线（从消失点到检测框下边中点）
                                cv2.line(img, vanishing_point, bottom_mid_point, (0, 255, 255), 2)
                                # 显示距离信息
                                distance_text = f"{actual_distance:.2f}m"
                                cv2.putText(img, distance_text, (bottom_mid_x - 50, bottom_mid_y - 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                rgb_img = img.copy()
                cv2.putText(rgb_img, "RGB Camera", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 处理通道1 (IR)
            if key == 1:
                # 运行YOLOv11推理
                results = model(img, conf=0.45, iou=0.5)
                for result in results:
                    boxes = result.boxes
                    if len(boxes.cls) > 0:
                        classes = boxes.cls.cpu().numpy()
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()

                        for i, (cls, box, conf) in enumerate(zip(classes, xyxy, confs)):
                            if cls in [0]:  # 处理类别
                                x1, y1, x2, y2 = map(int, box)
                                # 计算检测框下边中点
                                bottom_mid_x = (x1 + x2) // 2
                                bottom_mid_y = y2
                                bottom_mid_point = (bottom_mid_x, bottom_mid_y)

                                # 计算距离
                                d_p = bottom_mid_y
                                D = calculate_D(H, theta_c, h, fy_ir, d_p)
                                actual_distance = D

                                # 绘制检测框（蓝色）
                                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                # 绘制下边中点（绿色）
                                cv2.circle(img, bottom_mid_point, 5, (0, 255, 0), -1)
                                # 绘制测距线（从消失点到检测框下边中点）
                                cv2.line(img, vanishing_point, bottom_mid_point, (0, 255, 255), 2)
                                # 显示距离信息
                                distance_text = f"{actual_distance:.2f}m"
                                cv2.putText(img, distance_text, (bottom_mid_x - 50, bottom_mid_y - 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                ir_img = img.copy()
                cv2.putText(ir_img, "IR Camera", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Combine the two images side by side
        if rgb_img is not None and ir_img is not None:
            # Resize both images to the same dimensions if needed
            if rgb_img.shape != ir_img.shape:
                ir_img = cv2.resize(ir_img, (rgb_img.shape[1], rgb_img.shape[0]))

            combined_img = np.hstack((rgb_img, ir_img))

            # Add FPS text to the combined image
            cv2.putText(combined_img, fps_text, (20, combined_img.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Display the combined image
            cv2.imshow("Dual Camera View", combined_img)
        elif rgb_img is not None:
            cv2.imshow("Dual Camera View", rgb_img)
        elif ir_img is not None:
            cv2.imshow("Dual Camera View", ir_img)

        # 等待键盘事件，1毫秒响应，确保流畅显示
        kkk = cv2.waitKey(1)
        if kkk == ord('q'):
            break

    # 释放资源
    hkcam_muls.release()
    cv2.destroyAllWindows()