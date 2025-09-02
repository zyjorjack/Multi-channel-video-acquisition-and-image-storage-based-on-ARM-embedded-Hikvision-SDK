# coding=utf-8
from HCNetSDK import *
from PlayCtrl import *

import os
import platform
from cv2 import waitKey
import numpy as np
import time
import cv2
import threading
import shutil


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
        # 如果 lRealPlayHandle 无效（即连接已断开），则不返回数据
        if not all(handle != -1 for handle in self.lRealPlayHandles):
            # print("摄像头连接断开")
            return {}  # 返回空字典，表示没有新的图像数据
        return self.recent_imgs

    def RealDataCallBack_V30(self, lPlayHandle, dwDataType, pBuffer, dwBufSize, pUser):
        idx = pUser if pUser else 0
        
        # 检查索引是否超出范围
        if idx >= len(self.PlayCtrl_Ports):
            # print(f"Invalid index {idx}. The list of ports has only {len(self.PlayCtrl_Ports)} elements.")
            return  # 退出函数，避免越界访问
        
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
            self.Playctrldll.PlayM4_InputData(self.PlayCtrl_Ports[idx], pBuffer, dwBufSize)
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
        """释放资源时正确清理线程和回调函数"""
        # 停止实时播放
        for item in self.lRealPlayHandles:
            self.Objdll.NET_DVR_StopRealPlay(item)
        # 停止并释放播放库端口
        for item in self.PlayCtrl_Ports:
            if item.value > -1:
                self.Playctrldll.PlayM4_Stop(item)
                self.Playctrldll.PlayM4_CloseStream(item)
                self.Playctrldll.PlayM4_FreePort(item)
                PlayCtrl_Port = c_long(-1)

        # # 增加--注销回调函数，防止持续占用资源
        # for item in self.lRealPlayHandles:
        #     self.Objdll.NET_DVR_SetRealDataCallBack(item, None, None)

        # 登出并释放相机资源
        for item in self.lUserIds:
            self.Objdll.NET_DVR_Logout(item)
        # 清理SDK
        self.Objdll.NET_DVR_Cleanup()
        print('释放资源结束')




import os
import sys
import time
import cv2
import shutil
import threading
import subprocess
from datetime import datetime


if __name__ == "__main__":
    # 相机配置
    camIPS = ['192.168.1.128'] * 2
    usernames = ['admin'] * 2
    passwords = ['EA2379650'] * 2

    # 系统参数配置
    save_interval = 30  # 图片帧保存间隔(秒)
    MAX_IMAGES_PER_FOLDER = 150  # 每分钟文件夹下最多允许的图片数量
    MAX_DD_FOLDERS = 60  # 最多保存多少天的文件夹(按天清理)
    save_dir = '/media/cat/6T/1.RGB_IR_Saved/saved_images'  # 根存储目录

    # 初始化相机对象
    hkcam_muls = HKCam_mul(camIPS, usernames, passwords)
    last_save_time = time.time()  # 初始化最后保存时间

    # 创建根目录(如果不存在)
    os.makedirs(save_dir, exist_ok=True)

    # 用于检测摄像头网络连接的函数
    def ping_host(host):
        """Ping给定的主机，检查网络连接是否正常"""
        response = subprocess.run(['ping', '-c', '1', host], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if response.returncode == 0:
            # print(f"摄像头[{host}]网络连接正常。")
            return True
        else:
            # print(f"摄像头[{host}]网络连接失败。")
            return False

    # 用多线程来执行网络检测
    def check_network_connection(cam_ips):
        """使用多线程检查每个摄像头的网络连接"""
        threads = []
        results = {}

        # 定义回调函数来收集结果
        def check_ip(ip):
            results[ip] = ping_host(ip)

        # 创建并启动线程
        for ip in cam_ips:
            thread = threading.Thread(target=check_ip, args=(ip,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 检查是否有摄像头连接失败
        for ip, success in results.items():
            if not success:
                print(f"摄像头[{ip}]无法连接，程序将退出。")
                sys.exit()
                
    def get_current_minute_folder():
        """获取当前分钟级文件夹路径，并根据MAX_IMAGES_PER_FOLDER自动创建新文件夹"""
        now = datetime.now()
        
        # 天级文件夹路径 (格式: YYYY_MM_DD)
        day_folder = os.path.join(save_dir, now.strftime("%Y_%m_%d"))
        os.makedirs(day_folder, exist_ok=True)
        
        # 小时级文件夹路径 (格式: YYYY_MM_DD_HH)
        hour_folder = os.path.join(day_folder, now.strftime("%Y_%m_%d_%H"))
        os.makedirs(hour_folder, exist_ok=True)
        
        # 获取当前分钟基础名称 (格式: YYYY_MM_DD_HH_MM)
        minute_base = now.strftime("%Y_%m_%d_%H_%M")
        
        # 查找同分钟前缀的所有文件夹
        existing_folders = []
        for f in os.listdir(hour_folder):
            if f.startswith(minute_base):
                folder_path = os.path.join(hour_folder, f)
                if os.path.isdir(folder_path):
                    # 获取文件夹中的图片数量
                    img_count = len([name for name in os.listdir(folder_path) 
                                   if name.endswith('.jpg')])
                    existing_folders.append((folder_path, img_count))
        
        # 如果有未满的文件夹，使用最新的一个
        for folder, count in sorted(existing_folders, key=lambda x: x[0], reverse=True):
            if count < MAX_IMAGES_PER_FOLDER:
                return folder
        
        # 否则创建新的分钟级文件夹
        # 如果有多个同分钟前缀的文件夹，添加序号 (格式: YYYY_MM_DD_HH_MM_序号)
        if existing_folders:
            new_folder = f"{minute_base}_{len(existing_folders)}"
        else:
            new_folder = minute_base
            
        minute_folder = os.path.join(hour_folder, new_folder)
        os.makedirs(minute_folder)
        return minute_folder

    def cleanup_old_days():
        """清理超过MAX_DD_FOLDERS天数的旧文件夹"""
        day_folders = []
        for f in os.listdir(save_dir):
            full_path = os.path.join(save_dir, f)
            if os.path.isdir(full_path) and len(f.split('_')) == 3:  # 检查YYYY_MM_DD格式
                try:
                    # 验证文件夹名是否为有效日期
                    datetime.strptime(f, "%Y_%m_%d")
                    ctime = os.path.getctime(full_path)
                    day_folders.append((full_path, ctime))
                except ValueError:
                    continue
        
        # 按创建时间排序(最早的在前)
        day_folders.sort(key=lambda x: x[1])
        
        # 删除超过限制的最旧文件夹
        while len(day_folders) > MAX_DD_FOLDERS:
            oldest_folder = day_folders.pop(0)[0]
            shutil.rmtree(oldest_folder)
            print(f"已删除最旧日期文件夹: {oldest_folder}")

    while True:
        # 网络检查，断网则退出
        check_network_connection(camIPS)

        # 读取相机图像
        imgs = hkcam_muls.read()
        
        # 显示图像(可选)
        # for key in imgs.keys():
        #     stamp, img = imgs[key]
        #     cv2.imshow(f'{key}', img)

        # 定时保存逻辑
        current_time = time.time()
        if current_time - last_save_time >= save_interval:
            # 获取当前时间字符串(精确到毫秒)
            now = datetime.now()
            time_str = now.strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
            
            # 获取当前分钟级文件夹
            minute_folder = get_current_minute_folder()

                # 读取图像
            imgs = hkcam_muls.read()
            if not imgs:  # 如果返回的是空数据（即摄像头断开）
                print("摄像头连接断开，停止保存图像,退出程序。")
                sys.exit()  # 退出程序，或者 break 退出保存循环或执行其他适当的错误处理
            
            # 保存所有相机图像
            for key in imgs.keys():
                stamp, img = imgs[key]
                # 文件名格式: YYYY_MM_DD_HH_MM_SS_毫秒_通道号.jpg
                filename = f"{minute_folder}/{time_str}_{key}.jpg"
                cv2.imwrite(filename, img)
                # print(f"图像已保存: {filename}")
            
            # 每小时整点检查清理旧文件夹
            if now.minute == 0 and now.second == 0:
                cleanup_old_days()
            
            last_save_time = current_time  # 更新最后保存时间

        # 退出条件
        if cv2.waitKey(1) == ord('q'):
            break

    # 释放资源
    hkcam_muls.release()
    cv2.destroyAllWindows()