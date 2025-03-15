# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.decomposition import FastICA
import time
import queue
import threading
import os

'''把遇到的问题/bug/待修改的部分写在这里
格式:[行号](语句)问题
为了防止更改时行号变化 应把语句复制粘贴

[200](sum_1+=xb[i,j]) 在第一帧(0,0)位置会报warning:RuntimeWarning: overflow encountered in scalar add sum_2+=xg[i,j] 原因暂不清楚 但似乎不影响使用
[]()关于ICA算法 在plt的最后一张图表的最后一张图为fft的分析最终结果 似乎有的时候最后一张图直接是空的/是单调的没有峰值 这时候输出的最终结果直接是最小值60或最大值(最大值似乎与输入的视频秒数有关) 当这种情况发生时似乎应该认为ICA失败了
[]()关于摄像头 时间不能过短 1秒时ICA直接报错 5秒时plt最后一张图表直接是空的 --这个似乎也是ICA的问题 一旦输入过短就会不行
'''
print('123')
# 是否使用电脑摄像头作为源信息 注意 优先级比loadSequenceData低
useCamera=True
# 若使用摄像头 采集视频秒数 与 摄像头路径(若cameraPath=0 则使用默认摄像头)
videoCaptureTime=5
camraPath=0

# 保存波形数据到文件 避免每次都处理一遍视频耽误时间
saveSequenceData=False
loadSequenceData=False
# 文件夹路径
basePath='C://Users//wangt//Desktop//heartbeat-master (1)//heartbeat-master//atrial fibrillation//'
# 视频帧率 会影响下面算法
Fs=30

def waitKey(): # 用于cv2.imshow()之后 自带按esc退出功能
    key=cv2.waitKey(1)
    if(key==27):
        exit(0)

class delaylessVideoCapture: # 读取摄像头帧
    def __init__(self, path=0): # 电脑默认摄像头path = 0
        self.cap = cv2.VideoCapture(path)
        self.q = queue.Queue() # 摄像头读出的帧存放在队列中
        self.lock = threading.Lock()
        self.running = True  # 线程开关 指示摄像头是否录制
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True # 设置为主线程结束时直接结束摄像头线程
        self.t.start()

    def _reader(self): # 保存摄像头录制的图片到self.q中
        while self.running:
            ret,frameRead = self.cap.read()
            cv2.imshow("camera",frameRead)
            waitKey()
            if not ret:
                print("Cam error!")
                exit(0)
            self.q.put(frameRead)

    def read(self): # 从保存的图片队列(self.q)中取出帧
        if self.running == False and self.q.empty(): # 处理结束返回state 0
            state = 0
            ret = 0
        elif self.running == True and self.q.empty(): # 摄像头仍在录制但暂存的帧的队列self.q暂时为空 返回state1
            state = 1
            ret = 0
        else:
            state = 2 # 正常读取 返回state2
            ret = self.q.get()
        return state,ret

    def stop(self): # 关闭摄像头读取
        self.running = False
        self.t.join()
        self.cap.release()
    
    def capset(self,height,weight,fps): # 摄像头参数设置
        if not self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,weight)
            self.cap.set(cv2.CAP_PROP_FPS,fps)

if(loadSequenceData): # 加载保存的波形数组
    file=open(basePath+"sequence_n.txt","r")
    N=int(file.read())
    file.close()

    sequence_b = np.zeros(N)
    sequence_g = np.zeros(N)
    sequence_r = np.zeros(N)

    file=open(basePath+"sequence_b.txt","r")
    for idx in range(N):
        sequence_b[idx] = float(file.readline())
    file.close()

    file=open(basePath+"sequence_g.txt","r")
    for idx in range(N):
        sequence_g[idx] = float(file.readline())
    file.close()

    file=open(basePath+"sequence_r.txt","r")
    for idx in range(N):
        sequence_r[idx] = float(file.readline())
    file.close()

else:
    imageList=[] # 用于存放图片的队列

    if useCamera: # 采用从摄像头录制的方式
        cap=delaylessVideoCapture(camraPath)
        videoStartTime = time.time()
        while True:
            timeSpentVideo = time.time() - videoStartTime # 计时停止录制
            if timeSpentVideo >= videoCaptureTime:
                cap.stop()
            state,frame = cap.read()
            if state==1: # 摄像头仍在录制但暂存的帧的队列self.q暂时为空
                continue
            elif state==2: # 正常读取
                imageList.append(frame)
            else: # 读取结束
                break
            
    elif not useCamera: # 采用最原本的读取原视频方式

        #from tools.detect import MtcnnDetector
        #import faceswap
        n=0
        N=372
        num_zeros=1024-N
        # 加载模型参数，构造检测器
        #mtcnn_detector = MtcnnDetector(min_face_size=24, use_cuda=False) 

        while(n<N):
            
            #ret,frame=cap.read()
            print(basePath+str(n)+'.jpg')
            frame=cv2.imread(basePath+str(n)+'.jpg')
            imageList.append(frame)
            n+=1

    N=len(imageList)
    n=0
    sequence_b=np.zeros(N)
    sequence_g=np.zeros(N)
    sequence_r=np.zeros(N)
    while(n<N):

        ''' ----------人脸识别+取平均值部分----------'''
        #使用detectMultiScale进行人脸检测
        frame=imageList[n]
        image=frame

        gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(basePath+"haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor = 1.15,
                minNeighbors = 5,
            minSize = (3,3)
            )
        #使用mtcnn进行人脸检测
        #    image=frame
        #    #opencv所得图像为BGR，将其转换为RGB才能在dlib中使用
        #    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #    # 检测得到faces以及特征点
        #    faces, landmarks = mtcnn_detector.detect_face(RGB_image)
            #print(landmarks.shape)
        #    num_faces=faces[0,4]
        #    if num_faces==0:
        #        print("Sorry, there were no faces found")
        #        N-=1#某一帧检测不到人脸，将N减1
        #        continue

        #显示人脸
        for (x,y,w,h) in faces:
            print(x,y,w,h)
            # x=845
            # y=400
            # w=100
            # h=100
            # x=int(x+w*0.65)
            # y=int(y+h*0.5)
            # w=int(w*0.2)
            #h=int(h*0.2)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.imshow("capture",image)
            #分离RGB分量
            img=image[y:y+h,x:x+w]
            xb,xg,xr=cv2.split(img)
            # cv2.imshow("capture",img)
            waitKey()

            #取均值
            sum_1=0
            sum_2=0
            sum_3=0
            for i in range(xb.shape[0]):
                for j in range(xb.shape[1]):
                    sum_1+=xb[i,j]
                    sum_2+=xg[i,j]
                    sum_3+=xr[i,j]
            sequence_b[n]=sum_1/(xb.shape[0]*xb.shape[1])
            sequence_g[n]=sum_2/(xg.shape[0]*xg.shape[1])
            sequence_r[n]=sum_3/(xr.shape[0]*xr.shape[1])
            #print(n)
        n+=1
    cv2.destroyAllWindows()

plt.figure()
plt.subplot(3,1,1)
plt.plot(sequence_b)
plt.subplot(3,1,2)
plt.plot(sequence_g)
plt.subplot(3,1,3)
plt.plot(sequence_r)
plt.show()

if(saveSequenceData): # 保存波形数组到硬盘
    file=open(basePath+"sequence_n.txt","w")
    file.write(str(N))
    file.close()
    file=open(basePath+"sequence_b.txt","w")
    for idx in range(N):
        file.write(str(sequence_b[idx]))
        file.write('\n')
    file.close()
    file=open(basePath+"sequence_g.txt","w")
    for idx in range(N):
        file.write(str(sequence_g[idx]))
        file.write('\n')
    file.close()
    file=open(basePath+"sequence_r.txt","w")
    for idx in range(N):
        file.write(str(sequence_r[idx]))
        file.write('\n')
    file.close()

''' ----------ICA部分----------'''
S=np.c_[sequence_b,sequence_g,sequence_r]
print(S.shape)
for i in range(N):
    S[i,:]-=np.mean(S,axis=0)
S /= S.std(axis=0)

A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix

X = np.dot(S, A.T)  # Generate observations
# Compute ICA
ica = FastICA(n_components=3)
S1= ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix
assert np.allclose(X, np.dot(S1, A_.T) + ica.mean_)
a1=S1[:,0]
a2=S1[:,1]
a3=S1[:,2]
plt.figure()
plt.title('results of fastICA')
plt.subplot(3,1,1)
plt.plot(a1)
plt.subplot(3,1,2)
plt.plot(a2)
plt.subplot(3,1,3)
plt.plot(a3)
plt.show()

plt.figure()
plt.subplot(4,2,1)
plt.plot(S1)
plt.subplot(4,2,2)
plt.plot(a1)
plt.subplot(4,2,3)
plt.plot(a2)
plt.subplot(4,2,4)
plt.plot(a3)

corr_a1=(np.corrcoef(a1,sequence_g)[0,1])
corr_a2=(np.corrcoef(a2,sequence_g)[0,1])
corr_a3=(np.corrcoef(a3,sequence_g)[0,1])
#print(corr_a1)
#print(corr_a2)
#print(corr_a3)
if (corr_a1>=corr_a2)&(corr_a1>=corr_a3): a=a1
if (corr_a2>=corr_a1)&(corr_a2>=corr_a3): a=a2
if (corr_a3>=corr_a2)&(corr_a3>=corr_a1): a=a3
plt.subplot(4,2,5)
plt.plot(a)
plt.subplot(4,2,6)
plt.plot(sequence_g)
#滑动滤波器
b=np.zeros(N)
b[0]=a[0]

b[N-1]=a[N-1]
for i in range(1,N-2):
    b[i]=(a[i-1]+a[i]+a[i+1])/3

a=b
#加一个巴特沃斯滤波器    
#mm,nn=signal.butter(3,[0.1,0.5],'bandpass')
#采样频率为15赫兹，最大可测频率为7.5赫兹，脉搏的频率为1到4赫兹，0.75/7.5=0.1,3.75/7.5=0.5
#a=signal.filtfilt(mm,nn,a)
    
   
plt.figure()
plt.subplot(2,1,1)
plt.plot(a)
    
''' ----------fft部分----------'''
num_zeros=1024-N
sf=np.zeros(N+num_zeros)
for i in range(N):
    sf[i]=a[i]
transformed=np.fft.fft(sf)
plt.subplot(2,1,2)
#采样频率为Fs
trans_slice=abs(transformed)[int(1*N/Fs):int(1.5*N/Fs)]
trans_slice[0]/=2 
plt.plot(abs(trans_slice))
peaks=signal.argrelextrema(trans_slice,np.greater,axis=0,order=5)

plt.plot(peaks[0],trans_slice[peaks[0]],'*')    
plt.show()

peaks=np.array(peaks,dtype=float)
#print("the Fs is",Fs)
#print("the N is",N)
print("the most likely heart rate is",int((np.argmax(trans_slice)*Fs/N+1)*60))
#for i in range(peaks.shape[1]):
#    print("the argrelextrema is",((peaks[0,i]*Fs/N+1)*60))