# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.decomposition import FastICA

# 保存波形数据到文件 避免每次都处理一遍视频耽误时间
saveSequenceData=False
loadSequenceData=True
# 文件夹路径
basePath='C://Users//wangt//Desktop//heartbeat-master (1)//heartbeat-master//atrial fibrillation//'

if(loadSequenceData):
    file=open(basePath+"sequence_n.txt","r")
    N=int(file.read())
    file.close()
    file=open(basePath+"sequence_b.txt","r")
    for idx in range(N):
        sequence_b = np.zeros(N)
        sequence_b[idx] = float(file.readline())
        print(sequence_b[idx])
    file.close()
    file=open(basePath+"sequence_g.txt","r")
    for idx in range(N):
        sequence_g = np.zeros(N)
        sequence_g[idx] = float(file.readline())
    file.close()
    file=open(basePath+"sequence_r.txt","r")
    for idx in range(N):
        sequence_r = np.zeros(N)
        sequence_r[idx] = float(file.readline())
    file.close()

    for idx in range(N):
        print(sequence_b[idx])
else:
    #from tools.detect import MtcnnDetector
    #import faceswap
    cap=cv2.VideoCapture(basePath+'my_video.mp4')

    n=0
    rate=0
    Fs=30
    N=int(cap.get(7))#根据的对齐的图片数 确定N
    num_zeros=1024-N
    print(N)
    sequence_b=np.zeros(N)
    sequence_g=np.zeros(N)
    sequence_r=np.zeros(N)
    # 加载模型参数，构造检测器
    #mtcnn_detector = MtcnnDetector(min_face_size=24, use_cuda=False) 

    while(n<N):
        
        #ret,frame=cap.read()
        print(basePath+str(n)+'.jpg')
        frame=cv2.imread(basePath+str(n)+'.jpg')

    #使用detectMultiScale进行人脸检测
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
            x=845
            y=400
            w=100
            h=100
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
            key=cv2.waitKey(1)
            if(key==27):
                exit(0)

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
            n+=1
            #print(n)
    cap.release()
    cv2.destroyAllWindows()
    #对三个序列进行fastica分解
    N=n

plt.figure()
plt.subplot(3,1,1)
plt.plot(sequence_b)
plt.subplot(3,1,2)
plt.plot(sequence_g)
plt.subplot(3,1,3)
plt.plot(sequence_r)
plt.show()

print(sequence_b)
print(sequence_b.dtype)

if(saveSequenceData):
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
    

sf=np.zeros(N+num_zeros)
for i in range(N):
    sf[i]=a[i]

transformed=np.fft.fft(sf)
N+=num_zeros
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