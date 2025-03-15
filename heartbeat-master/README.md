# heartbeat

#### 介绍和原理
由于心脏搏动促使血液的流动,引起皮肤下的血管的容积随心脏呈脉动性变化,入射光的光程也会随之发生改变,以及血液对不同波段的光束的吸收作用不同,从而引起表层皮肤的颜色和形状变化,因此,反射光被摄像头接收到形成彩色视频图像,采集到的视频中的每帧图像在红、绿、蓝三颜色通道的亮度变化包含脉动信息,特别是绿色通道图像信号最能够反映心血管活动中心脏搏动的时间变化及其周期,即彩色视频中含有心率信息。


#### 使用说明

一、直接使用：
1.下载atrial fibrillation压缩包后，解压到文件夹atrial fibrillation
2.使用anaconda3环境，spyder(python=3.5)打开atrial fibrillation 中的heart_rate_readvideo.py，运行
3.运行结果，打印出最可能的心率和几张图，最后一张图代表所选人脸区域中的频率信息。

二、如果要另外测试，需要：
1.先拍摄含有人脸的一段视频（用手机拍摄时把手机横过来拍，一般拍摄5-10秒即可），假设命名为1.avi，将1.avi放入atrial fibrillation文件夹中
2.使用anaconda3环境，spyder(python=3.5)打开atrial fibrillation 中的faceswap.py
3.将第203行，’my_video(1).mp4’改为‘1.avi’，然后运行
4.使用anaconda3环境，spyder (python=3.5)打开atrial fibrillation 中的heart_rate_readvideo.py，将第9行中的’my_video(1).mp4’改为‘1.avi’，将第12行Fs改为1.avi的采样率（通过右键视频查看属性可以知道），运行
关于结果：
如果最后一张图出现有两个波峰，结果是不正确的，原因应该是ICA分解没找到最优解，重新运行heart_rate_readvideo.py，直到只有一个波峰。
前三张图是B G R信号，接下来三张是对B G R进行ICA分解后得到的三个独立信源，接下来六张图中的第1，2，3，4张是ICA分解后的独立信源，第五张是根据相关性所选择的独立信源中的一个信源，第六张是原来BGR中的G信号，第五和第六张应该具有较强的相关性。最后两张图，第一张是滤波之后的心率图，第二张是该心率图对应的快速傅里叶变换在1-1.5Hz的取值

三、关于环境：
安装anaconda3后默认使用的是python3.5将anaconda中的Scripts添加到环境变量PATH中
安装opencv3, scipy, scikit_learn，matplotlib, numpy五个模块，在anaconda navigator中搜索安装很方便。

四、其他注意事项：
用手机拍摄视频时，请把手机横过来拍，否则传到电脑上的视频人脸是旋转了九十度的，将无法识别。

五、参考文献：
  [1]姚丽峰. 基于PPG和彩色视频的非接触式心率测量[D].天津大学,2012.
   [2]Poh M Z, McDuff D J, Picard R W. Non-contact, automated cardiac pulse measurements using video imaging and blind source separation[J]. Optics express, 2010, 18(10): 10762-10774.

