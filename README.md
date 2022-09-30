## Faceswape换脸模型 –Tensorflow2 
---

## 目录  
1. [所需环境 Environment](#所需环境) 
2. [注意事项 Attention](#注意事项) 
3. [效果展示 Effect](#效果展示) 
4. [训练步骤 Train](#训练步骤) 

## 所需环境  
1. Python3.7
2. Tensorflow-gpu>=2.5.0 
3. Numpy==1.19.5
4. Pillow==8.2.0
5. Opencv-contrib-python==4.5.1.48
6. CUDA 11.0+
7. Cudnn 8.0.4+

## 注意事项  
1. 新增大尺寸(128×128)的faceswap网络  
2．模型输入数据标准化方式、输出激活方式更改  
3. 加入正则化操作，降低过拟合影响  
4. 数据、训练参数等均位于config.py  

## 效果展示  
![image](https://github.com/JJASMINE22/Faceswape/blob/main/sample/sample1.jpg)  
![image](https://github.com/JJASMINE22/Faceswape/blob/main/sample/sample2.jpg)  

## 训练步骤  
运行train.py  

