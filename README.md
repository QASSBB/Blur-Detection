演算法
目的:
判斷影像是否模糊
方法:
使用局部功率譜斜率，透過輸入的影像轉換成FFT的形式，得到其頻譜影像，當中包含每個頻率的振幅大小(強度)。影像是二維，FFT變換是將空間域變為頻域，對空間域而言顏色變換的程度就是頻率(色彩變化越小頻率越低)。 
步驟:
Input : 輸入影像I(x, y)，x = 0, …, M-1，y = 0, …, N-1，彩色(三通道)
Step1 : 將I(x, y)轉換成灰階(單通道)影像Ig(x, y)
Step2 : 對Ig(x, y)作FFT得到f(u,v)
Step3 : 將f (u,v)的原點從左上角移到中心，得到影像fs(u,v)
Step4 : 傅立葉轉換後的結果是複數的形式，包含了圖像在不同頻率下的振幅和相位信息。
若fs(u,v) = a + jb，則振幅影像 ，相位影像 。
使用np.abs (fs(u,v))可以直接得到fm，np.angle(fs(u,v))可以直接得到fp

Step5 : 對fm取離散平方值，讓影像中模糊與清楚的部分更加明顯  。
Step6 : 將fss(u,v)進行極座標轉換為 。若(u, v)位於二、三象限時θ要再加上180度以保證θ的值在360以內。
Step7 : 將 轉換成一維數列 。Y軸為S(r)，X軸為r，就可以作出功率譜, X軸為頻率Y軸為振幅的頻譜圖，得到每個半徑的振幅的和(S(r))。
Step8: 將50張模糊圖像與50張清楚圖像作為訓練集(70)與測試集(30)，使用SVM模型(Support Vector Machine)，找到一個決策邊界(decision boundary)讓模糊與清楚兩類之間的邊界(margins)最大化，使其可以完美區隔開來，進而利用此邊界去判斷圖像是否為模糊。

程式碼:
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
# 讀取圖片
img = cv.imread("E:/uu187/Pictures/dd/21.jpg")
# step1 : 圖片灰階 
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# step2 : FFT轉換
f = np.fft.fft2(gray)
# step3 : 將轉換後的結果移到中心
fshift = np.fft.fftshift(f)
# step4 : 對fshift取log，將fshift轉成實數陣列
fr = np.log(1 + np.abs(fshift))
# step5 : 取離散平方
height,width= fr.shape
fss = (np.abs(fr)**2)/math.sqrt(width*height)
# step6 :　極座標轉換
theta_array = np.zeros(fss.shape)
r_array = np.zeros(fss.shape)
# 對"fss"每個點進行極座標轉換從[0,0]開始
for v in range(fss.shape[0]): # 高(row)
    for u in range(fss.shape[1]): # 寬(column)
        # r 為將直角坐標轉換為極座標的半徑，先將X、Y的(直角坐標-中心座標)**2再將兩者相加開根號，最後取整數
        r = int(np.sqrt((u-width/2)**2 + (v-height/2)**2))  # // => 除完取整數
        r_array[v,u] = r # 儲存所有座標的半徑
# step7 : 將r_array轉換成一維數列，Y軸為S(r)，X軸為r，就可以作出功率譜, X軸為頻率Y軸為振幅的頻譜圖
r_array = r_array.astype(int) # 把陣列r_array從浮點數轉為整數
# 設一個大小為np.max(r_array)的零陣列，用來儲存半徑 1~最大值(ex:右上角)的所有振幅的值(fss)
sr = [] 
sr_max = int(np.max(r_array)) 
sr = np.zeros(sr_max)
count = np.zeros(sr_max)
# 把r_array每個位置的值(半徑)，都加上fss的值
for v in range(r_array.shape[0]): # 高(row)
    for u in range(r_array.shape[1]): # 寬(column)
            sr[r_array[v,u]:r_array[v,u]+1] += fss[v,u] # 將相同半徑(r_array)的所有振幅(fss)加總至一維陣列sr
            count[r_array[v,u]:r_array[v,u]+1] += 1
        # 將sr取平均(除以對應半徑的振幅數量)
for i in range(len(sr)):
    if count[i] > 0:
        sr[i] /= count[i]
        
plt.plot(sr)
plt.title('Image_Clear')
plt.xlabel('r')
plt.ylabel('s(r)')
 
 
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import svm

# 從CSV文件中讀取數據
df = pd.read_csv('E:/python_program/asr.csv',encoding='UTF-8')
# 提取特徵和標籤
# X = 每一筆資料的 第1個特徵到第400個特徵
X = df.loc[:, '.sr.1':'.sr.400'].values
# y = 每一筆資料的 種類
y = df.loc[:, 'variety'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pca = PCA(n_components=2, iterated_power=1)
# 使用PCA對訓練資料X_train進行轉換，得到二維降維後的數據train_reduced
train_reduced = pca.fit_transform(X_train)
# 建立支援向量機模型，並指定參數C為1，C代表著錯誤項的懲罰因子
linearSvcModel=svm.SVC(kernel='linear', C=1.0)
# 使用訓練資料train_reduced和對應的標籤y_train訓練支援向量機模型
linearSvcModel.fit(train_reduced, y_train)
# 計算分類邊界 
# w = 模型權重
w = linearSvcModel.coef_[0]
a = -w[0] / w[1]
# 分類邊界線範圍(-200,200)
XX = np.linspace(-200, 200)
yy = a * XX - linearSvcModel.intercept_[0] / w[1]
# 線為黑色('c' = cyan 藍綠, 'r' = red 紅)
plt.plot(XX, yy,'k')
plt.scatter(train_reduced[:, 0], train_reduced[:, 1], c=y_train)
plt.legend()
plt.show()
 
