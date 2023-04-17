# Blur-Detection演算法
## 目的:
### 判斷影像是否模糊
## 方法:
### 使用局部功率譜斜率，透過輸入的影像轉換成FFT的形式，得到其頻譜影像，當中包含每個頻率的振幅大小(強度)。影像是二維，FFT變換是將空間域變為頻域，對空間域而言顏色變換的程度就是頻率(色彩變化越小頻率越低)。 
## 步驟:
### Input : 輸入影像I(x, y)，x = 0, …, M-1，y = 0, …, N-1，彩色(三通道)
### Step1 : 將I(x, y)轉換成灰階(單通道)影像Ig(x, y)
### Step2 : 對Ig(x, y)作FFT得到f(u,v)
### Step3 : 將f (u,v)的原點從左上角移到中心，得到影像fs(u,v)
### Step4 : 傅立葉轉換後的結果是複數的形式，包含了圖像在不同頻率下的振幅和相位信息。若fs(u,v) = a + jb，則振幅影像，相位影像。使用np.abs (fs(u,v))可以直接得到fm，np.angle(fs(u,v))可以直接得到fp

### Step5 : 對fm取離散平方值，讓影像中模糊與清楚的部分更加明顯  。
### Step6 : 將fss(u,v)進行極座標轉換為 。若(u, v)位於二、三象限時θ要再加上180度以保證θ的值在360以內。
### Step7 : 將 轉換成一維數列 。Y軸為S(r)，X軸為r，就可以作出功率譜, X軸為頻率Y軸為振幅的頻譜圖，得到每個半徑的振幅的和(S(r))。
### Step8: 將50張模糊圖像與50張清楚圖像作為訓練集(70)與測試集(30)，使用SVM模型(Support Vector Machine)，找到一個決策邊界(decision boundary)讓模糊與清楚兩類之間的邊界(margins)最大化，使其可以完美區隔開來，進而利用此邊界去判斷圖像是否為模糊。
```
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
```
## ![image](https://user-images.githubusercontent.com/79627981/232484721-100a6cf4-dc04-4c61-a408-2c95f85e1814.png)
## ![image](https://user-images.githubusercontent.com/79627981/232484756-7fc50fac-f291-4c69-93fa-3deffd8c5d08.png)
# 
## 有無PCA差異
### 無PCA 
### ![image](https://user-images.githubusercontent.com/79627981/232486691-88f8d215-f5b0-40e9-88c9-b86b438d631f.png) ![image](https://user-images.githubusercontent.com/79627981/232486817-ec09bded-d67e-4073-a2cc-e497ee1b3f39.png) ![image](https://user-images.githubusercontent.com/79627981/232486833-a3cca5ac-e7ce-4c14-a4b0-cf5720a77e02.png)

### 有PCA ![image](https://user-images.githubusercontent.com/79627981/232486904-78cdc8ba-2488-4638-b561-31a72ef6196a.png) ![image](https://user-images.githubusercontent.com/79627981/232486923-d8cfb662-e51b-4ac9-8b53-a900b173b15d.png) ![image](https://user-images.githubusercontent.com/79627981/232486943-6023ca8f-da7b-4259-81a5-f34467f85927.png)


