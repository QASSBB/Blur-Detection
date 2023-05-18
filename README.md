# Blur-Detection演算法
## 目的:
### 判斷影像是否模糊
## 方法:
### 為了判斷一張影像是否模糊，而提出了一種模糊影像判別演算法，這個演算法主要是使用局部功率譜斜率，透過輸入的影像轉換成FFT的形式，得到其頻譜影像，當中包含每個頻率的振幅大小(強度)。因為輸入影像是二維，FFT變換是將空間域變為頻域，對空間域而言顏色變換的程度就是頻率(色彩變化越小頻率越低)，而色彩變化越小就代表影像越模糊，所以透過對FFT影像的頻譜圖建立一個分類器，以區分模糊影像與清楚影像。
## 步驟:
## 輸入影像I(x, y)，x = 0, …, M-1，y = 0, …, N-1，彩色(三通道)。將原始影像I(x, y)轉換成灰階(單通道)影像Ig(x, y) 。對Ig(x, y)作FFT得到影像f(u,v) 。將f (u,v)的原點從左上角移到中心，得到影像fs(u,v) 。傅立葉轉換後的結果是複數的形式，包含了圖像在不同頻率下的振幅和相位信息，所以若fs(u,v) = a + jb，則振幅影像為![image](https://github.com/QASSBB/Blur-Detection/assets/79627981/4a8467a9-2640-4da4-a3d8-d57dca8199b5)，相位影像為：![image](https://github.com/QASSBB/Blur-Detection/assets/79627981/30e05c99-d30f-44af-81a3-145c66cc57ad)
## 而使用np.abs (fs(u,v))可以直接得到fm，np.angle(fs(u,v))可以直接得到fp。得到fm 後，對其值取離散平方值，能讓影像中模糊與清楚的頻率響應更加明顯![image](https://github.com/QASSBB/Blur-Detection/assets/79627981/7a97ad76-23f8-446b-a156-e0bdeeaecd5e)再將fss(u,v)進行極座標轉換 



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
### ![image](https://user-images.githubusercontent.com/79627981/232486691-88f8d215-f5b0-40e9-88c9-b86b438d631f.png) 
### ![image](https://user-images.githubusercontent.com/79627981/232486817-ec09bded-d67e-4073-a2cc-e497ee1b3f39.png) 
### ![image](https://user-images.githubusercontent.com/79627981/232486833-a3cca5ac-e7ce-4c14-a4b0-cf5720a77e02.png)

### 有PCA 
### ![image](https://user-images.githubusercontent.com/79627981/232486904-78cdc8ba-2488-4638-b561-31a72ef6196a.png) 
### ![image](https://user-images.githubusercontent.com/79627981/232486923-d8cfb662-e51b-4ac9-8b53-a900b173b15d.png) 
### ![image](https://user-images.githubusercontent.com/79627981/232486943-6023ca8f-da7b-4259-81a5-f34467f85927.png)

## 使用excel內建函式將400個特徵減少為4個特徵
## 分別為:
### 1.	var(點與平均值之差的平方的平均值(方差)): 使用=VAR
### 2.	quartile(中位數): =QUARTILE
### 3.	slope(斜率): 使用=LINEST
### 4.	intercept(截距): 使用=LINEST
### ![image](https://user-images.githubusercontent.com/79627981/232487279-71990324-7f43-4549-8a23-3cdadc63c006.png)
## 輸出結果:
### ![image](https://user-images.githubusercontent.com/79627981/232487300-c6094364-c7a6-4601-8437-6a0357c7023a.png)
## 程式碼:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score

# 從CSV文件中讀取數據
df = pd.read_csv('E:/python_program/test.csv', encoding='UTF-8')
# 提取特徵和標籤
X = df.loc[:, ['var','quartile','slope','intercept']].values
y = df.loc[:, 'variety'].values
# 將數據集分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 建立支援向量機模型，並指定參數C為1。C代表著錯誤項的懲罰因子，越大代表更嚴格的限制
linearSvcModel = svm.SVC(kernel='linear', C=1.0)
# 使用訓練資料train_reduced和對應的標籤y_train訓練支援向量機模型
linearSvcModel.fit(X_train, y_train)
# 使用測試資料進行預測
y_pred = linearSvcModel.predict(X_test)
# 計算混淆矩陣
# 混淆矩陣是一個用於評估分類模型預測結果的矩陣，用於展示實際樣本和預測樣本之間的差異。
# 混淆矩陣列出了四種結果：真陽性（TP）、假陽性（FP）、真陰性（TN）和假陰性（FN）。
# 混淆矩陣可用於計算模型的準確性、精確度、召回率和F1分數等評估指標。

cm = confusion_matrix(y_test, y_pred)
print("混淆矩陣:\n", cm)
# 計算準確率
accuracy = linearSvcModel.score(X_train, y_train)
print('準確率:',accuracy)
plt.title('LinearSVC (linear kernel)_NoPCA'+ '\n' + 'Accuracy:%.2f' % accuracy)
# 將訓練集資料 X_train 中的第一個和第二個特徵以點的形式繪製在二維平面上。其中，參數 c=y_train 代表將資料按照 y_train 的不同取值(0,1)，分別用不同的顏色標記在圖中
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.show()
```
