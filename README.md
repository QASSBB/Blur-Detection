# Blur-Detection演算法
## 目的:
### 判斷影像是否模糊
## 方法:
### 為了判斷一張影像是否模糊，而提出了一種模糊影像判別演算法，這個演算法主要是使用局部功率譜斜率，透過輸入的影像轉換成FFT的形式，得到其頻譜影像，當中包含每個頻率的振幅大小(強度)。因為輸入影像是二維，FFT變換是將空間域變為頻域，對空間域而言顏色變換的程度就是頻率(色彩變化越小頻率越低)，而色彩變化越小就代表影像越模糊，所以透過對FFT影像的頻譜圖建立一個分類器，以區分模糊影像與清楚影像。
## 步驟:
### 輸入影像後將影像轉為灰階格式，並對其做FFT轉換，因為經過FFT轉換後的結果是複數的形式，包含了圖像在不同頻率下的振幅和相位信息
### 振幅影像
![image](https://github.com/QASSBB/Blur-Detection/assets/79627981/5d9878b8-ddfc-495c-a291-3052c398b8a9)
### 相位影像
![image](https://github.com/QASSBB/Blur-Detection/assets/79627981/5666c91f-4c83-44d6-b85e-d9f0daecd757)
### 而使用np.abs (fs(u,v))可以直接得到fm，np.angle(fs(u,v))可以直接得到fp。因為只需要傅立葉轉換的振幅訊息，所以得到fm 後，對其值取離散平方值，能讓影像中模糊與清楚的頻率響應更加明顯。取離散平方值後，對其進行極座標轉換，並將轉換後得到的(r,θ)設為一維陣列。
### 將Y軸代表S(r)，X軸為r，即可以作出X軸為頻率，Y軸為振幅的頻譜圖，得到每個半徑之振幅的和
![image](https://github.com/QASSBB/Blur-Detection/assets/79627981/b6272260-a3b5-43db-877a-94792ac8f98b)
![image](https://github.com/QASSBB/Blur-Detection/assets/79627981/54808c03-b7f5-4e95-9ed2-84974944f4f9)
### 之後，將它們匯入至csv格式的Excel檔案中，並以S(r)作為特徵(feature)，以清楚(1)與模糊(0)當作標籤(label)，使這100張影像分別作為訓練集(70%)與測試集(30%)
![image](https://github.com/QASSBB/Blur-Detection/assets/79627981/57ec2ad4-0b83-42c9-b1a8-3b9d6a50be19)
### 當特徵維度過高時，會造成計算資源的浪費和運算效率的降低，同時也容易產生維度災難的問題(curse of dimensionality)。因此在使用SVM等機器學習模型作為分類器時，通常會先進行特徵選擇或降維處理，將原本高維的特徵向量轉化為較低維的特徵向量，以提高模型的訓練和預測效率。在這個情況下，因為對於一張影像，當r的數量為400，亦即一張影像的特徵有400個；為了將特徵減少，所以本研究使用冪函數進行S(r)曲線的擬合，而可以使用函數的3個參數a、b、c作為特徵。
### 其中a、b兩個特徵使用冪函數來建立y=a*x^b，在針對S(r)曲線的擬合時，將r設為x，S(r)設為y，再取對數得log*y = log*a + b*log*x，對每個數據帶入此算式可得:![image](https://github.com/QASSBB/Blur-Detection/assets/79627981/91761652-63e5-4c56-a013-1f91ff0dab7a)
其中![image](https://github.com/QASSBB/Blur-Detection/assets/79627981/67b4cdad-ab24-43af-b539-c67d28770934)

### 邊緣偵測可以拿來當作辨識模糊影像的特徵，主要是因為模糊影像中的邊緣通常會失去清晰的邊緣細節。因此，透過邊緣偵測，可以提取出影像中的邊緣特徵，並且利用這些特徵來進行模糊影像的辨識。最後參數c使用邊緣偵測，而當中有許多種方法，像Canny、Laplacian等，但Sobel算子的濾波器大小和閾值可以進行調整且效果較佳，所以選擇Sobel算子為邊緣偵測之方法。由於Sobel算子可以調整濾波器的大小(Ksize)，而一般常見的ksize大小有3、5、7等。當ksize=3時，表示使用一個大小為3x3的Sobel算子進行邊緣檢測，此時檢測到的邊緣會比較銳利，但可能存在噪聲。當ksize=5或7時，表示使用一個更大的Sobel算子進行邊緣檢測，此時檢測到的邊緣可能會更平滑，但可能會失去一些細節信息，所以檢測模糊影像時使用的是ksize=3的大小來進行檢測。然後分別計算每個像素dx、dy兩個方向的邊緣強度值，再分別平方後相加，開根號得到的值，即為像素的邊緣響應。接著，計算所有像素的邊緣響應總和平均，並使用正規化將參數c的值限制在[0-1]之間。透過上述的方法，可將每一張影像的特徵由S(r)的400個特徵，減少為3個特徵(a, b,c)。這樣可以有效地降低計算和存儲的成本，同時也可以避免過擬合(overfitting)的問題。

## 分類器
### 線性SVC模型
### 使用svm.LinearSVC 參數對資料進行訓練，設正則化參數C為10控制最大化邊界與保持觀測值的適當分類之間的權衡，並設是指最大迭代次數max_iter為100000000，再分割訓練集與測試集的比例為7:3。
![image](https://github.com/QASSBB/Blur-Detection/assets/79627981/7b4eeff3-4fd6-4362-8478-3d77dfc04192)
![image](https://github.com/QASSBB/Blur-Detection/assets/79627981/13a3b306-51be-4ff2-baa9-4496f37bc857)
### 非線性SVC模型
### 使用svm. SVC 參數對資料進行訓練，設正則化參數C為10控制最大化邊界與保持觀測值的適當分類之間的權衡，並設是指最大迭代次數max_iter為100000000，再分割訓練集與測試集的比例為7:3。
![image](https://github.com/QASSBB/Blur-Detection/assets/79627981/d7525e47-7dae-452c-8d81-862fc860b5f5)
![image](https://github.com/QASSBB/Blur-Detection/assets/79627981/78af4d77-2d67-40b8-8beb-8be1bc70c71d)
### 貝式分類模型
### 使用GaussianNB模型進行訓練，分割訓練集與測試集的比例為7:3。
![image](https://github.com/QASSBB/Blur-Detection/assets/79627981/11c25b1b-2791-4a63-816f-1e05721a77d7)
 ![image](https://github.com/QASSBB/Blur-Detection/assets/79627981/ecf92f3e-4cba-4daa-92ad-8b4fff82f0ba)
### 隨機森林模型
### 使用RandomForestClassifier模型進行訓練，設定森林中樹木的數量n_estimators為100使模型效果變好，與隨機種子random_state為42。
![image](https://github.com/QASSBB/Blur-Detection/assets/79627981/27eea49a-1ef9-437a-a329-f32bcc492a29)
![image](https://github.com/QASSBB/Blur-Detection/assets/79627981/b11edcba-65a7-4b4a-9d7b-91b15ce509ca)
### 羅吉斯回歸模型
### 使用LogisticRegression模型進行訓練，分割訓練集與測試集的比例為7:3。
![image](https://github.com/QASSBB/Blur-Detection/assets/79627981/27fe2d2d-e334-43ee-a663-cddc5c5f42d4)
![image](https://github.com/QASSBB/Blur-Detection/assets/79627981/8ca26142-cec5-4901-92cc-16c40a75bcc7)
### 最後可以發現非線性與隨機森林這兩個模型偵測出來的結果最好，而線性分類、羅吉斯回歸、貝式分類這三個模型偵測的錯誤大多都是1、11、13、15、18、19 這幾張影像偵測錯誤，所以應該要針對這幾張影像的結果對模糊的定義進行修正。
![image](https://github.com/QASSBB/Blur-Detection/assets/79627981/a93da7e7-b3fd-46bd-839a-828c14181256)
![image](https://github.com/QASSBB/Blur-Detection/assets/79627981/f221bece-dbea-4b3e-8ccf-9d9fef3653de)

```
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
# 讀取圖片
img = cv.imread('E:/uu187/Pictures/ImageBlur/模糊/37.jpg')
# 固定圖片大小
img = cv.resize(img,(640,480))
# cv.imshow('img',img)
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
#fss = (np.abs(fr)**2)/math.sqrt(width*height)
fss = (np.abs(fr)**2)
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
# 設一個大小為np.max(r_array)的零陣列，用來儲存半徑 1~最大值(ex:右上角)的所有振幅(fss)的值
# 因為要把每一個半徑的所有角度的值加總起來
sr = [] 
asr = [] 
sr_max = int(np.max(r_array)) 
sr = np.zeros(sr_max)
count = np.zeros(sr_max)


# 把r_array每個位置的值(半徑)，都加上fss的值
for v in range(r_array.shape[0]): # 高(row)
    for u in range(r_array.shape[1]): # 寬(column)
            # sr[]裡面放的是r_array[v,u]的值，也就是位置r_array[v,u]的半徑
            # 第43行會把[v,u]位置的值(震幅)加入sr[r_array[v,u]]的位置中，而這個位置也是半徑的大小
            # 所以有相同的半徑的話他的值就會累加上去
            sr[r_array[v,u]:r_array[v,u]+1] += fss[v,u] # 將'相同半徑(r_array)'的'所有振幅(fss)'加總至一維陣列sr
            count[r_array[v,u]:r_array[v,u]+1] += 1
# 將sr取平均(除以對應半徑的振幅數量)
# Numpy 陣列設為另一個 Numpy 陣列的值時，它們實際上是指向同一個內存位置的，所以當你更改一個陣列的值時，另一個陣列也會發生變化。
# 所以要加上.copy()
asr = sr.copy() 
for i in range(len(sr)):
    if count[i] > 0:
        asr[i] /= count[i]
        
plt.plot(asr)
plt.title('Image_Clear')
plt.xlabel('r')
plt.ylabel('s(r)')
plt.xticks(np.arange(0, 400, 50)) # 設X軸的刻度範圍在0-400，50為一個刻度
plt.yticks(np.arange(0, 400, 50)) # 設Y軸的刻度範圍在0-400，50為一個刻度 
plt.show()  
# 解power function ，設y=ax^b，(x,y)為(r,s(r))
# theta = [log a ,b] Y=[(log y1),...,(log yn)] X=[(1,log x1),...(1,log xn)]
# 解聯立: (X^T X)^(-1) X^T Y = theta = [log a ,b] , a= 10^log a
r = np.arange(1, 401)
s_r = asr.copy()
# 建立X、Y矩陣
# np.column_stack: 將兩個一維陣列以列的形式組合成一個二維陣列，第一行是元素均為 1，第二行是 r 陣列元素的自然對數。
# (np.log(r)/ np.log(10)): 因為np.log是以e為底數，所以要以10為底數的話要在除以log10才行
X = np.column_stack( ( np.ones(len(r)), (np.log(r) / np.log(10)) ) )
Y = ( np.log(s_r) / np.log(10) )
# 計算
# np.dot: 矩陣乘法，np.linalg.inv: 求逆矩陣，X.T: 求矩陣X的ˊ轉置矩陣X^T
theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
# log 以10為底
a = 10 ** theta[0]
b = theta[1]
print("a =", a)
print("b =", b)
print('Power Function y=ax^b之解為:')
print('y=',a,'x^',b)
```
