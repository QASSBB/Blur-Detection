# Blur-Detection
模糊偵測
模糊圖像判斷方法:
簡介: 使用Sobel運算子對目標圖像進行卷積，若圖像低於設定的閥值則為模糊。
步驟:
1.	讀取圖片，並將圖片灰階化:
image = cv2.imread('圖片.jpg')
 ![image](https://user-images.githubusercontent.com/79627981/205481691-ba87e1de-3a9c-4fc5-92ec-2b36b6898692.png)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 ![image](https://user-images.githubusercontent.com/79627981/205481697-b3f16156-1fac-4c92-9cb1-4cc8d9689af1.png)

2.	使用Sobel進行卷積:
 ![image](https://user-images.githubusercontent.com/79627981/205481704-d056a957-2fa1-4443-8184-526c21cc9042.png)
負責水平邊界 
![image](https://user-images.githubusercontent.com/79627981/205481708-f0ba145a-1f0b-4844-b64e-9f2077b32ec6.png)

 ![image](https://user-images.githubusercontent.com/79627981/205481711-a833bce3-c909-4c66-88a3-52b02d8b466d.png)

 負責垂直邊界
 ![image](https://user-images.githubusercontent.com/79627981/205481715-e19c920c-2854-4b80-a3e2-cae8a984ae1f.png)

A為影像，G為梯度，兩個分量結合為梯度
 ![image](https://user-images.githubusercontent.com/79627981/205481719-5a8a653b-2233-4812-8723-22cf035572f8.png)

3.	計算圖片的平方差
 ![image](https://user-images.githubusercontent.com/79627981/205481722-ff899415-7e62-44f1-879c-10c618e62093.png)

4.	若平方差小於設定好的閥值，則該圖片為模糊圖片
計算出來的值為70.8777490022391
左上:灰階 右上:sobel 之X方向 左下: sobel 之Y方向 右下: sobel之X+Y
![image](https://user-images.githubusercontent.com/79627981/205481728-7af755ed-345f-4276-9a68-91c7b032fa7d.png)
![image](https://user-images.githubusercontent.com/79627981/205481733-8a4e917c-fcef-48e1-aef7-d10015930088.png)
 ![image](https://user-images.githubusercontent.com/79627981/205481739-8926d694-b1d7-4fe7-9637-115810854d4b.png)
 ![image](https://user-images.githubusercontent.com/79627981/205481880-823ccec9-9afe-4f3a-9d85-218659765390.png)

## Reference
https://medium.com/%E9%9B%BB%E8%85%A6%E8%A6%96%E8%A6%BA/%E9%82%8A%E7%B7%A3%E5%81%B5%E6%B8%AC-%E7%B4%A2%E4%BC%AF%E7%AE%97%E5%AD%90-sobel-operator-95ca51c8d78a
