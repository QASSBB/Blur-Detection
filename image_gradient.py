import numpy as np
import cv2

def Sobel_gradient(f, direction = 1):
    sobel_x = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    sobel_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    if direction ==1:
        grad_x = cv2.filter2D( f, cv2.CV_32F , sobel_x)
        gx = abs(grad_x)
        g = np.uint8(np.clip(gx,0,255))
    elif direction == 2:
        grad_y = cv2.filter2D( f, cv2.CV_32F , sobel_y)
        gy = abs(grad_y)
        g = np.uint8(np.clip(gy,0,255))
    else:
        grad_x = cv2.filter2D( f, cv2.CV_32F , sobel_x)
        grad_y = cv2.filter2D( f, cv2.CV_32F , sobel_y)
        magnitude = abs(grad_x) + abs(grad_y)
        g = np.uint8(np.clip(magnitude,0,255))
    return g

def main():
    img = cv2.imread("E:/專研/51983張圖/1105306198_20220411_110708.jpg")
    
    img = cv2.resize(img, None, fx=0.25, fy=0.25)
    cv2.imshow('aa',img)
    img =  cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
    
    gx = Sobel_gradient(img,1)
    gy = Sobel_gradient(img,2)
    g = Sobel_gradient(img,3)
    value = g.var() 
    print(str(value**0.5))
    
    hitch  = np.hstack((img,gx))
    hitch2 = np.hstack((gy,g))
    vitch = np.vstack((hitch,hitch2))
    cv2.namedWindow("all",0)
    cv2.imshow("all",vitch)

    cv2.waitKey(0)
    cv2.destoryAllWindows()
main()