import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.metrics import structural_similarity as ssim

def psnr(img1,img2):
    img1=img1.astype(np.float64)
    img2=img2.astype(np.float64)
    
    mse=np.mean((img1-img2)**2)
    if(mse==0):
        return float('inf')

    mpixel=255
    psnr_val=10*math.log10(mpixel**2/mse)
    
    return psnr_val

def jpegCompression(img,scaleFactor):
    
    h,w=img.shape
    img_shifted=img.astype(np.float32)-128
    
    Q = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float32)
    
    Q=Q*scaleFactor
    
    reImage=np.zeros_like(img_shifted,dtype=np.float32)
    
    for i in range(0,h,8):
        for j in range(0,w,8):
            block=img_shifted[i:i+8,j:j+8]
            dct_block=cv.dct(block)
            qblock=np.round(dct_block/Q)
            dqblock=qblock*Q
            idct_block=cv.idct(dqblock)
            reImage[i:i+8,j:j+8]=idct_block
    
    compressedImage=np.clip(reImage+128,0,255)
    compressedImage = compressedImage.astype(np.uint8)
    return compressedImage
    

img=cv.imread("imgB_prenoise.png",0)

compressedImage=jpegCompression(img,1)

h,w=img.shape
h_proc = (h // 8) * 8
w_proc = (w // 8) * 8
imgCropped = img[:h_proc, :w_proc]
compressedImageCropped = compressedImage[:h_proc, :w_proc]
psnr_value=psnr(imgCropped,compressedImageCropped)

cv.imwrite('compressed_image.jpeg', compressedImage)
cv.imshow("img",img)
cv.waitKey(0)
cv.imshow( "compressed Image",compressedImage)
cv.waitKey(0)
cv.destroyAllWindows()


scaling_factors = [0.5, 1, 2, 4, 8]
psnr_results = []
ssim_results = []


h, w = img.shape
h_proc, w_proc = (h // 8) * 8, (w // 8) * 8
original_cropped = img[:h_proc, :w_proc]

for factor in scaling_factors:
    reimage = jpegCompression(img, factor)

    reimage_cropped = reimage[:h_proc, :w_proc]
    
    # 2. Compute PSNR and SSIM values
    psnr_val = psnr(original_cropped, reimage_cropped)
    ssim_val = ssim(original_cropped,reimage_cropped, data_range=255)
    
    psnr_results.append(psnr_val)
    ssim_results.append(ssim_val)


plt.figure(figsize=(14, 6))


plt.subplot(1, 2, 1)
plt.plot(scaling_factors, psnr_results, marker='o', linestyle='-', color='b')
plt.title('psnr vs. quantization scaling Ffctor', fontsize=14)
plt.xlabel('scaling factor', fontsize=12)
plt.ylabel('psnr', fontsize=12)
plt.grid(True)
plt.xticks(scaling_factors)


plt.subplot(1, 2, 2)
plt.plot(scaling_factors, ssim_results, marker='s', linestyle='-', color='r')
plt.title('ssim vs quantization scaling factor', fontsize=14)
plt.xlabel('scaling factor', fontsize=12)
plt.ylabel('ssim', fontsize=12)
plt.grid(True)
plt.xticks(scaling_factors)

plt.suptitle('impact of quantization on image quality', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()