import cv2 as cv
import numpy as np
from math import log10, sqrt
from scipy import stats




def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    mu1 = cv.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()

with open("MOS_Scores.txt",'r') as file:
    ground_vals = []
    for line in file:
        line = line.strip()        
        line = line.replace("'", "") 
        if line != "":
            ground_vals.append(float(line))
    

ssim_vals=[]

for i in range(0, 10):
    for j in range(14*i +1,14*i+15):
        img1=cv.imread(f"Images/{141+i}.png",0)
        img2=cv.imread(f"Images/{j}.png",0)
        ssim_val=ssim(img1,img2)
        ssim_vals.append(float(ssim_val))
        
print(ssim_vals)

ssim_vals=np.array(ssim_vals)
ground_vals=np.array(ground_vals)

plcc, _=(stats.pearsonr(ssim_vals,ground_vals))
srcc=(stats.spearmanr(ssim_vals,ground_vals).correlation)

# for i in range(0, 10):
#     for j in range(14*i +1,14*i+15):
#         img1=cv.imread(f"Images/{141+i}.png",0)
#         img2=cv.imread(f"Images/{j}.png",0)
plcc_result=abs(plcc)*4
srcc_result=abs(srcc)*4
#         ssim_val=ssim(img1,img2)
#         ssim_vals.append(float(ssim_val))


print(f"Pearson Linear Correlation Coefficient (PLCC) : {plcc_result}")
print(f"Spearman Rank-Order Correlation Coefficient (SRCC) : {srcc_result}")
    
