import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def loadimg(p):
    img=cv2.imread(p)
    if img is None:
        raise FileNotFoundError(f"cant find: {p}")
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)

def showkp(img,kps):
    i=cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_GRAY2BGR)
    for k in kps:
        y,x=k['pt']
        cv2.circle(i,(int(x),int(y)),5,(0,255,0),1)
    return i

def showmatch(img1,img2,kp1,kp2,m):
    h1,w1=img1.shape
    h2,w2=img2.shape
    v=np.zeros((max(h1,h2),w1+w2,3),dtype=np.uint8)
    v[:h1,:w1]=cv2.cvtColor(img1.astype(np.uint8),cv2.COLOR_GRAY2BGR)
    v[:h2,w1:]=cv2.cvtColor(img2.astype(np.uint8),cv2.COLOR_GRAY2BGR)
    for mm in m:
        p1=(int(kp1[mm['queryIdx']]['pt'][1]),int(kp1[mm['queryIdx']]['pt'][0]))
        p2=(int(kp2[mm['trainIdx']]['pt'][1])+w1,int(kp2[mm['trainIdx']]['pt'][0]))
        cv2.line(v,p1,p2,(255,0,0),1)
    return v

def mysift(img):
    g=np.float32(img)
    d=cv2.cornerHarris(g,2,3,0.04)
    d=cv2.dilate(d,None)
    pts=np.argwhere(d>0.1*d.max())
    k=[{'pt':(y,x)} for y,x in pts]
    desc=[]
    fkp=[]
    ps=16
    for kp in k:
        y,x=int(kp['pt'][0]),int(kp['pt'][1])
        if y>ps//2 and y<img.shape[0]-ps//2 and x>ps//2 and x<img.shape[1]-ps//2:
            p=img[y-ps//2:y+ps//2,x-ps//2:x+ps//2]
            dsc=p.flatten()
            dsc=(dsc-np.mean(dsc))/(np.std(dsc)+1e-7)
            desc.append(dsc)
            fkp.append(kp)
    return fkp,np.array(desc)

def match(d1,d2,rt=0.75):
    m=[]
    for i,x in enumerate(d1):
        dis=[]
        for j,y in enumerate(d2):
            dist=np.linalg.norm(x-y)
            dis.append((dist,j))
        dis.sort(key=lambda z:z[0])
        if len(dis)>1 and dis[0][0]<rt*dis[1][0]:
            m.append({'queryIdx':i,'trainIdx':dis[0][1]})
    return m

def hom(src,dst):
    if src.shape[0]<4: return None
    a=[]
    for i in range(4):
        x,y=src[i][0],src[i][1]
        u,v=dst[i][0],dst[i][1]
        a.append([-x,-y,-1,0,0,0,u*x,u*y,u])
        a.append([0,0,0,-x,-y,-1,v*x,v*y,v])
    a=np.asarray(a)
    u,s,vh=np.linalg.svd(a)
    l=vh[-1,:]/vh[-1,-1]
    h=l.reshape(3,3)
    return h

def ransac(kp1,kp2,m,it=1000,th=5):
    if len(m)<4: return [],None
    best=[]
    bh=None
    s1=np.float32([kp1[x['queryIdx']]['pt'] for x in m])
    s2=np.float32([kp2[x['trainIdx']]['pt'] for x in m])
    for _ in range(it):
        idx=random.sample(range(len(m)),4)
        ss1=np.float32([s1[i] for i in idx])
        ss2=np.float32([s2[i] for i in idx])
        h=hom(ss1[:,[1,0]],ss2[:,[1,0]])
        if h is None: continue
        inl=[]
        for i in range(len(m)):
            p1=np.array([s1[i][1],s1[i][0],1])
            p2=np.array([s2[i][1],s2[i][0]])
            pt=h@p1
            pt=pt/pt[2]
            e=np.linalg.norm(p2-pt[:2])
            if e<th:
                inl.append(m[i])
        if len(inl)>len(best):
            best=inl
            bh=h
    return best,bh

if __name__=="__main__":
    try:
        i1=loadimg('train.jpg')
        i2=loadimg('query.jpg')
        k1,d1=mysift(i1)
        k2,d2=mysift(i2)
        print("finding key points")
        v1=showkp(i1,k1)
        v2=showkp(i2,k2)
        fig,ax=plt.subplots(1,2,figsize=(15,7))
        ax[0].imshow(v1);ax[0].axis('off')
        ax[1].imshow(v2);ax[1].axis('off')
        raw=match(d1,d2,0.8)
        print("mathcing keypoints")
        vm=showmatch(i1,i2,k1,k2,raw)
        
        plt.figure(figsize=(20,10))
        plt.imshow(vm);plt.axis('off')
        inl,h=ransac(k1,k2,raw,th=5)
        vr=showmatch(i1,i2,k1,k2,inl)
        plt.figure(figsize=(20,10))
        plt.imshow(vr);plt.axis('off')
        plt.show()
    except FileNotFoundError as e:
        print(e)
