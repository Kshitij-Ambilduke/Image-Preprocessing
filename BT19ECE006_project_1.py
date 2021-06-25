import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mimage
from os import listdir
import random
import numpy as np
import math

class preprocess(object):

    def reading_in_batches(self):
        batch = []
        image_names = listdir(self.location)
#         a = np.arange(0,len(image_names)+1,self.batch_size)
#         print(image_names)
        if self.shuffle:
            random.shuffle(image_names)
        ans = [[]]
        image_names = image_names[0:self.batch_size]
        for i in image_names:
            ans[0].append(i)
        return [ans]
        
    def __init__(self, dataset_location="", batch_size=1,shuffle=False):
        
        self.location = dataset_location
        image_names = listdir(self.location)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batchwise_data = self.reading_in_batches()[0]
        self.idx = np.arange(len(self.batchwise_data))
        self.length= len(image_names)
        print(self.batchwise_data)
        
    def translate(self,tx,ty):
        
        trans = {}
#         print(self.batchwise_data)
        for i in self.batchwise_data:
            for j in i:
                image = mimage.imread(self.location+'/'+j)
                image = np.array(image).astype('uint8')
                new_image1 = np.zeros(image.shape)
                new_image2 = np.zeros(image.shape)
                if np.ndim(image)==2:
#                 if tx<=image.shape[0] or ty<=image.shape[1]:
                    new_image1[:-ty,:] = image[:-ty,:] 
                    new_image2[:,tx:] = new_image1[:,:-tx]
                else:
                    new_image1[:-ty,:,:] = image[:-ty,:,:] 
                    new_image2[:,tx:,:] = new_image1[:,:-tx,:]
                trans[j]=new_image2.astype(np.uint8)
        return trans
    
    def crop(self,id1,id2,id3,id4):
        ans = {}
        new_imgs = []
        for i in self.batchwise_data:
            for j in i:
                image = mimage.imread(self.location+'/'+j)
                img = np.array(image).astype('uint16')
                if img.ndim == 2:
                    new_img = img[img.shape[0]-id1[1]:img.shape[0]-id4[1], id1[0]:id2[0]]
                    new_img = new_img.astype(np.uint8)
                    
                elif img.ndim == 3:
                    new_img = img[img.shape[0]-id1[1]:img.shape[0]-id4[1], id1[0]:id2[0], :]
                    new_img = new_img.astype(np.uint8)
                
                ans[j]=new_img    
                
        return ans
    
    def rgb2gray(self):
        ans = {}
        for i in self.batchwise_data:
            for j in i:
                image = mimage.imread(self.location+'/'+j)
                image = np.array(image).astype('uint16')
                check = image.shape
                if len(check)==3:
                    r = image[:,:,0]
                    g = image[:,:,1]
                    b = image[:,:,2]
                    bw = 0.2989*r + 0.5870*g + 0.1140*b
                else:
                    bw = image[:]
                ans[j]=bw
        return ans
        
    def edge_detection(self): 
        
        ret = {}
        
        gx = [[-1,0,1],
              [-2,0,2],
              [-1,0,1]]

        gy = [[1,2,1],
              [0,0,0],
              [-1,-2,-1]]

        gx = np.array(gx)
        gy = np.array(gy)
        for i in self.batchwise_data:
            for j in i:
                a = mimage.imread(self.location+'/'+j)
                bob = np.zeros((a.shape[0]+2,a.shape[1]+2))
                if np.ndim(a)==3:
                    r = a[:,:,0]
                    g = a[:,:,1]
                    b = a[:,:,2]
                    a = 0.2989*r + 0.5870*g + 0.1140*b
                
                a = np.array(a)
                bob[1:-1,1:-1]=a[:]
#                 print(bob.shape)
                a = bob
                ans1 = np.zeros(a.shape)
                ans2 = np.zeros(a.shape)
                
                for v in range(1,len(a)-1):
                    for w in range(1,len(a[0])-1):
                        ans1[v,w]=np.sum(a[v-1:v+2,w-1:w+2]*gx)
                        ans2[v,w]=np.sum(a[v-1:v+2,w-1:w+2]*gy)

                ans1 = ans1[1:-1,1:-1]
                ans2 = ans2[1:-1,1:-1]

                ans1 = ans1**2
                ans2 = ans2**2
                ans = np.sqrt(ans1+ans2)
                ret[j]=ans.astype(np.uint8)            
        return ret
    
    def blur(self):
        ret={}
        for i in self.batchwise_data:
            for j in i:
                a = mimage.imread(self.location+'/'+j)
                a = np.array(a).astype('uint8')
                ans1 = np.zeros(a.shape).astype('uint8')
                if np.ndim(a)==2:
                    for v in range(1,len(a)-1):
                        for w in range(1,len(a[0])-1):
                            temp=a[v-1:v+2,w-1:w+2].reshape(-1)
                            temp.sort()
                            ans1[v,w]=temp[4]
                elif np.ndim(a)==3:
                    for bittu in range(3):
                        for v in range(1,len(a)-1):
                            for w in range(1,len(a[0])-1):
                                temp=a[v-1:v+2,w-1:w+2,bittu].reshape(-1)
                                temp.sort()
                                ans1[v,w,bittu]=temp[4]
                ret[j] = ans1[1:-1,1:-1].astype(np.uint8)
 
        return ret

    def __getitem__(self,index):
        ans = {}
        for i in self.batchwise_data:
            for j in i:
                a = mimage.imread(self.location+'/'+j)
                a = np.array(a).astype('uint8')
                ans[j] = a
        return ans
    
    def rescale(self,s):
        ans = {}
        cols=[]
        rows=[]
        for x in self.batchwise_data:
            for y in x:
                a = mimage.imread(self.location+'/'+y)
                if np.ndim(a)==2:
                    b = np.zeros((a.shape[0]*s,a.shape[1]*s))

                    for i in range(len(a)):
                        for j in range(len(a[i])):
                            b[i*s,j*s]=a[i,j]

                    temp = []
                    for i in range(len(b)):
                        for j in range(len(b[0])):
                            if b[i,j]!=0:
                                temp.append((i,j))
    #                 rows = []
                    point = {}
                    for i in temp:
                        rows.append(i[0])
                        if i[0] not in list(point.keys()):
                            point[i[0]]=[]
                        point[i[0]].append(i[1])

                    for i in point:
                        l_horizontal = point[i][1]-point[i][0]
                        break

                    for i in list(point.keys()):
                        for k in range(len(point[i])-1): 
                            for j in range(point[i][k],point[i][k+1]):
                                l=1
                                add = (b[i,point[i][k]]*(l_horizontal-l))+(b[i,point[i][k+1]]*(l))
                                l+=1
                                if b[i,j]==0:
                                    b[i,j]=add/l_horizontal    
    #                 cols = []
                    point = {}
                    temp = []
                    for i in range(len(b)):
                        for j in range(len(b[0])):
                            if b[i,j]!=0:
                                temp.append((i,j))

                    for i in temp:
                        cols.append(i[1])
                        if i[1] not in list(point.keys()):
                            point[i[1]]=[]
                        point[i[1]].append(i[0])

                    for i in point:
                        l_vertical = point[i][1]-point[i][0]
                        break

                    for i in list(point.keys()):
                        for k in range(len(point[i])-1): 
                            for j in range(point[i][k],point[i][k+1]):
                                l=1
                                add = (b[point[i][k],i]*(l_vertical-l))+(b[point[i][k+1],i]*(l))
                                l+=1
                                if b[j,i]==0:
                                    b[j,i]=add/l_vertical
                    ans[y]=b.astype(np.uint8)
                
                else:
                    b = np.zeros((a.shape[0]*s,a.shape[1]*s,a.shape[2]))
                    for bittu in range(3):
                        for i in range(len(a)):
                            for j in range(len(a[i])):
                                b[i*s,j*s,bittu]=a[i,j,bittu]

                        temp = []
                        for i in range(len(b)):
                            for j in range(len(b[0])):
                                if b[i,j,bittu]!=0:
                                    temp.append((i,j))

                        point = {}
                        for i in temp:
                            if i[0] not in list(point.keys()):
                                point[i[0]]=[]
                            point[i[0]].append(i[1])

                        for i in point:
                            l_horizontal = point[i][1]-point[i][0]
                            break

                        for i in list(point.keys()):
                            for k in range(len(point[i])-1): 
                                for j in range(point[i][k],point[i][k+1]):
                                    l=1
                                    add = (b[i,point[i][k],bittu]*(l_horizontal-l))+(b[i,point[i][k+1],bittu]*(l))
                                    l+=1
                                    if b[i,j,bittu]==0:
                                        b[i,j,bittu]=add/l_horizontal    

                        point = {}
                        temp = []
                        for i in range(len(b)):
                            for j in range(len(b[0])):
                                if b[i,j,bittu]!=0:
                                    temp.append((i,j,bittu))

                        for i in temp:
                            if i[1] not in list(point.keys()):
                                point[i[1]]=[]
                            point[i[1]].append(i[0])

                        for i in point:
                            l_vertical = point[i][1]-point[i][0]
                            break

                        for i in list(point.keys()):
                            for k in range(len(point[i])-1): 
                                for j in range(point[i][k],point[i][k+1]):
                                    l=1
                                    add = (b[point[i][k],i,bittu]*(l_vertical-l))+(b[point[i][k+1],i,bittu]*(l))
                                    l+=1
                                    if b[j,i,bittu]==0:
                                        b[j,i,bittu]=add/l_vertical

                            ans[y]=b.astype(np.uint8)
        return ans
    
    def resize(self,h,w):
        ans = {}
        for x in self.batchwise_data:
            for y in x:
                a = mimage.imread(self.location+'/'+y)
                h1 = a.shape[0]
                w1 = a.shape[1]
                if h>h1 and w>w1:
                    
                    m = int(h/h1)
                    n = int(w/w1)
                    if np.ndim(a)==2:
                        b = np.zeros((a.shape[0]*m,a.shape[1]*n))
                        bb = np.zeros((a.shape[0]*m,a.shape[1]*n))
                        
                        for i in range(len(a)):
                            for j in range(len(a[i])):
                                b[i*m,j*n]=a[i,j]

                        temp = []
                        for i in range(len(b)):
                            for j in range(len(b[0])):
                                if b[i,j]!=0:
                                    temp.append((i,j))
                        row=0
                        col=0

                        for j in range(len(a)):
                            col=0
                            for k in range(len(a[j])):
                                b[row:row+m,col:col+n] = a[j,k]
                                col+=n
                            row+=m
                        bb = b
                    else:
                        b = np.zeros((a.shape[0]*m,a.shape[1]*n))
                        bb = np.zeros((a.shape[0]*m,a.shape[1]*n,a.shape[2]))
                        for bittu in range(3):

                            for i in range(len(a)):
                                for j in range(len(a[i])):
                                    b[i*m,j*n]=a[i,j,bittu]

                            temp = []
                            for i in range(len(b)):
                                for j in range(len(b[0])):
                                    if b[i,j]!=0:
                                        temp.append((i,j))
                            row=0
                            col=0

                            for j in range(len(a)):
                                col=0
                                for k in range(len(a[j])):
                                    b[row:row+m,col:col+n] = a[j,k,bittu]
                                    col+=n
                                row+=m
                            bb[:,:,bittu]=b
                    ans[y]=bb.astype(np.uint8)
                else:
                    m = int(h1/h)
                    n = int(w1/w)
                    if np.ndim(a)==2:
                        b = np.zeros((h,w))
                        r=0
                        c=0
                        for i in range(len(b)):
                            c=0
                            for j in range(len(b[i])):
                                b[i,j] = np.max(a[r:r+m,c:c+n][0,0])
                        
                                c+=n
                            r+=m
                    else:
                        b = np.zeros((h,w,a.shape[2]))
                        r=0
                        c=0
                        for bittu in range(3):
                            r=0
                            for i in range(len(b)):
                                c=0
                                for j in range(len(b[i])):
                                    b[i,j,bittu] = np.max(a[r:r+m,c:c+n,bittu])
                            #         print(r,c)
                                    c+=n
                                r+=m
                    ans[y]=b.astype(np.uint8)
            
            
            return ans


    def rotate(self,theta):
        t = theta
        cos = math.cos(math.radians(theta))
        sin = math.sin(math.radians(theta))
        ans = {}
        for ex in self.batchwise_data:
            for why in ex:
                a = mimage.imread(self.location+'/'+why)
                l = a.shape[0]
                b = a.shape[1]
                if np.ndim(a)==2:
                    bo = np.zeros((round(l*abs(cos)+b*abs(sin)),round(l*abs(sin)+b*abs(cos))))
                else:
                    bo = np.zeros((round(l*abs(cos)+b*abs(sin)),round(l*abs(sin)+b*abs(cos)),a.shape[2]))
                if np.ndim(a)==2:
                    for y in range(len(a)):
                        for x in range(len(a[y])):
                            x_new = round(x*cos + y*sin)
                            y_new = round(-x*sin + y*cos)
                    #         print("before",x_new)
                            if t<0:
                                x_new+=round(l*abs(sin))
                                x_new = round(x_new)
                            elif t>0:
                                y_new+=round(b*abs(sin))
                                y_new = round(y_new)
                            if x_new<0 or y_new<0:
                                raise ValueError("pagal")
                    #         print(x_new)

                            try:   
                                bo[y_new,x_new] = a[y,x]
                            except:
                                pass
                elif np.ndim(a)==3:
                    for y in range(len(a)):
                        for x in range(len(a[y])):
                            x_new = round(x*cos + y*sin)
                            y_new = round(-x*sin + y*cos)
                    #         print("before",x_new)
                            if t<0:
                                x_new+=round(l*abs(sin))
                                x_new = round(x_new)
                            elif t>0:
                                y_new+=round(b*abs(sin))
                                y_new = round(y_new)
                            if x_new<0 or y_new<0:
                                raise ValueError("pagal")
                    #         print(x_new)

                            try:   
                                bo[y_new,x_new,:] = a[y,x,:]
                            except:
                                pass
                ans[why] = bo.astype(np.uint8)
        return ans


a = preprocess('C:\\Users\\kshitij\\Desktop\\2nd Year\\Project\\TEST',2,False)
b = a.edge_detection()
for i in b:
    plt.imshow(b[i])
    plt.show()

            
        