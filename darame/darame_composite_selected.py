#!/usr/bin/env python
# coding: utf-8

# In[13]:


import cv2
import numpy as np
import glob
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

def load_npz():
    load_results=np.load('result.npz')
    masks=load_results['masks']
    rois=load_results['rois']
    classIds=load_results['classId']
    return masks, rois, classIds

def pos_inBox(X,Y,rois):
    cnt=0
    for box in rois:
        if ((box[1]<X<box[3]) and (box[0]<Y<box[2])) :
            return cnt
        cnt +=1
        
        assert cnt==len(rois), "다시 터치하세요" # 에러메세지 전달해야됨

def load_images(background_path, foreground_path):
    try:
        background= glob.glob(background_path+'/*.jpg')[0]
    except:
        background= glob.glob(background_path+'/*.png')[0]

    background = load_img(background)
    background = img_to_array(background)
    
    try:
        foreground= glob.glob(foreground_path+'/*.jpg')[0]
    except:
        foreground= glob.glob(foreground_path+'/*.png')[0]

    foreground = load_img(foreground)
    foreground = img_to_array(foreground)

    return foreground, background

def boolstr_to_floatstr(v):
    if v == True:
        return '1'
    else:
        return '0'

def select_mask(index,masks):

    mask_selected=masks[:,:,index]
    
    mask_selected=np.vectorize(boolstr_to_floatstr)(mask_selected).astype(np.uint8)

    return mask_selected

def compositeImages(background, foreground, mask_selected):
    bg_h, bg_w, _ = background.shape
    fg_h, fg_w, _ = foreground.shape
    background = cv2.resize(background, dsize=(fg_w, int(fg_w * bg_h / bg_w)))
    bg_h, bg_w, _ = background.shape
    margin = (bg_h - fg_h) // 2
    
    if margin > 0:
        background = background[margin:-margin, :, :]
    else:
        background = cv2.copyMakeBorder(background, top=abs(margin), bottom=abs(margin), left=0, right=0, borderType=cv2.BORDER_REPLICATE)
    
    background = cv2.resize(background, dsize=(fg_w, fg_h))
    #background = cv2.cvtColor(background,cv2.COLOR_BGR2RGB)
    
    _,alpha = cv2.threshold(mask_selected,0,255,cv2.THRESH_BINARY)

    alpha = cv2.GaussianBlur(alpha,(7,7),0).astype(float)

    alpha = alpha / 255. # (height, width)

    alpha = np.repeat(np.expand_dims(alpha, axis=2), 3, axis=2) # (height, width, 3)

    #pyplot.imshow(alpha)
    foreground = cv2.multiply(alpha, foreground.astype(float))
    background = cv2.multiply(1. - alpha, background.astype(float))  

    compositeImage = cv2.add(foreground, background).astype(np.uint8)
    
    Image.fromarray(compositeImage).save('output/display_result_of_composite.png')

def main():
    background_path = 'img/background'
    foreground_path = 'img/foreground'
    masks, rois, classIds = load_npz()

    X,Y = (100,100) # 지금은 임의의 값을 지정했지만, 안드로이드 소켓통신으로 User가 터치한 좌표를 전달받아야 함
    index = pos_inBox(X,Y,rois)
    foreground,background = load_images(background_path, foreground_path)
    mask_selected = select_mask(index, masks)
    
    compositeImages(background,foreground,mask_selected)

if __name__== "__main__":
    main()

