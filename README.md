# CNN-PhotoEditor-App
### 데이터 상생플러스 스터디 CNN Image segmentation을 활용한 사진 배경 합성/ 모자이크 앱 2020.4.13~6.5
 
 - CNN으로 image segmentation에 대해 스터디: segmentation모델로 사람을 선택하고, 선택한 사람 이외/선택한 사람만 처리하는 기능구현
 
 - pre-trained MASK RCNN모델을 활용해 이미지가 입력되면 기능이 적용된 이미지가 출력되는 pipline으로 사용할 수 있게 함수를 만들고 코드를 정리하는 작업을 진행
 
 - 안드로이드(사진 선택) <-> 파이썬(사진 처리 및 전송)
 
 - [어플 구현](https://github.com/jihyejjang/darame-app)

----

## 1주차 : Image Segmentation Study
image segmentation에 대한 searching

**<image를 처리하는 computer vision에서의 문제>**

**1. classification** : image당 하나의 label을 분류

- AlexNet, ResNet, Xception 등의 모델

**2. object detection(localization)** : image안의 특정 object의 위치를 detect

- face detection의 경우 얼굴의 위치만을 검출, face recognition은 누구의 얼굴인지를 예측

- YOLO,R-CNN 등의 모델

**3. segmentation** : image안의 특정 object의 위치를 픽셀단위로 검출

- FCN, SegNet, DeepLab 등의 모델

**<구현할 것>**

1. 배경 합성
<img src="https://user-images.githubusercontent.com/61912635/82907431-92d1de80-9fa1-11ea-9c8d-ad8819bae79d.png" width="80%"></img>

2. 사람 인식 -> 모자이크

----

## 2주차 : Semantic Segmentation study
https://github.com/kairess/semantic-segmentation-pytorch 링크의 source code를 실행해보고, instance Segmentation에 대한 searching

**컴퓨터비전 스킬의 분류**

<img src = "https://user-images.githubusercontent.com/61912635/82907718-efcd9480-9fa1-11ea-8f23-3f8e75cb58bb.png" width ="80%"> </img>

이미지를 의미있는 픽셀 단위로 분할해내지만, object 개별 요소에 대한 구분을 하지 않음 (즉 사람이 많은 image에서 사람을 개별적 사람1, 사람2 사람3,.. 으로 분할하는 게 아니라 그냥 "사람" 으로 label한다는 뜻)

-> instance segmentation에서 이 문제를 해결함

-> classification을 통해 진행됨

<img src ="https://user-images.githubusercontent.com/61912635/82907881-2b685e80-9fa2-11ea-828a-7ed529591fca.png" width ="90%"></img>

Semantic Segmentation은 사람, 자동차, 신호등, 표지판 등을 구별해내야 하는 자율주행 자동차와 폐,뼈,간 등을 구별해내는 흉부X-Ray사진 등에 활용될 수 있다.

Semantic Segmentation의 대표적인 알고리즘으로 FCN(Fully Connected Network), DeepLab V3+모델이 있다

----
## 3주차 : Instance Segmentation
각 영역을 categorizing하는 semantic segmentation과 달리 object의 개별 instance를 구별한다.
Mask R-CNN을 사용한다

<img src = "https://user-images.githubusercontent.com/61912635/82908045-636fa180-9fa2-11ea-9339-3f943f410397.png" width="80%"></img>

**Faster R-CNN**
CNN의 input으로 image가 들어가고, convolutional feature map을 output으로 도출한 후, convolutional feature map에서 object 영역을 식별하고, RoI 풀링 계층 통과 하여 고정 크기로 재구성 -> feature map을 feature vector에 mappint하기 위해 FC(Fully connected) 계층을 사용하고, softmax 레이어 사용

output: object 후보들, bounding-box의 class label
<img src = "https://user-images.githubusercontent.com/61912635/82908158-89954180-9fa2-11ea-9d00-4fde764b8973.png" width="80%"></img>

### Mask R-CNN (우리가 사용할 모델)
  Mask Regional Convolutional Neural Network (Mask R-CNN)은 object detection에서 사용되는 faster R-CNN의 확장형으로, Roi(Object box area)에서 픽셀 단위로 segmentation mask를 예측하는 알고리즘이다.

1. object에 포함될 가능성이 높은 영역을 image에서 scan함

2. scan한 영역을 분류하고 detection box와 mask를(픽셀단위 영역) 만든다

  output: 후보 object, object의 bounding box(Faster R-CNN과 동일) + object mask
  
  object의 bounding boxes를 제공하는 R-CNN의 맨 끝층에 FCN을 추가하여, pixel이 object에 속하면 1, 아니면 0 인 matrix를 제공한다. 즉, box검출 -> mask검출 두 단계로 이루어진다.
  
  <img src = "https://user-images.githubusercontent.com/61912635/82908350-ce20dd00-9fa2-11ea-962e-ddae7e5235a6.png" width="80%"></img>

https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py  Mask R-CNN Model Source

**Mask R-CNN implementation using Keras**

https://towardsdatascience.com/computer-vision-instance-segmentation-with-mask-r-cnn-7983502fcad1 Mask R-CNN Implementation

**1.Mask R-CNN모델 github 저장소 다운로드하기**

-> setup.py install

<img src="https://user-images.githubusercontent.com/61912635/82908635-396aaf00-9fa3-11ea-9706-a5019f3654f3.png" width="70%"></img>

**2.Mask R-CNN로 미리 트레인한 가중치 coco를 matterport(Mask R-CNN 모델 코드 제작자)에서 다운로드하여 Mask R-CNN 폴더에 넣어준 후, 패키지를 import**

<pre>
<code>
# import the necessary packages
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn
import numpy as np
import colorsys
import argparse
import imutils
import random
import cv2
import os
from matplotlib import pyplot
from matplotlib.patches import Rectangle
%matplotlib inline
</code>
</pre>

**3.myMaskRCNNConfig 라는 클래스와 상속클래스(잘모름)**

<pre>
<code>
class myMaskRCNNConfig(Config):
    # give the configuration a recognizable name
    NAME = "MaskRCNN_inference"
 
    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # number of classes (we would normally add +1 for the background
    # but the background class is *already* included in the class
    # names)
    NUM_CLASSES = 1+80
</code>
</pre>

<pre>
<code>
config = myMaskRCNNConfig()
</code>
</pre>

**4. config 사용하여 Mask R-CNN을 inference로 초기화(?)**

<pre>
<code>
print("loading  weights for Mask R-CNN model…")
model = modellib.MaskRCNN(mode="inference", config=config, model_dir='C:/Users/user/anaconda3/Lib/site-packages/Mask_RCNN')
</code>
</pre>

**5. 미리 받아놓은 가중치(coco) 로드하기 (이때 현재 작업 폴더에 가중치파일을 넣어야함)**

<pre>
<code>
model.load_weights('mask_rcnn_coco.h5', by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
 'bus', 'train', 'truck', 'boat', 'traffic light',
 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
 'kite', 'baseball bat', 'baseball glove', 'skateboard',
 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
 'donut','cake', 'chair', 'couch', 'potted plant', 'bed',
 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
 'teddy bear', 'hair drier', 'toothbrush']
 
 </code>
 </pre>
 
Mask-RCNN을 활용한 배경 합성과, 모자이크 메인함수는 darame_mosaic.py , darame_composite.py에 저장  
6주차 이후~ Android studio를 활용하여 py와 소켓통신하는 앱 개발  
 
### References

**Semantic Segmentation**

[논문]

Fully convolutional networks for semantic segmentation(FCN).2014 CVPR.Long et al.

[이론]

https://bskyvision.com/491

https://medium.com/hyunjulie/1%ED%8E%B8-semantic-segmentation-%EC%B2%AB%EA%B1%B8%EC%9D%8C-4180367ec9cb

https://zzsza.github.io/data/2018/05/30/cs231n-detection-and-segmentation/

[코드]

https://github.com/GeorgeSeif/Semantic-Segmentation-Suite

**Instance Segmentation**

[이론]

https://missinglink.ai/guides/neural-network-concepts/instance-segmentation-deep-learning/

https://towardsdatascience.com/instance-segmentation-using-mask-r-cnn-7f77bdd46abd

https://towardsdatascience.com/computer-vision-instance-segmentation-with-mask-r-cnn-7983502fcad1

[코드]

https://github.com/matterport/Mask_RCNN

https://reyrei.tistory.com/11
