#!/usr/bin/env python
# coding: utf-8

# In[66]:


#darame 어플 서버 
#기능1: mosaic 할 사진 받기, composite할 전경/배경 사진 받기 (android->python)
#기능2: image instance 결과 전송하기 (python -> android), 최종 결과 전송
#기능3: 사용자가 터치한 좌표 받아오기 (android->python)

from socket import *
import socket
import os
import time
import sys
import base64
import io
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import darame_instance_segmentation
import darame_mosaic_selected


HOST = "192.168.0.6" # Symbolic name meaning all available interfaces
PORT = 9999 # Arbitrary non-privileged port
 
BACKLOG = 10

img_path = "C:/Users/user/데이터 상생플러스 스터디/CNN-PhotoEditor-App/darame/img/"


#서버 소켓 오픈
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(BACKLOG)
print("TCPServer Waiting for client on port 9999")


# 클라이언트 요청 대기중 .
    
client_socket, address = server_socket.accept()
    # 연결 요청 성공
print("I got a connection from ", address)

data = None

# Data 수신

recv_data = client_socket.recv(1024)
data = recv_data


while (recv_data):
    recv_data = client_socket.recv(1024)
    data += recv_data
    #print(recv_data)
        
# 받은 데이터 저장

print("finish img recv")
print(sys.getsizeof(data))
#print(data)
    
#i = base64.b64decode(data)
i = io.BytesIO(data)
i = mpimg.imread(i, format='JPG')

plt.imshow(i, interpolation='nearest')
plt.show()

Image.fromarray(np.array(i).astype(np.uint8)).save(img_path + "mosaic.png")

#마스크 검출
#darame_instance_segmentation.main(mosaic=True) #input 'img/mosaic.png', output 'output/display_maskes.png'
#마스크 보여주고, X,Y좌표 받기

f=open("img/mosaic.png",'rb')# open file as binary

data=f.read()

exx=client_socket.sendall(data)

f.flush()

f.close()

#X,Y좌표 받아서 모자이크하고, 결과 보여주기
X=200
Y=300 #임시
darame_mosaic_selected(X,Y) # output 'output/display_result_of_mosaic.png'

#Todo: 안드로이드에서 전송받은 이미지가 너무 크면 이미지가 돌아감 ㅠ

print("Finished ")

client_socket.close()
print("SOCKET closed... END")

