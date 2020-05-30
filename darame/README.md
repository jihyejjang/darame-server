# darame application
----

## folder

- img / foreground : darame 배경합성에서 foreground에 들어갈 사진 또는 mosaic할 사진 저장 폴더

- img / background : darame 배경합성에서 background에 들어갈 사진 저장 폴더

- output : darame에 전송할 image segmentation의 결과와 mosaic/composite의 최종결과 저장 폴더


## .py

- darame_image_segmentation.py : img/foreground의 사진을 input으로 받아 image segmentaion을 수행한 후, 결과 이미지를 output에 저장하고 mask, box, classid 배열을 .npz 파일로 저장

- darame_composite.py : img/foreground와 img/background의 사진을 input으로 받고, darame에서 터치한 X,Y좌표를 받아 해당 마스크와 composite를 수행해 output에 결과 저장

- darame_mosaic.py : img/foreground의 사진을 input으로 받고, darame에서 터치한 X,Y좌표를 받아 해당 마스크를 제외한 person을 mosaic하여 결과 이미지 저장

- darame_server.py : port# 9999에서 darame와 소켓 통신하여 이미지를 받아와 img/foreground에 저장
