# darame application
----

## folder

- img : darame에서 받은 이미지들. 모자이크 이미지는 "mosaic.png", 배경 이미지 "background.png", 전경 이미지"foreground.png" 라는 이름으로 저장됨

- output : darame에 전송할 image segmentation의 결과와 mosaic/composite의 최종결과 저장 폴더


## .py

- darame_image_segmentation.py : 'img/foreground.png or mosaic.png'를 input으로 받아 image segmentaion을 수행한 후, 결과 이미지를 output에 저장하고 mask, box, classid 배열을 .npz 파일로 저장

- darame_composite_selected.py : img/foreground.png와 img/background.png를 input으로 받고, darame에서 터치한 X,Y좌표를 받아 해당 마스크와 composite를 수행해 output에 결과 저장

- darame_mosaic_selected.py : img/mosaic.png를 input으로 받고, darame에서 터치한 X,Y좌표를 받아 해당 마스크를 제외한 person을 mosaic하여 결과 이미지 저장

- darame_server_mosaicAct.py : darame에서 모자이크 activity를 실행했을 때 port# 9999에서 darame와 소켓 통신하여 모자이크 process 수행

## mosaic process

1. darame (app) 에서 모자이크 > 사진 선택 > 모자이크 버튼 선택

2. python 서버와 socket통신 시작. 어플 실행하면서 darame_server_mosaicAct.py 코드 실행하여 소켓 열어준다 (같은 네트워크(와이파이) 사용하고 있어야 함, cmd>ipconfig로 자기 ip를 python과 java에 모두 입력해준다.)

3. darame_server_mosaicAct.py 에서 소켓통신
 
앱 -> 서버로 사용자가 선택한 모자이크할 이미지를 img폴더에 "mosaic.png"라는 이름으로 저장한다. 이미지 합성을 하려면 darame_instance_segmentation(mosaic=False) 를 입력해줘야함!!

서버 -> 앱 으로 이미지에 mask를 검출한 결과를 display

앱 -> 서버로 사용자가 원하는 mask를 터치한 X,Y send

서버 -> 앱으로 최종 결과 display

##issue

사이즈가 너무 큰 이미지를 전송하면 python에서 90도 회전시켜서 저장함..

 
