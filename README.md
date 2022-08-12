# Animal expression api file

## Introduction
dog-api is animal emotion classification project based on [ResidualMaskingNetwork](https://github.com/phamquiluan/ResidualMaskingNetwork) framework.

### In terminal

1. install python -> make virtual environment
2. 터미널에서 " python3 mmcv_install.py " 실행
3. 터미널에서 " python3 main.py " 실행
4. 내부망 주소창에 서버주소:3334/dog-rec 실행 [ ex) 192.168.*.*:3334/dog-rec ]
5. explain_api 폴더 내 순서로 진행.
6. 출력 : [ 0, 1, 2, 3 ] == ['행복/즐거움', '중립/안정', '슬픔/두려움', '화남/싫음']
  (고양이 ['행복/즐거움', '중립/안정', '화남/싫음'])

가능한 기능은 explain_api 폴더 안에 이미지로 설명하였습니다.
1. 개 & 고양이 (+crop) : 
> 1. '개' or '강' 단어 입력 시 => 개 감정인식 , else 고양이 감정인식 
> 2. 'test' 입력 시 => 1장 이미지 확인
> 3. 'crop' 입력 시 => Detect 모델을 통하지 않고 감정인식
2. 감성 인식 : 이미지 입력

입력된 이미지의 경우 saved/test/*로 저장합니다.

### Major Folder
- **checkpoint**  
Like the Voting Ensemble Project, It involves summing the predictions for each class label and predicting the class label with the most votes. The weights of the models were calculated experimentally through optuna, and the weights of the models can be changed through --proba_conf  
    **./real** : It is checkpoint to dog emotion predict.  
    **./cat** : Comming soon for cat emotion  
- **detect**  
  It is a model for predicting dog or cat box&poses. Even if not all models are used, this code is prepared for future finetunning
- **mmdetection**  
  It is folder to detect dog face. This folder is created as a result of running the installation file, mmcv_install.py .  
- **tools**  
  It is main folder to detect dog face and classificate emotion.  
    detect_class_api.py is main file that do all the processing  
    detect_class_tool.py is tool file that have all tools to precess this project    
- **templates**  
  This have HTML file for showing result from the window (emotion.html)  
- **saved**  
  This is result saved. if you excute [python3 main.py --dog_cat test] in terminal or Insert "test" into "개&고양이(+crop)" in window.  
It save crop_face images, if not, it save raw image  

### Main.py -- argparse
- **port**
: --port number, defalt 3334
- **device**
: --device gpu, default 'cuda:1'
- **save img path**
: --save_img_path save path, default './saved/test'
- **recognition checkpoint**
: --recg_ckdir checkpoint Path, default 'checkpoint/real'
- **model probability**
: --proba_conf [num, num, num], default [1.2, 1.4, 0.3]

## Data
This project model use [반려동물 구분을 위한 동물 영상](https://aihub.or.kr/aidata/34146) for Classification and [animalpose](https://sites.google.com/view/animal-pose/) for pose detection and [COCO](https://cocodataset.org/#home) for bbox detection

## License
This project is released under the [License](https://github.com/phamquiluan/ResidualMaskingNetwork/blob/master/LICENSE).