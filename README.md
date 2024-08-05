# LaneDet-Carla
![lanedet-carla-gif-min](https://github.com/user-attachments/assets/31db0618-9d6d-4368-807d-985cdc8641d8)
## Introduction

lane detection 모델 학습과 평가를 위한 베이스라인 코드   

original source code: https://github.com/Turoad/lanedet 

<details>
    <summary>Support Model</summary>
    <div markdown="1">
        <ul>
            <li><a href='https://arxiv.org/pdf/1712.06080'>SCNN</a></li>
            <li><a href='https://arxiv.org/pdf/2004.11757'>UFLD</a></li>
            <li><a href='https://arxiv.org/pdf/2010.12035'>LaneATT</a></li>
            <li><a href='https://arxiv.org/pdf/2105.05003'>CondLane</a></li>
        </ul>
    </div>
</details>
<details>
    <summary>Support Dataset</summary>
    <div markdown="1">
        <ul>
            <li>TuSimple</li>
            <ul> 
                <li><a href='https://drive.google.com/file/d/1cTCLcsTVF2M6rIxUac-3DsyBPHvh1Jom/view?usp=sharing'>download link</a> (학습에 바로 사용 가능한 형태로 구성)</li>
                <li>support metric :  accuracy</li>
            </ul>
            <li>DssDataset</li>
                <ul> 
                    <li>고신뢰성 물리 기반 자율주행 시뮬레이션인 Divine Sim Suite(DSS)에서 수집된 자율주행 인공지능 모델 학습용 데이터</li>
                    <li>support metric :  dice</li>
                </ul>
            <li>CULane</li>
            <ul> 
                <li>support metric :  f1-score</li>
            </ul>
        </ul>
    </div>

</details>

## New Features
<details> 
    <summary>1st. Support DssDataset</summary>
    <div markdown="1">
        <ul>
            <li>DssDataset에 대한 모델 학습과 평가 지원</li>
        </ul>
    </div>
</details>
<details> 
    <summary>2nd. Carla Demo</summary>
    <div markdown="1">
        <ul>
            <li>오픈소스 자율주행 시뮬레이터인 Carla에서 학습된 모델 검증</li>
        </ul>
    </div>
</details>
<details> 
    <summary>3rd.  Experiment Tracking</summary>
    <div markdown="1">
        <ul>
            <li>지속적인 모델의 성능 비교와 학습과정 모니터링을 위한 유틸리티 제공</li> 
            <ul>
                <li><a href='https://kr.wandb.ai/'>WandB</a>a - learning rate, training loss, vailation metric tracking 가능</li>
                <li>학습 과정에서의 inference 결과 모니터링 가능</li>
                <li>학습 단위로 실험의 config와 best model의 가중치 파일 저장</li>
            </ul>
        </ul>
    </div>
</details>
<details> 
    <summary>4th. Custom Augmentation</summary>
    <div markdown="1">
        <ul>
            <li>Albumentation 라이브러리를 활용한 Custom Augmentation 구현 및 적용 가능</li> 
            <li>RandAugment 지원</li> 
        </ul>
    </div>
</details>

## Setup Environment

```bash
# 1. create workspace
$ cd ~ && mkdir lanedet-workspace && cd lanedet-workspace 

# 2. clone repository 
$ git clone https://github.com/kjs2109/lanedet-carla.git 

# 3. prepare dataset (you should have folder structure like below)
lanedet-workspace
├── lanedet-carla
└── DssDataset 
        ├── rawData
        └── labelingData

# 4. docker pull and run 
$ docker pull jusungkim/lanedet-carla-devel:v1.5
$ docker run -it --shm-size=8g --gpus all --name lanedet-carla-v15 -v ./lanedet-carla:/root/lanedet-carla -v ./DssDataset:/root/DssDataset jusungkim/lanedet-carla-devel:v1.5  

# 5. setup and build 
$ cd /root/lanedet-carla 
$ python setup.py build develop 

# 6. prepare dss dataset
$ python tools/prepare_dss_dataset.py  
```

## Quick Start

```bash
# train model 
$ python main.py --config ./configs/scnn/resnet18_dss.py --exp exp1_scnn-resnet18_dss_base 

# validation 
$ cp ./work_dirs/DssDataset/{exp_name}/{exp_date}/config.py ./demo/configs/ 
$ cp ./work_dirs/DssDatast/{exp_name}/{exp_date}/ckpt/best.pth ./demo/checkpoints
$ python main.py --evaluate --config ./demo/configs/config.py --load_from ./demo/checkpoints/best.pth 

# inference 
$ python tools/my_detect.py --config ./demo/configs/config.py --img ./demo/images --load_from ./demo/checkpoints/best.pth --savedir ./vis

```

## Deep Dive

For more details, please refer to [this guide](https://github.com/kjs2109/lanedet-carla/blob/main/docs/Landet-Carla%20%ED%99%9C%EC%9A%A9%20%EA%B0%80%EC%9D%B4%EB%93%9C.pdf)

### Experiment Tracking

1. create a WandB account 
2. Set the entity in the main.py file to your username.
3. train the model with the `--tracking` option set to True  
`python main.py --config ./configs/scnn/resnet18_dss.py --tracking True` 

### Custom Augmentation

1. define and register your custom augmentation in `lanedet/datasets/process/transforms.py` 
2. configure the custom augmentation defined in the config file

### Carla Demo

1. install [carla 0.9.13](https://carla.readthedocs.io/en/latest/start_quickstart/#b-package-installation) (package version) 
2. start the carla server  
`./CarlaUE4.sh` 
3. start the carla client (demo scripts)  
`python PythonAPI/examples/demo_lanedet.py --host {server IP}`
