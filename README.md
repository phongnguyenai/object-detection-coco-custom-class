---
title: OBJECT DETECTION - CUSTOM CLASS AND KEEP 80 CLASS OF COCO DATASET
---

0. Prerequisite
You should know:
- Python
- Pytorch
- Detectron2
- COCO
- Object Detection

1. Install Detectron2
```
!pip3 install -U torch torchvision
!pip3 install git+https://github.com/facebookresearch/fvcore.git
!git clone https://github.com/facebookresearch/detectron2 detectron2_repo
!pip3 install -e detectron2_repo
```
2. Download COCO dataset
COCO Dataset includes 81 classes:
- COCO train image (18GB): http://images.cocodataset.org/zips/train2017.zip
- COCO Annotations (241MB): http://images.cocodataset.org/annotations/annotations_trainval2017.zip

3. Prepare Custom Dataset: 
- Install labelme 
```
!pip3 install labelme
```
- Label your dataset (https://github.com/wkentaro/labelme). When you finish there should be json files next to your images
- Download labelme2coco.py: https://github.com/Tony607/labelme2coco/blob/master/labelme2coco.py
- Open label2coco.py to adjust the ID of custom class to avoid conflict with ID in COCO dataset
```
self.annID = 860001 (860001 because I want to make sure it is larger than the ID in COCO)
...
image["id"] = num + 860001
...
category["id"] = 91 # or 81 (I see 91 but COCO say it should be 81, I am researching it, please free to let me know when you discover it)
...
annotation["image_id"] = num + 860001
```
- Convert data to COCO format (trainval is the directory containing images and annotation json files)
```
!python3 labelme2coco.py trainval 
```
- Now you have trainval.json. It is similar to the annotations/instances_train2017.json

4. Merge trainval.json with instances_train2017.json.
- Copy all your images of custom class to folder train2017
- Merging json file:
```
import json
with open("trainval.json") as j:
    my_json = json.load(j)
with open("annotations/instances_train2017.json") as i:
    coco_json = json.load(i)
coco_json['images'].extend(my_json['images'])
coco_json['categories'].extend(my_json['categories'])
coco_json['annotations'].extend(my_json['annotations'])
with open('json_final.json', 'w') as fp:
    json.dump(coco_json, fp)
```

5. Register your dataset

```
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
register_coco_instances("coco_custom", {}, "json_final.json", "train2017")
cocogun_metadata = MetadataCatalog.get("coco_custom")
dataset_dicts = DatasetCatalog.get("coco_custom")
```

6. Train

```
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
cfg = get_cfg()
cfg.merge_from_file(
    "./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
)
cfg.DATASETS.TRAIN = ("coco_custom",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = (
    2000
)  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 81 
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()
```

7. Inference

```
from detectron2.config import get_cfg
import os
from detectron2.engine import DefaultPredictor
cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.MODEL.WEIGHTS = "output/model_final.pth"
predictor = DefaultPredictor(cfg)
```