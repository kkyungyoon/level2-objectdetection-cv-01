import copy
import torch
from detectron2.data import detection_utils
from detectron2.data import transforms as T

import detectron2.data.transforms as T
import torch
from detectron2.structures import BoxMode
import cv2
import numpy as np


class AnnotationBasedCenterCropAndResize:
    def __init__(self,crop_ratios=[0.5, 0.7, 0.9],min_size=5):
        self.crop_ratios = crop_ratios
        self.min_size = min_size
        

    def apply(self, image, annotations):
        image = np.copy(image)

        img_h, img_w = image.shape[:2]

        # 모든 annotations에 대해 처리 (bounding box 정보 활용)
        for ann in annotations:
            bbox = ann["bbox"]
            bbox_mode = ann["bbox_mode"]

            # Detectron2는 다양한 포맷의 bbox를 처리할 수 있음 (포맷 변환)
            if bbox_mode == BoxMode.XYWH_ABS:
                x, y, w, h = bbox
            elif bbox_mode == BoxMode.XYXY_ABS:
                x, y, x2, y2 = bbox
                w = x2 - x
                h = y2 - y
            else:
                raise ValueError(f"Unknown bbox format {bbox_mode}")

            crop_ratio = np.random.choice(self.crop_ratios)

            # 선택된 비율을 사용하여 bbox 크기를 조정
            new_w = int(w * crop_ratio)
            new_h = int(h * crop_ratio)
            if new_w < self.min_size or new_h < self.min_size:
                continue  # 너무 작은 bbox는 건너뜀

            # bbox 안에서 center crop
            center_x = x + w // 2
            center_y = y + h // 2

            # crop 영역 계산
            crop_x1 = int(max(0, center_x - new_w // 2))
            crop_y1 = int(max(0, center_y - new_h // 2))
            crop_x2 = int(min(img_w, center_x + new_w // 2))
            crop_y2 = int(min(img_h, center_y + new_h // 2))

            # 이미지 자르기
            cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]

            # OpenCV를 사용하여 다시 원래 bbox 크기로 resize
            resized_image = cv2.resize(cropped_image, (int(w), int(h)))


            # 자른 부분을 원래 이미지에 덮어쓰기
            image[int(y):int(y) + int(h), int(x):int(x) + int(w)] = resized_image

        return image, annotations


class CustomTrainMapper:

    def __init__(self,image_format):

        self.image_format = image_format
    
    def __call__(self, dataset_dict):
    
        dataset_dict = copy.deepcopy(dataset_dict)
        image = detection_utils.read_image(dataset_dict['file_name'], format=self.image_format)
        
        transform_list = [
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomBrightness(0.8, 1.8),
            T.RandomContrast(0.6, 1.3)
        ]
        crop_resize = AnnotationBasedCenterCropAndResize()
        image, _ = crop_resize.apply(image, dataset_dict['annotations'])

        image, transforms = T.apply_transform_gens(transform_list, image)
        
        dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32'))
        
        annos = [
            detection_utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop('annotations')
            if obj.get('iscrowd', 0) == 0
        ]
        
        instances = detection_utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict['instances'] = detection_utils.filter_empty_instances(instances)
        
        return dataset_dict
