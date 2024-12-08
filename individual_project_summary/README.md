
# 실험의 목적 : Small Bbox의 낮은 mAP 해결

## EDA
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/f95b21fa-9b2a-4972-9d8e-83c3202130d8" alt="Image 1" width="400"><br>
      <strong>bins=1000</strong>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/2a85bc2c-d57b-4c40-9ce3-6de16b9f9afc" alt="Image 2" width="400"><br>
      <strong>bins=100</strong>
    </td>
  </tr>
</table>
EDA를 진행해보니, bbox의 넓이가 넓게 분포해있음을 확인했다.
그래프가 잘 시각화되지않아, 10만단위 이상은 자르고, 이하만 확인했다.

<img src="https://github.com/user-attachments/assets/c05251d6-9cc2-4aef-ac2a-4189eddb17d0" alt="Image" width="830">    

0 ~ 10,000 구간에 작은 바운딩 박스들이 집중되어 있고, 특히 0 ~ 5,000에서 빈도가 가장 높았다.   
10,000 ~ 40,000 구간은 중간 크기의 바운딩 박스들이 주로 분포되어 있다.   
40,000 이상은 상대적으로 큰 바운딩 박스로 분류할 수 있을 것 같다는 인사이트를 얻었다.   

<br>
<br>
<br>

| | **실험 제목**                     | **주요 내용**                                                                                                                     |
|---------------|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| **1**         | **Backbone 변경을 통한 Small BBox 검출** | **Backbone을 Swin Transformer로 변경**하여 Small BBox 검출 성능을 향상시키고자 했습니다. |
| **2**         | **이미지 해상도 변경을 통한 Small BBox 검출** | **해상도를 조절해 Small BBox 검출 성능을 향상**시키고자 했습니다. |
| **3**         | **Small BBox만 검출하는 모델을 만든 후 앙상블** | **Small BBox만을 검출하는 전용 모델을 학습**하여 앙상블을 통해 성능 향상을 시도했습니다. **32x32 이하, 96x96 이하로 Small BBox 기준을 설정**하고, IoU Threshold를 train_cfg: 0.6, test_cfg: 0.7로 조정했습니다.|
| **4**         | **BBox 크기별로 가중치를 부여하는 실험** | **BBox 크기별로 가중치를 다르게 적용한 Loss 함수 (SizeWeightedLoss) 정의** 후 학습했습니다.|

<br>
<br>
<br>

## 실험 1) **Backbone을 바꿔 Small Bbox를 검출하려는 실험** : Faster R-CNN + Backbone(Swin Transformer)

### **가정**
- 기존 **Faster R-CNN의 Backbone ResNet**에서 **Swin Transformer**로 바꾸면, **전역적, 지역적 특징을 모두 살필 수 있어 작은 BBox를 더 잘 검출**할 수 있을 것이라고 가정했습니다.

---

### **방법**
- **Backbone을 Swin Transformer**로 변경하고, **다양한 Swin 모델을 실험**했습니다.
  - **swin_tiny_patch4_window7_224**
  - **swin_small_patch4_window7_224**
  - **swin_base_patch4_window7_224_22k**

---

### **결론**
- **Backbone을 Swin Transformer로 변경하니 ResNet보다 성능이 향상**되었습니다.
- **swin_tiny_patch4_window7_224**가 **가장 높은 성능**을 기록했습니다.

---

### **성능 비교**

| **Model**         | **Backbone (Image Resolution)** | **LB (Public)** | **LB (Private)** | **mAP50**       | **bbox_mAP_s**   |
|-------------------|---------------------------------|-----------------|------------------|-----------------|-----------------|
| **Faster R-CNN**  | **ResNet (1024x1024)**          | -               | -                | 0.4125      | 0.0321      |
| **Faster R-CNN**  | **Swin_tiny (1024x1024)**       | ✅**0.4999**      | 0.4806       | ✅**0.5100**      | ✅**0.0100**      |
| **Faster R-CNN**  | **Swin_tiny (1024x1024)**       | 0.3495          | 0.3490           | 0.4070          | 0.0000          |
| **Faster R-CNN**  | **Swin_small (224x224)**        | -               | -                | 0.0340          | 0.0000          |
| **Faster R-CNN**  | **Swin_base (224x224)**         | -               | -                | 0.0020          | 0.0000          |

---

### **회고**
1. **근거 부족**
   - **"전역적 특징을 잘 잡아내면 작은 BBox도 잘 잡아낼 수 있다"**는 가정의 근거가 부족했다고 판단됩니다.
   - Swin Transformer가 지역적, 전역적 특징을 모두 잘 잡아내어 성능이 향상되었다고 결론을 내리기에는 명확한 근거가 부족했습니다.
   - **Swin Transformer는 맞추고, CNN은 맞추지 못하는 이미지 샘플을 직접 확인하는 과정**이 필요했을 것입니다.

2. **모델 크기와 성능의 관계**
   - 작은 모델이 더 복잡한 모델보다 **오히려 더 높은 성능을 보일 수 있다**는 점을 확인했습니다.
   - **모델의 크기가 성능을 결정하는 유일한 요소가 아니라는 점**을 알게 되었으며, 데이터와 작업의 특성을 고려한 모델 선택이 중요하다는 것을 깨달았습니다.

<br>
<br>
<br>

## 실험 2) **이미지 사이즈를 변경해서 학습해서 small bbox를 검출하려는 실험**

### **가정**
- **1024x1024 해상도**로 모델을 학습한 후 **2048x2048 해상도**로 **fine-tuning**하면,  
  또는 **2048x2048 해상도**로 모델을 학습한 후 **1024x1024 해상도**로 **fine-tuning**하면,  
  **작은 BBox를 더 잘 검출할 수 있을 것**이라고 가정했습니다.

---

### **방법**
1. **해상도 조절을 통한 Fine-tuning**
   - **1024x1024로 모델 학습 후 2048x2048로 Fine-tuning** 
   - **2048x2048로 모델 학습 후 1024x1024로 Fine-tuning**

2. **Mosaic Augmentation 적용**
   - **2048x2048로 모델 학습 후 1024x1024로 Fine-tuning**(Mosaic Augmentation 20 epoch)  
   - **Mosaic Augmentation의 강도 조절**:  
     - 25 epoch 중 **80%에 Mosaic Augmentation을 적용**하고, **20%는 Augmentation 미적용**  

---

### **결론**
1. **1024x1024 → 2048x2048 Fine-tuning**
   - bbox_mAP_s가 **0.0100 → 0.0210**으로 소폭 상승했습니다.

2. **2048x2048 → 1024x1024 Fine-tuning**
   - **mAP50이 증가**했지만, **bbox_mAP_s는 하락**했습니다.  

3. **Mosaic Augmentation 적용**
   - **Mosaic Augmentation을 적용하니 전반적인 mAP50이 향상**되었지만,  
     **bbox_mAP_s는 오히려 감소**했습니다.

4. **Mosaic Augmentation 80% 적용 후 20% 미적용**
   - **성능 향상 효과는 없었습니다.**   
   - Mosaic Augmentation을 일부만 적용하는 방식이 항상 성능 향상을 보장하지 않는다는 것을 알았습니다.

---

### **성능 비교**

#### **표 1. 1024 → 2048 Fine-tuning**

| **Model**        | **Backbone (Resolution)**      | **LB (Public)** | **LB (Private)** | **mAP50**       | **bbox_mAP_s**   |
|-----------------|---------------------------------|-----------------|-----------------|-----------------|-----------------|
| **Faster R-CNN** | **Swin_tiny (1024)**            | 0.4999          | 0.4806          | 0.5100          | 0.0100       |
| **Faster R-CNN** | **Swin_tiny (1024 → 2048)**     | 0.4408          | 0.4288          | 0.4650          | ✅**0.0210**       |

---

#### **표 2. 2048 → 1024 Fine-tuning**

| **Model**        | **Backbone (Resolution)**      | **LB (Public)** | **LB (Private)** | **mAP50**       | **bbox_mAP_s**   |
|-----------------|---------------------------------|-----------------|-----------------|-----------------|-----------------|
| **Faster R-CNN** | **Swin_tiny (2048)**            | 0.4237          | 0.4138          | 0.4520          | ✅**0.0200**       |
| **Faster R-CNN** | **Swin_tiny (2048 → 1024)**     | -               | -               | ✅**0.5250**      | 0.0080       |

---

#### **표 3. 2048 → 1024 Fine-tuning + Mosaic Augmentation**

| **Model**        | **Backbone (Resolution)**               | **LB (Public)** | **LB (Private)** | **mAP50**       | **bbox_mAP_s**   |
|-----------------|------------------------------------------|-----------------|-----------------|-----------------|-----------------|
| **Faster R-CNN** | **Swin_tiny (2048 → 1024, Mosaic 20 ep)**| ✅**0.5076**      | 0.4909          | 0.5290      | ✅**0.0080**       |

---

#### **표 4. Mosaic 80% 적용 + 추가 5 epoch 학습**

| **Model**        | **Backbone (Resolution)**               | **mAP50**       | **bbox_mAP_s**   |
|-----------------|------------------------------------------|-----------------|-----------------|
| **Faster R-CNN** | **Swin_tiny (2048 → 1024, Mosaic 20 ep + 5 ep 추가 학습)** | 0.4920          | 0.0140       |

---

### **회고**
1. 해상도를 높여 Fine-Tuning하면 성능이 미세하게 향상될 수 있지만, 그 차이가 크지 않을 수 있다는 사실을 알았습니다. 해상도 증가가 항상 큰 성능 향상을 보장하지는 않으며, 데이터 특성에 따라 효과가 제한적일 수 있다는 점을 알았습니다.
2. Mosaic augmentation 내의 설정 및 에폭을 다양하게 바꿔가며 시도해봤어야했다고 생각합니다.

<br>
<br>
<br>

## 실험 3) **Small Bbox만 검출하는 모델 만들어서 앙상블하는 실험**

## 실험 3-1) COCO 공식홈페이지 기준 Small Bbox 32 by 32 사이즈에 해당하는 데이터셋을 만들어 따로 학습
### **가정**
- 작은 객체(bbox)를 더 효과적으로 검출하기 위해, **Small BBox만을 집중적으로 탐지하는 특화된 모델**을 만드는 것이 유리할 수 있다고 가정했습니다.  
- 이후, **전체 객체 검출 모델과 앙상블**하면 다양한 크기의 객체를 더 정확하게 감지할 수 있으며, 특히 **작은 BBox에 대한 검출 성능을 크게 개선**할 수 있을 것이라고 기대했습니다.

---

### **방법**
**Small BBox 데이터셋 생성 및 학습**
   - **COCO 공식 기준**에 따라 **Small BBox(32x32 이하) 데이터셋**을 만들어 따로 학습했습니다.
     ![image](https://github.com/user-attachments/assets/14eecc58-fccb-49f3-80c6-2475c1e1073d)   

---

### **결론**
- **Small BBox만을 검출하는 모델의 bbox_mAP_s 성능이 좋지 않았습니다.**

#### **COCO 공식 기준 (32x32 이하)으로 Small BBox만 학습**

| **Model**               | **실험 설명**                                          | **bbox_mAP_s**  |
|-------------------------|-----------------------------------------------------|-----------------|
| **Faster R-CNN**        | **Baseline** (모든 BBox 학습)                         | ✅**0.0321**      |
| **Retinanet (EffB3 + FPN)** | **32x32 이하 BBox만 학습**                          | 0.0130      |


### **회고**
- 같은 모델에 대해 실험했어야 했다고 생각합니다.
- small bbox를 정의하는 기준이 COCO 공식 홈페이지 기준이 아닌, 현재 데이터 셋에 근거한 기준을 새로 설정했어야 했다고 생각합니다.

<br>
<br>

## 실험 3-2) 32 by 32가 아닌 96 by 96으로 기준을 늘려서 mAP_s 확인
- 실험 3-1의 Output을 시각화해보니,
  ![image](https://github.com/user-attachments/assets/a22aa32e-a9bd-4934-bf51-91f8ee27f09a)
  위와 같은 결과가 나왔고, 이를 아래와 같이 분석했습니다.
  - Training Data Set에 작은 Bbox만 있어서 Predict도 작은 bbox만 나왔다.
  - 배경에 Bbox가 너무 많이 쳐져 있는 것으로 보아, 잘 판별이 안 된 것 같다.
  
  위와 같은 결과가 나온 이유로는 아래와 같이 정리했습니다.
  - 학습데이터가 부족했다.
  - 애초에 training set에 작은 bbox가 쳐져있던 객체들이 배경과 구분이 안 가는 상태였을 수 있다.
  - 32 by 32라는 기준이 COCO 공식 홈페이지 기준이라, 현재 데이터셋에 알맞는 기준을 다시 설정했어야 했다.

- 이에, 학습 데이터 확인을 위한 EDA를 다시 진행했습니다.   
  ![image](https://github.com/user-attachments/assets/0ac70b6b-c2cd-42b7-a5ce-55b1308f3bb1)
  - small bbox는 COCO 공식 홈페이지에 의하면, 
    32 by 32 =  약 900 → bin=100이면, bin 한 칸이 1000 → **가장 왼쪽 한 칸정도가 small bbox**
  - medium bbox는 COCO 공식 홈페이지에 의하면,
    96 by 96 = 약 10000 → bin=100이면, bin 한 칸이 1000 → **왼쪽부터 10칸정도가 medium bbox**
  - 직접 bbox 개수와 이미지 개수를 출력해봤습니다.
    ![image](https://github.com/user-attachments/assets/454149ff-fd2b-40f6-848b-59a4c3236b65)
    ![image](https://github.com/user-attachments/assets/0a5b7039-1125-435c-b587-48c0723c2f39)
  - 학습데이터가 부족한 것을 확인했고, 기준을 늘려야한다고 생각했습니다.
    
### **가정**
- COCO 기준 Medium BBox (96x96 이하)로 Small BBox 정의 후 학습하면, 학습데이터가 늘어나 성능이 좋아질 것이라 생각했습니다.

### **방법**
- COCO 기준 Medium BBox (96x96 이하)로 Small BBox Data Set 정의 후 학습

### **결론**
#### **COCO 기준 Medium BBox (96x96 이하)로 Small BBox 정의 후 학습**

| **Model**               | **실험 설명**                                          | **bbox_mAP_s**  |
|-------------------------|-----------------------------------------------------|-----------------|
| **Retinanet (EffB3 + FPN)** | **96x96 이하 BBox만 학습**                          | ✅**0.0010**      |

### **회고**
- 같은 모델에 대해 실험했어야 했다고 생각합니다.
- 단순히 데이터를 늘리는 것이 해결이 아니라, 현재 데이터 셋에 맞는 기준을 찾았어야 한다고 생각합니다.

<br>
<br>

## 실험 3-3) IoU threshold를 높이기
### **가정**
- 3-1을 시각화 해봤을 때, Output에 잘못 검출된 Bbox가 많았습니다. Recall은 높고 Precision은 낮다고 판단해서 IoU Threshold를 높이면 성능에 효과가 있으리라 가정했습니다.

### **방법**
- IoU Threshold를 높입니다.
  - test_cfg : 0.5 -> 0.7
  - train_cfg : 0 -> 0.6

### **결론**
#### **IoU Threshold 조정 (test_cfg: 0.5 → 0.7, train_cfg: 0 → 0.6)**

| **Model**                | **실험 설명**                                        | **bbox_mAP_s**  |
|--------------------------|---------------------------------------------------|-----------------|
| **Retinanet (EffB3 + FPN)** | **32x32 이하 BBox + IoU Threshold 조정**           | ✅**0.0030**      |
| **Retinanet (EffB3 + FPN)** | **96x96 이하 BBox + IoU Threshold 조정**           | 0.0000      |

<br>
<br>
<br>

## 실험 4) **Bbox 크기별로 가중치를 따로 부여하는 실험**

### **가정**
- **Small BBox의 개수가 상대적으로 적기 때문에**, 해당 객체들의 **Loss에 가중치를 부여**하면 모델이 작은 객체를 더 잘 학습할 수 있을 것이라고 가정했습니다.

---

### **방법**
- **BBox 크기별로 가중치를 부여하는 새로운 Loss 함수 (SizeWeightedLoss) 정의**
  - **size_weighted_loss.py**를 새로 정의하여 사용했습니다.
  - BBox 크기를 기준으로 **Small, Medium, Large로 구분**하고, 크기별로 다른 가중치를 부여했습니다.

---

### **결론**
- **BBox 크기별로 가중치를 부여했으나, bbox_mAP_s의 성능이 개선되지 않았습니다.**
- 오히려 **기본 L1Loss 기반의 baseline 성능보다 하락**했습니다.

---

### **성능 비교**

| **Model**        | **Backbone (Resolution)**   | **Loss (all)**                          | **mAP50**       | **bbox_mAP_s**   |
|------------------|----------------------------|-----------------------------------------|-----------------|-----------------|
| **Faster R-CNN** | **Baseline**               | **L1Loss (weight=1.0)**<br>**CrossEntropyLoss** | 0.4125      | ✅**0.0321**       |
| **Faster R-CNN** | **Swin_tiny (2048→1024)**   | **SizeWeightedLoss**<br>CrossEntropyLoss<br>**Small: 32x32, Medium: 96x96**<br>**Small weight: 3.0, Medium weight: 1.5** | 0.1680      | 0.0090       |
| **Faster R-CNN** | **Swin_tiny (2048→1024)**   | **SizeWeightedLoss**<br>CrossEntropyLoss<br>**Small: 16x16, Medium: 80x80**<br>**Small weight: 4.0, Medium weight: 1.5** | 0.0260      | 0.0000       |
| **Faster R-CNN** | **Swin_tiny (1024)**        | **SizeWeightedLoss**<br>CrossEntropyLoss<br>**Small: 50x50, Medium: 150x150**<br>**Small weight: 3.0, Medium weight: 1.5** | 0.1600      | 0.0040       |

---

### **회고**
1. **BBox 크기 기준의 문제**
   - **COCO 기준**(Small: 32x32, Medium: 96x96, Large: 96x96 이상)을 사용했지만, **현재 사용 중인 데이터셋의 특성과 맞지 않을 가능성**이 큽니다.
   - 예를 들어, **96x96도 작은 BBox로 간주해야 할 수 있는데**, 이를 고려하지 않고 COCO 기준을 그대로 적용했습니다.
   - **현재 데이터셋에 맞는 Small, Medium, Large BBox 기준을 새롭게 정의**했어야 했습니다.

2. **Loss의 가중치 조정 문제**
   - Small BBox에 가중치를 3.0 ~ 4.0으로 설정했지만, **과도한 가중치로 인해 전체 모델의 학습이 불안정**해졌을 가능성이 큽니다.
   - 특정 객체에 **너무 큰 가중치를 부여하면 모델의 일반화 성능이 저하**될 수 있음을 배웠습니다.
   - **가중치를 1.2, 1.5, 2.0 등으로 소폭 조정하는 실험**을 추가로 해봤어야 했습니다.
