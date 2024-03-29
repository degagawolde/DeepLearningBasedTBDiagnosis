# Deep Learning Based Chest X-ray Diagnosis

Used a pre-trained U-Net model with 97% dice coefficient and MobileNetV2 with 98% for lung area segmentation and Lung disease classification.

[Chext x-ray](https://www.mayoclinic.org/tests-procedures/chest-x-rays/about/pac-20393494) produce images of organs located under the chest(heart, lungs, blood vessels, airways, and the bones of chest and spine). Chest X-rays can also reveal fluid in or around the lungs or air surrounding a lung
is a chest radiography used for the diagnosis for diseases attacking lung. Radiologists can look at the image and examine if there is anomalies in the x-ray image. In this project, we traiend a deep learning model- [MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2/MobileNetV2) for the diagnosis of the chest x-ray images.

![chest x-ray](https://user-images.githubusercontent.com/39334921/184656878-2da1a3a8-825e-481e-a972-1911b5df420f.png)

## Data Preparation
The data used in this project comes from different institution/repository
- NIH - obtained from https://tbportals.niaid.nih.gov/Datasharing
- TBx11k - https://www.kaggle.com/datasets/usmanshams/tbx-11
- shenzhen - https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen
- montegomery - https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-montgomery
- pediatric-pnemonia - https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray

| Source|TB | NTB | NORM|
|-------|---|-----|------|
|NIH|4287|-|-|
|TBX11K|800|3800|3800|
|pediatric-pnemonia|-|4273|1514|
|**Total Training**|**5087**|**8073**|**5314**|
|Shenzhen|336|-|326|
|Montegomery|58|-|80|
|**Total**|||5394|

# Model Development
## Classification
- In this project we used MobileNetv2-it has a much less number of parametes compared to deep learning model such as VGG. 
- It uses depthwise separable convolution to reduce number of learned parameters.

- It's pretrained version can be import using:
```
tf.keras.applications.MobileNetV2(input_tensor = inputs, weights="imagenet", include_top=False, alpha=0.35)
```

- The finally FC layers are cut and replaced with GlobalAveragePooling2D and the final output Dense layer with softmax
```
def build_model(input_shape):
    inputs = Input(shape=input_shape, name="input_image")
    mobilenetv2 = tf.keras.applications.MobileNetV2(
        input_tensor = inputs, 
        weights="imagenet", include_top=False, alpha=0.35)
    
    x = mobilenetv2.get_layer('out_relu').output
    x = GlobalAveragePooling2D(name='gap')(x)
    output = Dense(3,activation='softmax')(x)
    return tf.keras.Model(inputs,output)
```
***Confusion Matrixs***

![cm](https://github.com/degagawolde/DeepLearningBasedTBDiagnosis/blob/main/images/confusionmatrix.png)

***Result***

| Class|Accuracy|Recall|Precision|f1 score|
|---|----|------|---------|--------|
|All|98|98|98|98|
|TB |99|99|99|99|
|NTB|98|98|98|98|
|NORM|98|98|98|98|

## Segmentation
Chst x-ray images contains different parts of the chest that are not imprtant for lung disease diagnosis. The main purpose of this section is segmenting lung area from the rest of the cheast parts. **UNET** is used for semantic segmentation task. Hence, we constructed **UNET** using the mobilenetv2 as a backbone encoder, as expressed [here](https://github.com/nikhilroxtomar/Unet-with-Pretrained-Encoder/blob/master/U-Net_with_Pretrained_MobileNetV2_as_Encoder.ipynb?ref=morioh.com&utm_source=morioh.com). 
![image](https://github.com/nikhilroxtomar/Unet-with-Pretrained-Encoder/raw/5898a1e1ee66df875239d679839a30e419b20375//images/u-net-architecture.png)
1. first build the backbone encoder from mobilenetv2
```
def build_model(inputs):
    mobilenetv2 = tf.keras.applications.MobileNetV2(
        input_tensor = inputs, 
        weights="imagenet", include_top=False, alpha=0.35)
    mobilenetv2.trainable = False
    x = mobilenetv2.get_layer('out_relu').output
    x = Conv2D(128,3,name='final_conv',padding='same',activation='relu')(x)
    x = GlobalAveragePooling2D(name='gap')(x)
    output = Dense(2,activation='sigmoid')(x)
    return tf.keras.Model(inputs,output)
```

2. construct the unet
```
def model():
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="input_image")
    encoder = build_model(inputs)
    skip_connection_names = ["input_image","block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
    encoder_output = encoder.get_layer("block_13_expand_relu").output
    f = [ 16, 32, 48, 64]
    x = encoder_output
    for i in range(1, len(skip_connection_names)+1, 1):
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)
    
    model = Model(inputs, x)
    
    return model
```
**Result**
| Metrices|value|
|------|--------|
|Dice_Coef    |97.1|    
|Jackard_index|94.4|  


## Localization
- CAM
- GradCAM
- GradCAM++
- CMR
