# Deep Learning Based Chest X-ray Diagnosis

[Chext x-ray](https://www.mayoclinic.org/tests-procedures/chest-x-rays/about/pac-20393494) produce images of organs located under the chest(heart, lungs, blood vessels, airways, and the bones of chest and spine). Chest X-rays can also reveal fluid in or around the lungs or air surrounding a lung
is a chest radiography used for the diagnosis for diseases attacking lung. Radiologists can look at the image and examine if there is anomalies in the x-ray image. In this project, we traiend a deep learning model- [MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2/MobileNetV2) for the diagnosis of the chest x-ray images.

![chest x-ray](https://user-images.githubusercontent.com/39334921/184656878-2da1a3a8-825e-481e-a972-1911b5df420f.png)

## Data Preparation
The data used in this project comes from different institution/repository
- NIH
- TBx11k
- shenzhen
- montegomery
- pediatric-pnemonia

| Source|TB | NTB | NORM|
|-------|---|-----|------|
|NIH|4287|-|-|
|TBX11K|800|3800|3800|
|pediatric-pnemonia|-|4273|1514|
|**Total Training**|**5087**|**8073**|**5314**|
|Shenzhen|336|-|326|
|Montegomery|58|-|80|
|**Total**|||5394|

## Model Development

In this project we used MobileNetv2-it has a much less number of parametes compared to deep learning model such as VGG. It uses depthwise separable convolution to reduce number of learned parameters.

It's pretrained version can be import using:
```
tf.keras.applications.MobileNetV2(input_tensor = inputs, weights="imagenet", include_top=False, alpha=0.35)
```

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

## Result of classification

## Result of Segmentation

## Localization
