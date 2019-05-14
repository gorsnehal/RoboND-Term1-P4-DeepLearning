[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Deep Learning Project ##

Following report explains technical details & learnings out of FollowMe project. Major aspects that will be covered here are
1. Implement the Semantic Segmentation Network
   a. Fully Convolutional Neural Network (FCN)
   b. Encode / Convolution layer
   c. 1x1 Convolution
   d. Decode layer
2. Parameters Selection
3. Data Collection
4. Results

[image_0]: ./Images/sim_screenshot.png
![alt text][image_0] 


## 1. Implement the Segmentation Network

**a. Fully Convolutional Neural Network**

- In Traditional Neural net used for object classification, Convolution layers are followed by Fully connected Neural network.
- Problem with that architecture is it do not preserves spatial information. So not capable at identifying for example where in the image particular object is. Also due to fully connected neural network at the end, it's not capable at handling different size images.
- Fully convolutional neural network replaces the Fully connected layers with convolutional layers. That's why Fully Convolutional Neural Network (FCN).
- FCN due to use of convolutional layers preserves spatial information & also capable of handling different size images
- My architecture looks like below

[image_1]: ./Images/FCN_Architecture.jpg
![alt text][image_1] 

In following few sections I will further describe individual blocks

**b. Encode / Convolution Layer**

- My very crude view - Convolutional layers are essentially feature extractors. Which combined with weights & trained through backpropogation capable of learning different patterns
(Somewhat correlated with Cascade Adaboost / SVM (traditional machine learning), where each individual weak learner is classifying in N dimensional feature space......Though CNN & DNN are much deeper & combined with huge data & processing power, achieving much more than traditional machine learning)
- Each successive convolution layer build on lower layers features to successively learn more & more complex features
- Encoding part implemented below

```
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```

- `Separable Convolutional`: Separating traditional convolution (3D) into 2D convolution (on each channel of input) followed by 1x1 convolution. Advantage of doing this is huge saving on number of parameters. Ex. For my case with input size of 160 x 160 x 3 & 3 x 3 x 3 kernel below traditional vs. separable convolution compared for 1st convolution (160 x 160 x 32)
    `Normal Convolutional - 32 x 3 x 3 x 3 = 864`
    `Separable Convolutional - 3 x 3 x 3 + 32 x 3 = 123`
   
- `Batch Normalization`: Normalizes each inputs of each layer. Advantage of doing this is, better regularization & faster & better training for deeper networks

**c: 1x1 Convolution**

`1x1 convolution:`

- As explained above in traditional neural net architecture, due to using fully connected layer spatial information has been lost. To avoid it 1 x 1 convolution has been used.
- 1 x 1 convolution is basically convolving with kernel size 1 x 1 & stride 1
```
encodl = conv2d_batchnorm(cnvl3, filters = num_classes, kernel_size = 1)
```

**d: Decode Layer**

- Decoders are fundamentally projecting learned features above to successively higher resolution representation
- One important part here is a skip connection. 

```
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    bl_layer = bilinear_upsample(small_ip_layer)
    
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    concat_layer = layers.concatenate([bl_layer, large_ip_layer])
    
    # TODO Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(concat_layer, filters)
    
    return output_layer
```

`Skip Connections:` 
- As you can see in above FCN architecture, We are successively encoding information in smaller spatial region. To improve the multi-resolution capabilities, skip connections are being added. Whereby Encoder layers output is being concatenated to Decoder layer.


**My Final FCN Network**

```
def fcn_model(inputs, num_classes):
    
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    cnvl0 = encoder_block(inputs, filters = 128, strides = [1, 1])
    cnvl1 = encoder_block(cnvl0, filters = 256, strides = [2, 2])
    cnvl2 = encoder_block(cnvl1, filters = 512, strides = [2, 2])
    cnvl3 = encoder_block(cnvl2, filters = 1024, strides = [2, 2])
    #cnvl4 = encoder_block(cnvl3, filters = 512, strides = [2, 2])

    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    encodl = conv2d_batchnorm(cnvl3, filters = num_classes, kernel_size = 1)
    
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    #decd3 = decoder_block(encodl, cnvl3, filters=cnvl3.get_shape().as_list()[-1])
    decd2 = decoder_block(encodl, cnvl2, filters=cnvl2.get_shape().as_list()[-1])
    decd1 = decoder_block(decd2, cnvl1, filters=cnvl1.get_shape().as_list()[-1])
    decdl = decoder_block(decd1, cnvl0, filters=cnvl0.get_shape().as_list()[-1])
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(decdl)
```


## 2. Parameters Selection

Below I'm describing how I tuned different parameters
1. Steps per epoch, validation step & workers I kept the default values
2. Batch size kept as per database size
3. First version of the network (as shown above), trained with 32 filters at first convolution & successive convolution increased depth 64 -> 128 -> 256. With this network best score, I got is around 0.4
4. Since errors on smaller size objects (hero & other objects) were high, I made following changes
   a. Added `more images` with very far hero & other pedestrians
   b. I added `more filters` to capture more variation / features, making it four times (in 2 steps) the first version. with         this depths of the network (Encoder part) became 128 -> 256 -> 512 -> 1024. With this network, I got score around 0.457

[image_2]: ./Images/Loss.png
![alt text][image_2]

5. Number of epochs decided as per `bias - variance` analysis. As I didn't observe larger variance between training & validation set, further images not being added.
6. I also tried `going deeper with more convolution layer`, but observed that it has not improved performance that much. I believe it could be due to while going deeper I also reduced spatial resolution. Which didn't help a problem of smaller object detection.


## 3. Data Collection

Although I have collected multiple data sets, on observing performance data. I decided to utilize far patterns images. I have added around 600 images with far hero / other pedestrian patterns.

Although I do feel that with addition of more data & extensive training, performance can be improved further.

## 4. Results

```
Weight value as per TP/FP & FN : 0.783
Final IOU                      : 0.583
Final Score                    : 0.4574
```

[image_3]: ./Images/0_run1cam1_00010_prediction.png
![alt text][image_3]

[image_4]: ./Images/0_run1cam1_00018_prediction.png
![alt text][image_4]

[image_5]: ./Images/2_run2cam1_00017_prediction.png
![alt text][image_5]

[image_6]: ./Images/2_run2cam1_00131_prediction.png
![alt text][image_6]

[image_7]: ./Images/2_run2cam1_00296_prediction.png
![alt text][image_7]

[image_8]: ./Images/2_run2cam1_00851_prediction.png
![alt text][image_8]


## Thoughts on Improving it further & using for other problems :)

- I feel I can try with addition of few more layers, but with stride of 1
- Addition of more data should improve performance
- For identifying cat or dog instead of a human / pedestrian in images. Same network cannot be reused entirely, but initial encoders layers can be reused. Following are few concerns
  a. As successive convolution layers learns more & more complex features, may be deeper layers requires some fine tuning or can be discarded
  b. New database size, learning rate & similarity with human database, can become a factor for re-training
  
**Clarifying above comments as per Reviewer comments**

What my understanding is, above network can be used but not entirely as it is. I have following views
1. As each convolution layer learns successively complex patterns, later layers (encoding) will learn features related to human / pedestrian. (Reference Lesson 33 - Section 6)
2. So later layers has to be retrained for car, cat, dog patterns, while initial layers which are learning general features like edges, color, etc. can be reused as is
3. Also as we are re-using the network which is already trained, may be learning rate required to be low. Because we don't want to disturb it too much
4. Further as it's already optimal, my understanding is it should work for car, cat or dog patterns with smaller database & retuning

I hope above clarifies my points further. Please share reference, if my understanding is not correct.
