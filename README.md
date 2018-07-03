
# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[raw_dist]: ./graph_sup/raw_dist.png "raw_dist"
[balanced]: ./graph_sup/balanced_dist.png "balanced_dist"
[augumentation]: ./graph_sup/augumentation_sample.png "augumentation_sample"
[preprocessed_image]: ./graph_sup/preprocessed_image.png "preprocessed_image"
[model_structure]: ./graph_sup/model_structure.png "model_structure"
[val_loss]: ./graph_sup/val_loss.png "val_loss"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* Behavioral_Cloning_Model.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The Behavioral_Cloning_Model.ipynb file contains the code for data balance, augumentation, data preprocess, training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The following graph showed the model architecture (Nvidia CNN architecture) I have used in this project. 
* To train the model faster and robust, the Elu was chosen as activation function. 
* The dropout layer was added to the end of the convolusion layer to prevent overfitting. 
* The Adam optimizer used was chosen, so the learning rate was not tuned manually. 
* The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. The details of the model architecture shown below. 

![model_structure][model_structure]

#### 2. Creation of the Training Set & Training Process

First, I tried to collect the data myself on simulator. However, I could not succussfully collect the "smooth" data. In my data collection, the steering angle is not constant which means that the steering angle is 0 when it is supposed to be the same as last frame (negative steering angle). In this case, the similiar frame with different steering angle will confuse the model. However, the Udacity has provided the simple dataset which I used in this project. 

Each data record will contains three images from different angle (center, left, right). To simulate my vehicle being in different positions, the steering angle corrections are added to the image from left and right camera (+0.25 and -0.25 respectively). 

As the figure shown below, the sample dataset looks very skewed, the samples for steering angle 0, 0.25 and -0.25 are significant higher than the other samples. This will inject 'straight bias' to the model so that the model will not perform well on turning events (especially in Track 2). So, the data balance need to be applied before training.


![alt text][raw_dist]

The dynamic data balance stratage was used below. The higher absolute steering angle, the higher chance to be keeped in training set.The following figure shows the simple distribution after data balance.

![alt text][balanced]

For purpose of recovering and provide robust model. The jittered image are generated for training. My data augumentation incluses the following steps. The following stratages are randomly used based on probability. 

* Random Image flip 
* Random Image translation
* Random Image brightness 
* Random Image shadow 
* Random jitter training image

The following figure shows the jitter effects for training dataset. 

![augumentation][augumentation]

The data preprocessing is based on the Nvidia self driving car paper. The input images are resize to (66, 200, 3) with YUV color channel type. Shown below.

![alt text][preprocessed_image]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 30 as evidenced by volidation loss curve.
![alt text][val_loss]


Finally, the test video shows that the vehicle can successfully drive on the track1. Also, the vehicle can adjust the position when it drive close to the road edge. However, my model does not work well on track2 since there are more sharp curve in track2. I think collect more data on the curve for training model will be very helpful to solve this problem. 

