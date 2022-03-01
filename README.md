# Real Time Gender Detection Web App

 A web interface for my real time gender detection project

## Introduction

This is a Web app using the model that I trained in my gender detection project. It uses flask and bootstrap for the web side of things, while OpenCV and tensorflow allow the real-time usage of the model previously mentioned. The model was developed on a CUDA enabled GPU, so to run this project, I would recommend you have the same: however it shouldn't be necessary.

### Prerequisites

Refer to the requirements.txt file for the requirements to run this project.
Running the following should be sufficient!
```
pip install -r requirements.txt
```

### Running the Project

Either a 
```
python  app.py
```
should be sufficient, or
```
export FLASK_APP=app.py && flask run
```
should also work

## Built With (to be added)

* [Flask](https://flask.palletsprojects.com/en/2.0.x/) - Micro web framework for python
* [Tensorflow](https://www.tensorflow.org) - low level deep learning API made by google
* [Keras](https://keras.io/) - high level Deep Learning API wrapper
* [CUDA](https://developer.nvidia.com/cuda-toolkit) - parallel computing platform and programming model developed by Nvidia for computing on GPUs
* [cuDNN](https://developer.nvidia.com/cudnn) - GPU accelerated library for Deep Neural Networks based on CUDA
* [OpenCV](https://opencv.org/) - a real-time optimized Computer Vision library
* [Boostrap](https://getbootstrap.com/) - a free and open-source CSS framework directed at responsive, mobile-first front-end web development
## Authors

* **Mashrur Khondokar**

## Acknowledgments

* Inspired by my best friend Connor, who indirectly pointed me towards flask.
