# Tensorflow-custom-object-detection-with-ip-camera

The project developed using TensorFlow object detection api to detect unauthorized objects(in this case mobiles) indoors using security cameras/ ip camera/ CCTV cameras... It works in both normal mode & in night vision mode.

A faster R-CNN pretrained model was used with custom dataset to train the detection model "rnn".

Python multi-threading library was used to optimize the rstp stream from the ip camera. A HIK Vision ip camera was used.

### Installation guides

* Use anaconda.<br>
* install tensorflow using : conda install tensorflow-gpu<br>
* It will take care of all the necessary drivers to run tensorflow-gpu ... <br>
* clone this repository. <br>
* install dependencies of tensorflow. follow this guide : <a href='https://github.com/AKNiloy/Tensorflow-custom-object-detection-with-ip-camera/blob/master/research/object_detection/g3doc/installation.md'>TF_INstall </a> <br>

* go to Tensorflow-custom-object-detection-with-ip-camera\research & run: <br>
protoc object_detection/protos/*.proto --python_out=. <br>
* then run: python setup.py build <br>
* & then : python setup.py install <br>
* object detection module will be ready to be deployed.

you need to install opencv & imutils library too <br>
pip install imutils <br>
pip install opencv-contrib-python <br>

go to: Tensorflow-custom-object-detection-with-ip-camera\research\object_detection <br>
**open ip_camera.py & edit it to give your own ipcamera's rstp address.**

**run : python od_ip.py** <br>
you should be good to go. <br>


![Alt Text](https://media.giphy.com/media/el7mDLdCjF5e7ZhioK/giphy.gif)



