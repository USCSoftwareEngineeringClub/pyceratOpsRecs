# pyceratOpsRecs :crocodile:
[![Code Health](https://landscape.io/github/USCSoftwareEngineeringClub/pyceratOpsRecs/master/landscape.svg?style=flat)](https://landscape.io/github/USCSoftwareEngineeringClub/pyceratOpsRecs/master)
[![Build Status](https://travis-ci.org/USCSoftwareEngineeringClub/pyceratOpsRecs.svg?branch=master)](https://travis-ci.org/USCSoftwareEngineeringClub/pyceratOpsRecs)

## What this is

This is a basic OCR arithmetic calculator implemented in Python with Tensorflow.

## How to set up

We (:crocodile:) believe developers should spend most of their time developing, and not setting up their enviroment. For this reason, we strongly encourage you to use the `vagrantized` of this project.

1. Install Vagrant: https://www.vagrantup.com/downloads.html
1. cd /location/of/pyceratOpsRecs-master
1. run `vagrant up`
1. run `vagrant ssh`
1. ???
1. profit

### Getting an error for the TensorFlow Import?
1. run `vagrant up`
1. run `vagrant ssh`
1. run `sudo apt-get install python-pip python-dev`
1. run `export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc1-cp27-none-linux_x86_64.whl`
1. run `sudo pip install --upgrade $TF_BINARY_URL`

## How to use it

### How to run the current (10/20/16) version of the multi_cnn.py script
1. cd /location/of/pyceratOpsRecs-master
1. run `vagrant up`
1. run `vagrant ssh`
1. run `cd src`
1. run `python multi_cnn.py`
1. wait...

## How to contribute

### orcModel Branch
- [x] Fix train method in the Multi_Orc.py
- [ ] Fix the stub parameters in ORCModel.py
- [ ] Fix mnist.py to take in our own training images (might have to have multiple options)
- [ ] Write the run method in the Multi_Orc.py
- [ ] Add requirements to conda -- you can do this in scripts/install.sh
