# inpainting

- Understanding the data
- Visualizing reconstructions when training
- experiment with bottleneck size
- FC layers
- Run a real-life picture (not in the data) through the network
- multiple boxes
- different data
- Dropout
- Generally, where are the good datasets? Kaggle, (UCI repository), github,gutenberg https://www.v7labs.com/open-datasets

* next projects: image colorization, frame interpolation
* explore interpolation in the feature space
* medical stuff, object detection, object recog, segmentation
* find paper, try to reproduce
*

https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

# USING: https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256

IMAGENET(64x64): https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet/data

# USING: TINY IMAGENET http://cs231n.stanford.edu/tiny-imagenet-200.zip

https://github.com/unsplash/datasets

https://shotdeck.com/

https://shotdeck.com/browse/stills#

- https://paperswithcode.com/datasets?task=image-inpainting

https://paperswithcode.com/dataset/imagenet

- https://paperswithcode.com/dataset/unsplash-1k (used for the masks)

- https://www.kaggle.com/datasets/quadeer15sh/image-super-resolution-from-unsplash (used for the images)

https://github.com/JBlitzar/cnn-experiments

step 1: autoencoder ✔️ (mnist)

step 2: convolutional autoencoder ✔️ (imagenet)

reccommended convolutional layers getting smaller, flatten at bottom, conv getting bigger (make sure architecture is symmetric)

![image](structure.png)

All conv layers, dont flatten (still works, harder-to-work-with bottleneck)

step 3: inpainting

inpainting (content aware fill) input it to the autoencoder, reconstruct it with those features. Train it so that a box covering an area decodes to a complete image.

Traversing feature space?
