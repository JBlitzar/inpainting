# inpainting

https://shotdeck.com/

https://paperswithcode.com/datasets?task=image-inpainting

https://paperswithcode.com/dataset/imagenet

https://paperswithcode.com/dataset/unsplash-1k

https://github.com/JBlitzar/cnn-experiments


step 1: autoencoder (non-image data)

step 2: convolutional autoencoder:

reccommended convolutional layers getting smaller, flatten at bottom, conv getting bigger (make sure architecture is symmetric)

All conv layers, dont flatten (still works, harder-to-work-with bottleneck)

step 3: inpainting

inpainting (content aware fill) input it to the autoencoder, reconstruct it with those features. Train it so that a box covering an area decodes to a complete image.

Traversing feature space?
