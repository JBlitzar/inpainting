Jan 28:
Today I tested without dropout. Much less blurry, but splotchy and dappled. Dropout 0.2 on every other convolutional layer will lead to only 0.26 of the image remaining! no wonder so bad. Retraining with no dropout.
Retrained. Works remarkably well, black areas are indistinguishable from other areas. Still blurry, training some more
Scary stuff with git lfs
the discoloration is real

Jan 30:
Does not generalize well to faces at different angles, farther away, non-centered
Learning curve has plateaued, but works relatively well