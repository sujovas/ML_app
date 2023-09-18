import os

import unet_with_batch_normalization

def inputSize():
    patch_size = 256
    IMG_HEIGHT = patch_size
    IMG_WIDTH = patch_size
    IMG_CHANNELS = 1
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    return input_shape


def loadModel(inputSize):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    weights_path = os.path.join(parent_dir, 'model_unet_focal_tv_in')

    model = unet_with_batch_normalization.build_unet(inputSize, 1)
    model.load_weights(weights_path)

    return model
