import numpy as np

from utils import loadModel, inputSize
from patchify import unpatchify


def detectShots(patches, predict_patch, model):
    input_size = inputSize()
    model = loadModel(input_size)
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i, j, :, :]
            image_list = [np.array(single_patch)]
            x = np.asarray(image_list)
            single_patch_prediction = model.predict(x)
            single_patch_prediction = single_patch_prediction[0, :, :, 0]
            predict_patch[i, j, :, :] = single_patch_prediction

    return predict_patch


def reconstruct(image_in_patches, shape):
    reconstructed_image = unpatchify(image_in_patches, shape)
    reconstructed_image = reconstructed_image.astype(np.uint8)

    return reconstructed_image