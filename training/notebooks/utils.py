import numpy as np
import skimage as sk
import torch
from PIL import Image

# --- Augmentation Classes ---
# Adapted from Microsoft TrOCR Repo:
# https://github.com/microsoft/unilm/blob/master/trocr/augmentation/noise.py

class GaussianNoise:
    def __call__(self, img, mag=-1, prob=1.0):
        if np.random.uniform(0, 1) > prob:
            return img
        b = [0.08, 0.1, 0.12]
        index = 0 if mag < 0 or mag >= len(b) else mag
        a = b[index]
        c = np.random.uniform(a, a + 0.03)
        img = np.array(img) / 255.0
        img = np.clip(img + np.random.normal(size=img.shape, scale=c), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))


class ShotNoise:
    def __call__(self, img, mag=-1, prob=1.0):
        if np.random.uniform(0, 1) > prob:
            return img
        b = [13, 8, 3]
        index = 2 if mag < 0 or mag >= len(b) else mag
        a = b[index]
        c = np.random.uniform(a, a + 7)
        img = np.array(img) / 255.0
        img = np.clip(np.random.poisson(img * c) / float(c), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))


class ImpulseNoise:
    def __call__(self, img, mag=-1, prob=1.0):
        if np.random.uniform(0, 1) > prob:
            return img
        b = [0.03, 0.07, 0.11]
        index = 0 if mag < 0 or mag >= len(b) else mag
        a = b[index]
        c = np.random.uniform(a, a + 0.04)
        img = sk.util.random_noise(np.array(img) / 255.0, mode="s&p", amount=c) * 255
        return Image.fromarray(img.astype(np.uint8))


class SpeckleNoise:
    def __call__(self, img, mag=-1, prob=1.0):
        if np.random.uniform(0, 1) > prob:
            return img
        b = [0.15, 0.2, 0.25]
        index = 0 if mag < 0 or mag >= len(b) else mag
        a = b[index]
        c = np.random.uniform(a, a + 0.05)
        img = np.array(img) / 255.0
        img = np.clip(img + img * np.random.normal(size=img.shape, scale=c), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))


class RandomNoise:
    def __init__(self, prob=0.25):
        self.prob = prob
        self.noises = [GaussianNoise(), ShotNoise(), ImpulseNoise(), SpeckleNoise()]

    def __call__(self, img):
        if np.random.uniform(0, 1) > self.prob:
            return img
        noise_fn = np.random.choice(self.noises)
        return noise_fn(img, mag=-1, prob=1.0)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.

    Copied from Hugging Face Transformers library (modeling_vision_encoder_decoder.py):
    https://github.com/huggingface/transformers
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids
