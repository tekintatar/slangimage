"""Compile-time contracts for the packaged ``py.typed`` surface."""

from typing import assert_type

import numpy as np

import slangpy as spy

import slangimage as si
from slangimage.fx import GaussianBlurEffect


def check_image_and_texture_contracts(
    image: si.Image,
    texture: spy.Texture,
    effect: GaussianBlurEffect,
) -> None:
    assert_type(si.ops.gaussian_blur(image, radius=2, sigma=1.0), si.Image)
    assert_type(si.ops.resize(image, scale=0.5), si.Image)
    assert_type(si.ops.blend(image, image), si.Image)
    assert_type(si.ops.invert(texture), spy.Texture)
    assert_type(si.ops.lut(texture, texture), spy.Texture)
    assert_type(effect.apply(image, blur_amount=2.0), si.Image)
    assert_type(effect.apply(texture, blur_amount=2.0), spy.Texture)
    assert_type(si.ops.random_noise(8, 8), si.Image)


def check_device_arguments(device: spy.Device, array: np.ndarray) -> None:
    assert_type(si.from_numpy(array, device=device), si.Image)
    assert_type(si.from_numpy(array, device="vulkan"), si.Image)
    assert_type(si.load("in.png", device=si.Backend.CUDA), si.Image)
