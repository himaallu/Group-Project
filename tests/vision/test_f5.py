"""Tests for the F5 detector helpers (no image files or pretrained weights needed).

Run: python -m pytest tests/vision/test_f5.py -q
"""
import torch
from PIL import Image

from src.vision.detectors import build_model, corrupt
from src.vision.evaluate_detectors import CKPT_NAME


def test_corruptions_keep_an_rgb_image():
    image = Image.new("RGB", (100, 50), (120, 30, 200))
    assert corrupt(image, "crop20").size == (80, 40)        # 10% off each side
    assert corrupt(image, "jpeg50").mode == "RGB"
    assert corrupt(image, "clean") is image


def test_only_last_four_blocks_are_trainable():
    resnet = build_model("resnet50", pretrained=False)
    trainable = {n.split(".")[0] for n, p in resnet.named_parameters() if p.requires_grad}
    assert trainable == {"layer3", "layer4", "fc"}
    assert not any(p.requires_grad for p in resnet.layer3[0].parameters())
    vit = build_model("vit_b16", pretrained=False)
    layers = {n.split(".")[2] for n, p in vit.named_parameters()
              if p.requires_grad and n.startswith("encoder.layers")}
    assert layers == {"encoder_layer_8", "encoder_layer_9", "encoder_layer_10", "encoder_layer_11"}
    assert vit(torch.zeros(1, 3, 224, 224)).shape == (1, 1)


def test_checkpoint_names():
    match = CKPT_NAME.match("vit_b16_unseen_biggan_s42")
    assert (match["arch"], match["setting"], match["seed"]) == ("vit_b16", "unseen_biggan", "42")
    assert CKPT_NAME.match("resnet50_seen_s43")["setting"] == "seen"
