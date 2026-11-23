import io
import pickle

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.vision.detectors import (CachedImages, build_model, eval_transform, freeze_batchnorm, frozen_batchnorms,
                                  load_cache, restore_imagenet_batchnorm)
from src.vision.f10_stats import choose_cutoff, paired_bootstrap
from src.vision.leakage_cv import MODELS, alr_table, changed, cross_validate, mcnemar


def test_frozen_batchnorms_are_the_untrained_ones():
    model = build_model("resnet50", pretrained=False)
    names = frozen_batchnorms(model)
    assert "bn1" in names and "layer1.0.bn1" in names and "layer3.4.bn3" in names
    assert not any(n.startswith(("layer3.5", "layer4")) for n in names)


def test_freeze_batchnorm_keeps_frozen_statistics_during_training():
    torch.manual_seed(0)
    model = build_model("resnet50", pretrained=False)
    before = model.layer1[0].bn1.running_mean.clone()
    trainable_before = model.layer4[0].bn1.running_mean.clone()
    model.train()
    freeze_batchnorm(model)
    model(torch.randn(2, 3, 64, 64) * 3 + 1)
    assert torch.equal(model.layer1[0].bn1.running_mean, before)
    assert not model.layer1[0].bn1.training and model.layer4[0].bn1.training
    assert not torch.equal(model.layer4[0].bn1.running_mean, trainable_before)


def test_without_freeze_the_frozen_statistics_drift():
    model = build_model("resnet50", pretrained=False).train()
    before = model.layer1[0].bn1.running_mean.clone()
    model(torch.randn(2, 3, 64, 64) * 3 + 1)
    assert not torch.equal(model.layer1[0].bn1.running_mean, before)


def test_restore_imagenet_batchnorm_only_touches_frozen_layers():
    model = build_model("resnet50", pretrained=False)
    reference = {k: v.clone() + 1 if k.endswith(("running_mean", "running_var")) else v.clone()
                 for k, v in model.state_dict().items()}
    trainable_before = model.layer4[0].bn1.running_mean.clone()
    names = restore_imagenet_batchnorm(model, reference)
    assert "layer1.0.bn1" in names
    assert torch.equal(model.layer1[0].bn1.running_mean, reference["layer1.0.bn1.running_mean"])
    assert torch.equal(model.layer4[0].bn1.running_mean, trainable_before)


def test_cache_reads_images_in_manifest_order(tmp_path):
    blobs, md5s = [], []
    for i in range(10):
        b = io.BytesIO()
        Image.new("RGB", (96, 96), (i * 20, 0, 0)).save(b, format="JPEG")
        blobs.append(b.getvalue())
        md5s.append(f"m{i}")
    path = tmp_path / "cache.pkl"
    with open(path, "wb") as f:
        pickle.dump({"md5": md5s, "jpeg": blobs, "jpeg_settings": 1}, f)
    picked = load_cache(path, ["m3", "m1"])
    assert picked == [blobs[3], blobs[1]]
    x, y, i = CachedImages(picked, [1.0, 0.0], eval_transform())[0]
    assert x.shape == (3, 224, 224) and y.item() == 1.0 and i == 0


def test_cutoff_is_chosen_on_validation_with_ties_nearest_half():
    y = np.array([0, 0, 1, 1])
    assert choose_cutoff(np.array([0.1, 0.2, 0.3, 0.4]), y) == 0.3
    assert choose_cutoff(np.array([0.0, 0.0, 1.0, 1.0]), y) == 0.5


def test_paired_bootstrap_detects_a_better_model():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(100), np.ones(100)].astype(int)
    weak, strong = y + rng.normal(0, 2, 200), y + rng.normal(0, 0.3, 200)
    r = paired_bootstrap(y, weak, strong, n_boot=200)
    assert r["difference"] > 0 and r["difference_low"] > 0 and r["p_bootstrap"] < 0.05


def fake_captions(n=10):
    rng = np.random.default_rng(1)
    rows = []
    for k in range(n):
        for label in ("true", "miscaptioned"):
            rows.append({"image_file": f"p{k}.jpg", "label": label, "y": int(label == "true"), "clip_index": k,
                         "split": "train", "sim_original": rng.normal(0.1 if label == "true" else 0.0, 0.05),
                         "sim_twin": rng.normal(0.1 if label == "true" else 0.0, 0.05),
                         "original_authenticity_score": rng.uniform(), "twin_authenticity_score": rng.uniform()})
    return pd.DataFrame(rows), rng.normal(size=(n, 32)), rng.normal(size=(n, 32))


def test_cross_validation_scores_every_photo_once_per_model_and_version():
    cap, orig, twin = fake_captions()
    pairs = cross_validate(cap, orig, twin, seed=42, folds=5, components=4)
    counts = pairs.groupby(["model", "version"]).size()
    assert (counts == 10).all() and len(counts) == len(MODELS) * 2
    assert pairs.groupby("image_file")["fold"].nunique().max() == 1


def test_alr_and_mcnemar_count_changed_verdicts():
    rows = []
    for k, (dec, fus) in enumerate([(False, True), (True, True), (False, False), (False, True)]):
        for model, flip in (("decoupled", dec), ("fused", fus), ("fused + image embedding", fus)):
            rows += [{"model": model, "version": "original", "image_file": f"p{k}", "picks_true_caption": True},
                     {"model": model, "version": "twin", "image_file": f"p{k}", "picks_true_caption": not flip}]
    pairs = pd.DataFrame(rows)
    assert changed(pairs, "fused").sum() == 3
    table = alr_table(pairs, n_boot=50).set_index("model")
    assert table.loc["decoupled", "ALR_percent"] == 25 and table.loc["fused", "ALR_percent"] == 75
    test = mcnemar(pairs)
    assert (test["only_a"], test["only_b"], test["both"]) == (0, 2, 1)
