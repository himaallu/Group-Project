# GenImage benchmark and generalisable AI-generated image detectors (prior work for TwinCheck MAAI901 side)

Scope: (a) the GenImage paper itself; (b) later detectors evaluated on GenImage (UnivFD/CLIP probes, CNN vs ViT backbones, DIRE, GenDet, PatchCraft, AIDE, NPR, C2P-CLIP, DRCT, B-Free, DDA); (c) the JPEG/PNG format and image-size biases in GenImage and what happens after debiasing.

Papers covered (title, authors, venue, year, URL):
1. GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image. Zhu, Chen, Yan, Huang, Lin, Li, Tu, Hu, Hu, Wang (Huawei Noah's Ark Lab). NeurIPS 2023 Datasets and Benchmarks (Advances in NeurIPS 36:77771–77782, citation as given in the DDA reference list). arXiv 2306.08571. https://arxiv.org/abs/2306.08571 ; code/data https://github.com/GenImage-Dataset/GenImage
2. Towards Universal Fake Image Detectors that Generalize Across Generative Models (UnivFD). Ojha, Li, Lee. CVPR 2023. https://arxiv.org/abs/2302.10174
3. Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets. Grommelt, Weiss, Pfreundt, Keuper. arXiv preprint 2024 (no peer-reviewed venue found). https://arxiv.org/abs/2403.17608 ; project https://www.unbiased-genimage.org/
4. A Sanity Check for AI-generated Image Detection (AIDE + Chameleon dataset). Yan, Li, Cai, Hao, Jiang, Hu, Xie. ICLR 2025 (per authors' repo). https://arxiv.org/abs/2406.19435 ; https://github.com/shilinyan99/AIDE
5. Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection (NPR). Tan, Liu, Zhao, Wei, Gu, Liu, Wei. CVPR 2024. https://arxiv.org/abs/2312.10461
6. A Bias-Free Training Paradigm for More General AI-generated Image Detection (B-Free). Guillaro, Zingarini, Usman, Sud, Cozzolino, Verdoliva. CVPR 2025, pp. 18685–18694. https://arxiv.org/abs/2412.17671 ; https://openaccess.thecvf.com/content/CVPR2025/papers/Guillaro_A_Bias-Free_Training_Paradigm_for_More_General_AI-generated_Image_Detection_CVPR_2025_paper.pdf
7. Aligned Datasets Improve Detection of Latent Diffusion-Generated Images. Rajan, Ojha, Schloesser, Lee. 2025 (arXiv 2410.11835 v3, Feb 2025; reported as ICLR 2025, venue not confirmed on the arXiv page). https://arxiv.org/abs/2410.11835
8. Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable (DDA). Chen, Xi, Yan, Zhang, Wu, Xie, Chen, Xu, Guan, Yao, Ding. NeurIPS 2025 (poster listed at https://neurips.cc/virtual/2025/poster/119323). https://arxiv.org/html/2505.14359v6

---

## Q1. What are GenImage's protocol and headline results (train-on-SD-v1.4 / test-on-others, BigGAN generalisation)?

### Takeaway
GenImage pairs all 1.33M ImageNet real images with 1.35M fakes from 8 generators (Midjourney V5, SD V1.4, SD V1.5, ADM, GLIDE, Wukong, VQDM, BigGAN) and defines two tasks: cross-generator classification (standard: train on SD V1.4, test on all 8) and degraded-image classification. In-generator accuracy is ~99%, but SD-V1.4-trained detectors collapse to near chance on Midjourney, ADM, VQDM and BigGAN (e.g. ResNet-50: 54.9 / 53.5 / 56.6 / 52.0%), giving averages of only 64–75% across the 7 detectors tested.

### Cited Findings
**Dataset construction**
- 2,681,167 images: 1,331,167 real and 1,350,000 fake. Real = all of ImageNet (1,281,167 train, 50,000 test). Fakes use the 1,000 ImageNet classes, 1,350 images per class (1,300 train, 50 test) — [GenImage, arXiv 2306.08571](https://ar5iv.labs.arxiv.org/html/2306.08571)
- Eight generators: BigGAN, GLIDE, VQDM, Stable Diffusion V1.4, SD V1.5, ADM, Midjourney (V5), Wukong. Each generator produces 162 train + 6 test images per class (SD V1.5: 166 train + 8 test). Each generator plus its own set of real images forms a "subset"; real images are not shared across subsets — [GenImage](https://ar5iv.labs.arxiv.org/html/2306.08571)
- Prompts follow the template "photo of [class]" (translated to Chinese for Wukong); ADM and BigGAN use their ImageNet-pretrained class-conditional models — [GenImage](https://ar5iv.labs.arxiv.org/html/2306.08571)
- Fake resolutions: Midjourney 1024×1024; Wukong, SD V1.4, SD V1.5 512×512; GLIDE and VQDM 256×256; BigGAN 128×128 (ADM 256×256 per Grommelt et al.) — [GenImage](https://ar5iv.labs.arxiv.org/html/2306.08571); [Fake or JPEG?](https://arxiv.org/html/2403.17608v2)
- Directory layout per generator: `<Generator>/{train,val}/{ai,nature}`; a full-dataset training folder `imagenet_ai` merges all generators. Data on Baidu Yunpan and (since 15 Mar 2024) Google Drive — [GenImage GitHub](https://github.com/GenImage-Dataset/GenImage)
- Humans reach only 61.3% accuracy at telling real from AI-generated images (cited by GenImage from prior work) — [GenImage](https://ar5iv.labs.arxiv.org/html/2306.08571)

**Detectors tested**
- Backbones trained from scratch as binary classifiers: ResNet-50 (CNN), DeiT-S and Swin-T (Transformers). Face-forgery detectors: F3Net (frequency-aware decomposition + local frequency statistics), GramNet (global texture). General fake-image detectors: CNNSpot (ResNet-50 with blur σ~U[0,3] and JPEG augmentation, each with 50% probability) and Spec (frequency spectrum input) — [GenImage](https://ar5iv.labs.arxiv.org/html/2306.08571)
- The original GenImage paper does NOT evaluate UnivFD or DIRE; those numbers on GenImage come from later papers (e.g. AIDE's table below). The authors later released GenDet (arXiv 2312.08880) as their own detector — [GenImage GitHub](https://github.com/GenImage-Dataset/GenImage); [AIDE](https://arxiv.org/html/2406.19435)

**Task 1 – Cross-generator results (Table 3: train on SD V1.4, accuracy %)**

| Method | MJ | SD1.4 | SD1.5 | ADM | GLIDE | Wukong | VQDM | BigGAN | Avg |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | 54.9 | 99.9 | 99.7 | 53.5 | 61.9 | 98.2 | 56.6 | 52.0 | 72.1 |
| DeiT-S | 55.6 | 99.9 | 99.8 | 49.8 | 58.1 | 98.9 | 56.9 | 53.5 | 71.6 |
| Swin-T | 62.1 | 99.9 | 99.8 | 49.8 | 67.6 | 99.1 | 62.3 | 57.6 | 74.8 |
| CNNSpot | 52.8 | 96.3 | 95.9 | 50.1 | 39.8 | 78.6 | 53.4 | 46.8 | 64.2 |
| Spec | 52.0 | 99.4 | 99.2 | 49.7 | 49.8 | 94.8 | 55.6 | 49.8 | 68.8 |
| F3Net | 50.1 | 99.9 | 99.9 | 49.9 | 50.0 | 99.9 | 49.9 | 49.9 | 68.7 |
| GramNet | 54.2 | 99.2 | 99.1 | 50.3 | 54.6 | 98.9 | 50.8 | 51.7 | 69.9 |

  — [GenImage](https://ar5iv.labs.arxiv.org/html/2306.08571); identical in [GitHub README](https://github.com/GenImage-Dataset/GenImage)
- Full 8×8 ResNet-50 cross-validation (Table 2): in-generator accuracy ≥98.8% for every subset; BigGAN-trained ResNet-50 scores 49.9% on MJ, SD1.4, SD1.5 and Wukong (avg 58.6%); VQDM-trained avg 59.7%; GLIDE-trained 60.2%; SD V1.4-trained gives the best average (72.1%), so SD V1.4 was chosen as the standard training subset. ADM-trained transfers well to GLIDE (97.1%) and BigGAN (88.3%) — [GenImage](https://ar5iv.labs.arxiv.org/html/2306.08571)
- Table 4 (8 models × 8 test sets, 64 results averaged): ResNet-50 66.9%, DeiT-S 67.6%, Swin-T 70.3%, CNNSpot 61.7%, Spec 65.1%, F3Net 64.6%, GramNet 64.7% — [GenImage](https://ar5iv.labs.arxiv.org/html/2306.08571)
- Authors' conclusions: CNN (ResNet-50) and Transformers (DeiT-S, Swin-T) perform similarly; Swin-T is best; CNNSpot and Spec, designed for GANs, are worse than plain backbones on this diffusion-dominated data; generators with similar architectures (SD V1.4, SD V1.5, Wukong) transfer to each other; Midjourney is the hardest to generalise to — [GenImage](https://ar5iv.labs.arxiv.org/html/2306.08571)
- Frequency analysis: BigGAN fakes show regular grid artefacts in the averaged spectrum; diffusion fakes and real ImageNet images show few artefacts, so diffusion images are harder to detect — [GenImage](https://ar5iv.labs.arxiv.org/html/2306.08571)
- Data scale: ResNet-50 trained on 1,600 SD V1.4 images gets 73.6% in-generator / 60.9% cross-generator; with 162,000 images 99.9% / 72.1% — [GenImage](https://ar5iv.labs.arxiv.org/html/2306.08571)

**Task 2 – Degraded images (Table 5: trained on SD V1.4, only test images degraded)**

| Method | LR 112 | LR 64 | JPEG q=65 | JPEG q=30 | Blur σ=3 | Blur σ=5 | Avg |
|---|---|---|---|---|---|---|---|
| ResNet-50 | 96.2 | 57.4 | 51.9 | 51.2 | 97.9 | 69.4 | 70.6 |
| DeiT-S | 97.1 | 54.0 | 55.6 | 50.5 | 94.4 | 67.2 | 69.8 |
| Swin-T | 97.4 | 54.6 | 52.5 | 50.9 | 94.5 | 52.5 | 67.0 |
| CNNSpot | 50.0 | 50.0 | 97.3 | 97.3 | 97.4 | 77.9 | 78.3 |
| Spec | 50.0 | 49.9 | 50.8 | 50.4 | 49.9 | 49.9 | 50.1 |
| F3Net | 50.0 | 50.0 | 89.0 | 74.4 | 57.9 | 51.7 | 62.1 |
| GramNet | 98.8 | 94.9 | 68.8 | 53.4 | 95.9 | 81.6 | 82.2 |

  — [GenImage](https://ar5iv.labs.arxiv.org/html/2306.08571)
- Authors note that JPEG compression "has a significant impact" on the backbones, and CNNSpot's robustness to JPEG and blur comes from using them as training augmentation — [GenImage](https://ar5iv.labs.arxiv.org/html/2306.08571)

### Inferences
- In the GenImage paper, BigGAN is a hard generalisation target (SD-V1.4-trained detectors get 46.8–57.6%), and BigGAN-trained detectors do not transfer to diffusion models at all. So "hold one generator out" results depend a lot on which generator is held out; TwinCheck's F1/F3 plan should report which generator is unseen and preferably pick one from a different family (e.g. train SD/Wukong, hold out Midjourney or BigGAN).
- The JPEG collapse of plain backbones (≈51% at q=65) is far worse than their blur robustness. Combined with Q3, this points to a learned "JPEG = real" shortcut rather than general fragility.

### Gaps
- I read the arXiv v1 preprint (marked "Preprint. Under review.") and the GitHub README. Their tables are identical, but I could not load the NeurIPS camera-ready PDF (timeout), so any extra detectors added in the camera-ready version were not checked.
- The GenImage paper does not state the file formats (JPEG vs PNG) of the real vs fake images; that comes from Grommelt et al. (Q3).

---

## Q2. How do CNN (ResNet-50) and ViT/CLIP-based detectors compare on unseen generators, and how robust are they to JPEG, resizing and cropping?

### Takeaway
On the standard GenImage SD-V1.4 protocol, ImageNet-style CNNs and ViTs trained end to end score about the same (72–75% average) because they overfit SD artefacts. A frozen-CLIP linear probe (UnivFD) is less accurate on SD itself (84%) but generalises better to Midjourney (73.2%) and BigGAN (80.3%). Hybrid or artefact-focused methods push the average to 82–87% (GenDet 81.56, PatchCraft 82.30, AIDE 86.88). Under JPEG compression every method drops sharply, but the CLIP-based UnivFD loses less than low-level artefact methods (LGrad, LNP fall to ≈50–54%).

### Cited Findings
**UnivFD (why frozen CLIP)**
- Training a deep network for real-vs-fake on GAN images "fails to detect fake images from newer breeds of generative models". The classifier becomes "asymmetrically tuned to detect patterns that make an image fake. The real class becomes a sink class holding anything that is not fake, including generated images from models not accessible during training" — [UnivFD, Ojha et al., CVPR 2023](https://arxiv.org/abs/2302.10174)
- Proposed fix: "real-vs-fake classification without learning", i.e. using a feature space not explicitly trained to separate real from fake (a large pretrained vision-language model, CLIP ViT-L/14), with nearest-neighbour and linear probing. Nearest neighbour improves on the SoTA by +15.07 mAP and +25.90% accuracy on unseen diffusion and autoregressive models — [UnivFD](https://arxiv.org/abs/2302.10174)
- (From background knowledge, not re-verified in this session: UnivFD is trained on ProGAN images, following the CNNSpot/ForenSynths setup.)

**Expanded GenImage table (train SD V1.4; from AIDE, ICLR 2025, Table on GenImage; rows 1–7 copied from GenImage)**

| Method | MJ | SD1.4 | SD1.5 | ADM | GLIDE | Wukong | VQDM | BigGAN | Mean |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | 54.90 | 99.90 | 99.70 | 53.50 | 61.90 | 98.20 | 56.60 | 52.00 | 72.09 |
| DeiT-S | 55.60 | 99.90 | 99.80 | 49.80 | 58.10 | 98.90 | 56.90 | 53.50 | 71.56 |
| Swin-T | 62.10 | 99.90 | 99.80 | 49.80 | 67.60 | 99.10 | 62.30 | 57.60 | 74.78 |
| DIRE | 60.20 | 99.90 | 99.80 | 50.90 | 55.00 | 99.20 | 50.10 | 50.20 | 70.66 |
| UnivFD | 73.20 | 84.20 | 84.00 | 55.20 | 76.90 | 75.60 | 56.90 | 80.30 | 73.29 |
| GenDet | 89.60 | 96.10 | 96.10 | 58.00 | 78.40 | 92.80 | 66.50 | 75.00 | 81.56 |
| PatchCraft | 79.00 | 89.50 | 89.30 | 77.30 | 78.40 | 89.30 | 83.70 | 72.40 | 82.30 |
| AIDE | 79.38 | 99.74 | 99.76 | 78.54 | 91.82 | 98.65 | 80.26 | 66.89 | 86.88 |

  — [AIDE, Yan et al., ICLR 2025](https://arxiv.org/html/2406.19435)
- AIDE combines a CLIP (ConvNeXt-based OpenCLIP) semantic branch with a DCT-scored branch that picks the highest- and lowest-frequency patches and passes them through SRM filters. The abstract claims +4.6% over SoTA on GenImage and +3.5% on AIGCDetectBenchmark — [AIDE](https://arxiv.org/abs/2406.19435); venue [AIDE GitHub ("accepted by ICLR 2025")](https://github.com/shilinyan99/AIDE)
- On AIDE's new Chameleon dataset (AI images that are hard for humans to spot), most off-the-shelf detectors (CNNSpot, FreDect, Fusing, GramNet, LNP, UnivFD, DIRE, PatchCraft, NPR, AIDE) score close to chance and misclassify fakes as real. AIDE itself gets 58.37% (trained on ProGAN), 62.60% (SD V1.4) and 65.77% (all of GenImage) — [AIDE](https://arxiv.org/html/2406.19435)

**Robustness (AIDE Table 6: average accuracy over 16 test sets of AIGCDetectBenchmark, not GenImage; reported as given)**

| Method | Orig | JPEG95 | JPEG90 | JPEG75 | JPEG50 | Blur σ1 | σ2 | σ3 | σ4 |
|---|---|---|---|---|---|---|---|---|---|
| CNNSpot | 70.78 | 64.03 | 62.26 | 60.65 | 59.66 | 68.39 | 67.26 | 67.13 | 65.85 |
| LGrad | 75.34 | 51.55 | 51.39 | 50.00 | 50.00 | 71.73 | 69.12 | 68.43 | 66.22 |
| LNP | 83.84 | 53.58 | 54.09 | 53.02 | 52.85 | 67.91 | 66.42 | 66.2 | 62.69 |
| UnivFD | 78.43 | 74.10 | 74.02 | 69.92 | 68.68 | 70.31 | 68.29 | 64.62 | 61.18 |
| PatchCraft | 89.31 | 72.48 | 71.41 | 69.43 | 67.78 | 75.99 | 74.90 | 73.53 | 72.28 |
| AIDE | 92.77 | 75.54 | 74.21 | 70.64 | 69.60 | 81.88 | 80.35 | 80.05 | 79.86 |

  — [AIDE](https://arxiv.org/html/2406.19435). (The scraper reported this table as "averaged 16 test sets in B1", which matches AIGCDetectBenchmark, not GenImage.)

**NPR (CNN, up-sampling artefacts)**
- NPR models "Neighboring Pixel Relationships" in local 2×2 grids, which up-sampling layers combined with CNN translation invariance leave behind. It is trained on 4-class ProGAN (car, cat, chair, horse) — [NPR, Tan et al., CVPR 2024](https://arxiv.org/html/2312.10461)
- NPR's own diffusion table uses a self-synthesised set (DDPM, IDDPM, ADM, Midjourney, DALL-E) rather than GenImage: NPR 80.1% mean accuracy vs UnivFD ("Ojha") 66.4% and LGrad 64.2% — [NPR](https://arxiv.org/html/2312.10461)
- On format-debiased GenImage, NPR falls to chance: balanced accuracy 51.5 ± 6.3 (DDA) and 50.7 bAcc / 50.1 AUC (B-Free). This suggests its GenImage-style gains depend on low-level cues that JPEG removes — [DDA](https://arxiv.org/html/2505.14359v6); [B-Free](https://arxiv.org/html/2412.17671)

**Debiased GenImage, CLIP/ViT vs CNN (cross-dataset, each method's released model)**
- B-Free's "GenImage unbiased" (5k real + 5k fake, fakes JPEG-compressed at a quality close to the real class), AUC / bAcc: CNNDetect 64.8/51.3; NPR 50.1/50.7; LGrad 35.8/39.6; DIRE 35.2/47.3; UnivFD 85.4/65.5; FatFormer 65.9/61.5; RINE 95.0/69.1; AIDE 84.2/60.2; C2P-CLIP 91.0/75.5; CoDE 81.6/71.7; DMID 93.2/79.0; B-Free (DINOv2+registers ViT) 97.4/89.3 — [B-Free, CVPR 2025](https://arxiv.org/html/2412.17671)
- DDA on GenImage with fakes at JPEG QF 96, balanced accuracy: NPR 51.5 ± 6.3; UnivFD 64.1 ± 10.8; FatFormer 62.8 ± 10.4; SAFE 50.3 ± 1.2; C2P-CLIP 74.4 ± 8.4; AIDE 61.2 ± 11.9; DRCT 84.7 ± 2.7; AlignedForensics 79.0 ± 22.7; DDA 91.7 ± 7.8 — [DDA, NeurIPS 2025](https://arxiv.org/html/2505.14359v6)
- DDA reports that methods without data alignment (NPR, SAFE, AIDE) show "poor robustness and large accuracy drops" under further JPEG compression (e.g. JPEG 60) on GenImage-JPEG96 — [DDA](https://arxiv.org/html/2505.14359v6)

### Inferences
- Pattern across papers: low-level CNN artefact detectors (NPR, LGrad, CNNSpot without augmentation) look strong on PNG fakes but drop to ≈50% once fakes are JPEG-compressed. Frozen or lightly tuned foundation-model features (CLIP: UnivFD, C2P-CLIP, RINE; DINOv2: B-Free) keep more accuracy (AUC 85–97 on debiased GenImage). This supports TwinCheck's plan to use a CLIP ViT-L/14 probe as the main V2 detector, and predicts that the ResNet-50 in V3 will be more JPEG-fragile than ViT-B/16 or the CLIP probe unless JPEG augmentation is used (CNNSpot's JPEG robustness in GenImage Table 5 comes entirely from augmentation).
- AIDE's high GenImage mean (86.88%) falls to 60.2–61.2% bAcc on debiased GenImage (B-Free, DDA). The headline GenImage leaderboard numbers are therefore inflated by the format shortcut and should not be quoted without this caveat.
- Note on comparability: the AIDE table numbers are trained on GenImage SD V1.4. The B-Free/DDA numbers are cross-dataset (each method's released checkpoint, trained on its own data). The two sets of numbers cannot be compared directly.

### Gaps
- I found no peer-reviewed paper in this session that runs a controlled ResNet-50 vs ViT-B/16 (same training data, same augmentation) comparison on GenImage with crop/resize robustness. GenImage's own ResNet-50 vs DeiT-S vs Swin-T comparison is the closest (trained from scratch).
- Cropping robustness specifically (e.g. 20% crop) was not reported in the papers I read. GenImage reports down-sampling (LR 112/64) only.
- UnivFD's own GenImage training numbers (as opposed to AIDE's re-run) were not checked.

---

## Q3. Which papers document the JPEG/PNG and size bias in GenImage, and what happens to accuracy after debiasing?

### Takeaway
Grommelt et al. ("Fake or JPEG?", 2024) showed that GenImage's real ImageNet images are JPEG (mostly quality factor 96) with varied sizes, while every fake is PNG at a fixed size per generator. Detectors learn "JPEG/size ⇒ real". Compressing fakes to JPEG QF 96 and restricting to 512×512-like sizes raised cross-generator accuracy by +11.06 points (ResNet-50, 71.68→82.74) and +11.74 points (Swin-T, 74.09→85.83), and improved robustness to JPEG. B-Free (CVPR 2025) and DDA (NeurIPS 2025) adopted JPEG-compressed-fake "unbiased GenImage" evaluation, under which many published SoTA detectors drop to 50–65% balanced accuracy.

### Cited Findings
**The bias (Grommelt et al.)**
- Abstract: "many datasets for AI-generated image detection contain biases related to JPEG compression and image size. Using the GenImage dataset, we demonstrate that detectors indeed learn from these undesired factors… removing the named biases substantially increases robustness to JPEG compression and significantly alters the cross-generator performance… more than 11 percentage points increase in cross-generator performance for ResNet50 and Swin-T detectors on the GenImage dataset" — [Fake or JPEG?, arXiv 2403.17608](https://arxiv.org/abs/2403.17608)
- Real (ImageNet) images are JPEG, typically quality factor 96. Fakes are PNG with a fixed size per generator: 1024×1024 (Midjourney), 512×512 (SD V1.4, SD V1.5, Wukong), 256×256 (GLIDE, ADM, VQDM), 128×128 (BigGAN) — [Fake or JPEG? (HTML)](https://arxiv.org/html/2403.17608v2)
- Evidence that detectors learned the shortcut: GenImage-trained detector accuracy on fakes "strongly declines" when the fakes are JPEG-compressed, even at quality 95. Conversely, on uncompressed (PNG) natural FFHQ images the detector's accuracy (i.e. calling them real) rises with JPEG compression: 80.45% for PNG, 94.84% at JPEG95, 100.0% at JPEG60 — [Fake or JPEG?](https://arxiv.org/html/2403.17608v2)

**Debiasing procedure**
- Train only on natural images with JPEG QF 96 and compress an equal number of generated images with the same QF 96. Size constraint: use generated images of 512×512 and natural images whose height and width are both in [450, 550], with equal real/fake counts per ImageNet class. Images are centre-cropped to 450 and then resized to 224 — [Fake or JPEG?](https://arxiv.org/html/2403.17608v2)

**Results after debiasing (cross-generator average accuracy %, "Classic" vs "Ours")**

| Detector | Train set | Classic | Debiased | Δ |
|---|---|---|---|---|
| ResNet-50 | SD1.5 | 72.16 | 83.90 | +11.74 |
| ResNet-50 | SD1.4 | 71.27 | 83.39 | +12.12 |
| ResNet-50 | Wukong | 71.61 | 80.93 | +9.32 |
| ResNet-50 | Total | 71.68 | 82.74 | +11.06 |
| Swin-T | SD1.5 | 74.14 | 85.90 | +11.76 |
| Swin-T | SD1.4 | 74.93 | 86.80 | +11.87 |
| Swin-T | Wukong | 73.20 | 84.80 | +11.60 |
| Swin-T | Total | 74.09 | 85.83 | +11.74 |

  — [Fake or JPEG?](https://arxiv.org/html/2403.17608v2)
- JPEG robustness (accuracy averaged over the generator matrix), classic GenImage vs JPEG96-debiased training: JPEG95 53.91→67.17 (+13.26); JPEG80 50.62→59.37 (+8.75); JPEG60 50.58→55.07 (+4.49) — [Fake or JPEG?](https://arxiv.org/html/2403.17608v2)

**Later confirmations and fixes**
- DDA: real images in existing datasets "are typically JPEG-encoded and vary in size, whereas synthetic images are uniformly PNG-encoded and fixed in size (such as multiples of 128 × 128)". DDA applies "JPEG compression with a quality factor of 96… to the synthetic images in GenImage, ForenSynths, and AIGCDetectionBenchmark to mitigate format bias", and observes that "detectors trained on more aligned data tend to achieve much higher average accuracy" — [DDA, NeurIPS 2025](https://arxiv.org/html/2505.14359v6)
- B-Free: performance "can be strongly impacted by spurious correlations such as content, format, or resolution". Format issues (lossless PNG fakes vs JPEG reals) have been "the Achilles' heel of forensic detectors since 2013", and resizing "wash[es] out low-level forensic clues". For evaluation, B-Free builds "GenImage unbiased" (5k real / 5k fake, fakes JPEG-compressed at a quality close to the reals) — [B-Free, CVPR 2025](https://arxiv.org/html/2412.17671); [CVF open access](https://openaccess.thecvf.com/content/CVPR2025/papers/Guillaro_A_Bias-Free_Training_Paradigm_for_More_General_AI-generated_Image_Detection_CVPR_2025_paper.pdf)
- B-Free training fix: create fakes from real images by self-conditioned reconstruction with the SD 2.1 inpainting model (empty mask → same content regenerated), plus inpainting content augmentation, blur + JPEG augmentation, a DINOv2+registers ViT, and no resizing during training. Result: 97.4 AUC / 89.3 bAcc on GenImage unbiased vs best prior RINE 95.0 AUC and DMID 79.0 bAcc — [B-Free](https://arxiv.org/html/2412.17671); [B-Free GitHub](https://github.com/grip-unina/B-Free)
- B-Free also shows that the same detector (RINE) on the same generator (DALL-E 3) can flip its score distribution from real to fake depending only on which real-image dataset (RAISE vs COCO) is used — [B-Free](https://arxiv.org/html/2412.17671)
- Rajan et al. (aligned datasets): fakes are made by passing real images only through the LDM autoencoder (encoder → decoder, no denoising), so real and fake share content, resolution, aspect ratio and format. With a ResNet-50 this is "significantly more robust" to resizing, JPEG, blur and colour jitter, generalises across SD, Midjourney, Kandinsky, Playground, PixArt-α and LCM, and costs about 10× less to build. Example: Playground accuracy 82.13 (Corvi) vs 94.85/99.48 (ours); post-processed Playground 24.27 vs 61.25/76.36 — [Rajan et al., arXiv 2410.11835](https://arxiv.org/html/2410.11835)

### Inferences
- For TwinCheck F1/F3/F5: if GenImage is used as-is, the CLIP probe and the ResNet-50/ViT-B/16 detectors may learn "PNG fixed-size ⇒ AI". This would also inflate any "unseen generator" result. Minimal fix, following Grommelt et al. and DDA: re-encode all fakes as JPEG QF 96 (matching ImageNet reals), and centre-crop/resize every image to the same size before training. Report accuracy on both raw and JPEG-aligned test sets.
- This matters for the authenticity-leakage (ALR) story. The `_V` twins (SD VAE encode→decode of true images, saved as JPEG 85) are the same idea as Rajan et al.'s aligned fakes and B-Free's self-conditioned reconstructions: same content, only generator fingerprints differ. A detector that relies on format or size shortcuts would not flag `_V` twins as AI (they are JPEG). A fingerprint-sensitive detector would flag them. Either way, the decoupled verdict head should keep the verdict unchanged. The leakage test is only convincing if the detector does flag `_V` twins as AI, so the detector's AI score on `_V` should be reported alongside ALR.
- Because the `_V` twins are VAE reconstructions, Rajan et al. is the closest prior work for the image-twin generator and should be cited as the source of the "autoencoder reconstruction = aligned fake" idea (say "to our knowledge" for any novelty claim about using it for fact-check leakage testing).

### Gaps
- "Fake or JPEG?" appears to be an arXiv preprint only; I found no peer-reviewed venue.
- The exact share of GenImage real images at QF 96 (vs other QFs) and the per-generator before/after table for each test generator were not extracted. Only the averages above were.
- Rajan et al.'s venue (ICLR 2025) is not stated on the arXiv page I read; I did not verify it independently.

---

## Q4. What is UnivFD (Ojha et al., CVPR 2023) and why are frozen CLIP features used for generalisation?

### Takeaway
UnivFD replaces end-to-end real-vs-fake training with a nearest-neighbour or linear probe on frozen CLIP ViT-L/14 image features. End-to-end classifiers learn the specific fake artefacts of the training generator and treat everything else, including unseen generators' images, as "real" (a "sink" class). A feature space that was never trained for this task does not have that asymmetry. On GenImage (SD V1.4 training) it trades in-domain accuracy (84%) for better transfer to Midjourney (73.2%) and BigGAN (80.3%), mean 73.29%. On debiased GenImage it keeps 85.4 AUC (B-Free) where CNN artefact detectors fall to chance.

### Cited Findings
- Method and rationale as quoted in Q2: the "asymmetrically tuned" classifier and "sink class" argument; nearest-neighbour and linear probing on a large pretrained vision-language model's feature space; +15.07 mAP and +25.90% accuracy over SoTA on unseen diffusion and autoregressive models — [UnivFD, arXiv 2302.10174](https://arxiv.org/abs/2302.10174)
- GenImage (train SD V1.4) UnivFD row: MJ 73.20, SD1.4 84.20, SD1.5 84.00, ADM 55.20, GLIDE 76.90, Wukong 75.60, VQDM 56.90, BigGAN 80.30, mean 73.29 — [AIDE](https://arxiv.org/html/2406.19435)
- Debiased GenImage: UnivFD 85.4 AUC / 65.5 bAcc (B-Free); 64.1 ± 10.8 bAcc (DDA) — [B-Free](https://arxiv.org/html/2412.17671); [DDA](https://arxiv.org/html/2505.14359v6)
- JPEG robustness (AIGCDetectBenchmark average): UnivFD 78.43 orig → 74.10 (JPEG95) → 68.68 (JPEG50), a much smaller drop than LGrad (75.34 → 51.55 → 50.00) — [AIDE](https://arxiv.org/html/2406.19435)

### Inferences
- The gap between UnivFD's AUC (85.4) and its bAcc (65.5) on unbiased GenImage suggests the CLIP features still separate real and fake well, but the decision threshold learned on other data is off. For TwinCheck's V2 logistic probe, it is worth fitting or calibrating the threshold on a validation split from the target distribution and reporting AUROC as well as accuracy.
- UnivFD-style probes are cheap (only a linear layer on cached features). This fits the Colab budget and the plan to cache CLIP ViT-L/14 embeddings in `features/vision/*.npy`.

### Gaps
- UnivFD's own reported numbers (on the ForenSynths/Ojha diffusion test sets) were not extracted beyond the abstract-level improvements.
- I did not confirm from the paper text in this session that UnivFD uses CLIP ViT-L/14 or trains on ProGAN. Both come from background knowledge and should be checked against the PDF before citing.
