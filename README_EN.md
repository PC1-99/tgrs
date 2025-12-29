# TPCA-Net: RGB-T Salient Object Detection

## Introduction

RGB-T salient object detection (SOD) for autonomous aerial vehicle (AAV) remote sensing aims to combine visible and thermal infrared imagery to precisely segment salient targets within complex urban and ecological scenes under low illumination, occlusion, and dynamic backgrounds. However, AAV-oriented RGB-T SOD still encounters three issues: (i) early fusion is susceptible to thermal noise and heat leakage, weakening saliency cues; (ii) cross-modal spatial misalignment hampers fine-grained correspondence; and (iii) coarse thermal boundaries blur local structures and perturb downstream reasoning. Inspired by these issues, we propose TPCA-Net, a unified language–thermal framework that injects thermal-physics priors into cross-modal fusion to stabilize the learning of both global context and local details. Specifically, we propose a Multi-dimensional Thermal Feature Enhancement (MTFE) to adaptively enhance thermodynamic cues while suppressing background leakage. In addition, we introduce a Heatmap Variation-Aware Dynamic Window (HVDW) that schedules pseudo-temporal dynamic windows to separate noise from structural variations. We further propose a Bi-directional Alignment-and-Fusion with Edge-Enhanced Decoding (BAF-ED) that performs bidirectional feature alignment with edge-aware decoding to restore sharp object boundaries. Extensive experiments demonstrate state-of-the-art performance and significant improvements in boundary precision in the remote sensing domain.

## File Structure

```
TGRS/
├── model.py           
├── train.py           
├── Encoder.py         
├── MTFE.py            
├── HVDW.py            
├── BAEF.py            
└── README.md          
```

## Quick Start

### Training

Run the following command to start training:

```bash
python train.py \
    --train-rgb-dir data/train/rgb \
    --train-thermal-dir data/train/thermal \
    --train-mask-dir data/train/mask \
    --test-rgb-dir data/test/rgb \
    --test-thermal-dir data/test/thermal \
    --test-mask-dir data/test/mask
```

For more training parameters, use `python train.py --help`. You can modify them according to your needs.


## Contact

If you have any questions or need access to our trained model weights, please contact us via:

**Email**: tangzhiri@jnu.edu.cn

We will reply as soon as possible after receiving your email.

