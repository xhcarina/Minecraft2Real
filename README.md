# Minecraft2Real — CycleGAN Image-to-Image Translation

Unpaired image-to-image translation: **Minecraft-style landscapes → photorealistic landscapes** (and vice versa) using Cycle-Consistent Adversarial Networks (CycleGAN) in PyTorch.

This project implements CycleGAN to learn a mapping between two domains without paired data: screenshots from Minecraft and real-world landscape photos. You can train the model and view intermediate results in an HTML gallery (similar to [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)).

---

## Results

- **X → Y**: Minecraft → Real landscape  
- **Y → X**: Real landscape → Minecraft  

Training progress is saved each epoch. After training, open **`checkpoints/minecraft2real/web/index.html`** in a browser to browse results by epoch (same idea as the official repo’s `checkpoints/<name>/web/index.html`).

*(Add a screenshot or two of your best epoch here once you have them.)*

---

## Prerequisites

- **Python 3** (tested on 3.8+)
- **PyTorch** (with CUDA if you want GPU)
- **Kaggle API** (for downloading datasets)

---

## Getting Started

### 1. Clone and setup

```bash
git clone https://github.com/xhcarina/Minecraft2Real.git
cd Minecraft2Real
```

### 2. Install dependencies

```bash
pip install torch torchvision matplotlib numpy pillow kagglehub
```

(Use a virtualenv or conda if you prefer.)

### 3. Datasets

The notebook pulls data via the Kaggle API:

- **Domain X (Minecraft)**: [minecraft-landscapes](https://www.kaggle.com/datasets/coreydobbs/minecraft-landscapes) (Kaggle)
- **Domain Y (Real)**: [cleaned-landscape-for-cyclegan](https://www.kaggle.com/datasets/xuehancarina/cleaned-landscape-for-cyclegan) (Kaggle)

Configure the Kaggle API (e.g. `~/.kaggle/kaggle.json`) so the notebook can download these automatically.

### 4. Train

1. Open **`CycleGanProject.ipynb`** in Jupyter or Google Colab (GPU recommended).
2. Run all cells in order.  
   - Training runs for 60 epochs by default.  
   - Every 5 epochs a checkpoint is saved as `result_epoch_<e>.pth`.  
   - Each epoch saves a progress figure into **`checkpoints/minecraft2real/web/images/`** and updates **`checkpoints/minecraft2real/web/index.html`**.

### 5. View training progress (web interface)

After (or during) training, open in a browser:

```text
checkpoints/minecraft2real/web/index.html
```

You’ll see one page per epoch (newest first), with the same style of visualization as in the official CycleGAN/pix2pix repo.

### 6. Test / inference

Use the later cells in the notebook to load a checkpoint (e.g. `result_epoch_49.pth`), run the generators on test images, and visualize Minecraft→Real and Real→Minecraft.

---

## Project layout

```text
Minecraft2Real/
├── README.md                 # This file
├── CycleGanProject.ipynb     # Full pipeline: data, train, visualize
├── util/
│   └── web_html.py           # Epoch gallery: save figures + index.html
└── checkpoints/
    └── minecraft2real/
        └── web/
            ├── index.html   # Open this to view progress
            └── images/      # epoch_000.png, epoch_001.png, ...
```

---

## Training details

- **Model**: CycleGAN (two generators G_XY, G_YX; two discriminators).
- **Losses**: GAN loss, cycle consistency (L1), identity loss; optional edge/sky weighting.
- **Epochs**: 60 (default); checkpoint every 5 epochs; optional LR decay after 40 epochs.
- **Output**: Checkpoints `result_epoch_<e>.pth` and web gallery in `checkpoints/minecraft2real/web/`.

---

## Citation

If you use this code or idea, consider citing the CycleGAN paper:

```bibtex
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={ICCV},
  year={2017}
}
```

---

## Related work

- [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) — PyTorch CycleGAN/pix2pix (this repo’s interface is inspired by their web results).
- [CycleGAN (Torch)](https://github.com/junyanz/CycleGAN) — Original Torch implementation.

---

## License

See repository for license information.
