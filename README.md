# Recurrent Inpainting Model (RIM)

This repository contains the implementation of the **RIM** (*Recurrent Inpainting Model*) applied to the reconstruction of strain maps or physical fields from incomplete data obtained via distributed sensors. The architecture is based on a conditional reverse diffusion process, with a denoising model trained on RGB-encoded spatial maps.

This code is designed as reproducible support for the experiments presented in the article:  
> **Full strain matrix estimation in thin-walled structures with recurrent inpainting model**  
> *Cruz-Alonso, Terroba, Cuesta-Infante*  
> [Journal name, year]

---

## Project Structure

```
rim/
├── main_rim.py                # Main training script
├── inference_rim.py
├── config/
│   └── models.ini             # Model and training configuration
├── models/
│   ├── rim.py                 # RIM model implementation
│   └── unet.py                # UNet model used as denoising network
├── utils/
│   ├── metrics.py             # Loss functions and metrics
│   ├── generate_mask.py       # Mask and data generation
│   └── mask_cube.npy          # (Optional) Predefined sensor mask
├── train/                     # Folder expected to contain training data (images)
└── test/                      # Folder expected to contain validation/test data
```

---

## Requirements

- Python ≥ 3.8  
- TensorFlow ≥ 2.10  
- NumPy, Pillow, matplotlib, OpenCV, wandb (optional)

Recommended installation (virtual environment):

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> Make sure to register custom loss functions in `utils/metrics.py`, such as `loss_inpainting`, `gradient_diff_loss`, etc.

---

## Training

Place your input data in the `train/` and `test/` folders. Ensure they are `.png` or `.jpg` images with square dimensions.

Then run:

```bash
python main_rim.py
```

The model will train using the configuration defined in `config/models.ini` and checkpoints will be saved in the `checkpoints/` folder.

---

## Configuration (`config/models.ini`)

Some key parameters are:

```ini
[RIM]
timesteps = 5
beta_min = 0.05
beta_max = 0.2
intensity_realce = 0.03
intensity_smooth = 0.2
umbral_realce = 20
color_peak = 1.0, 1.0, 0.0       # Highlight for peaks (RGB)
color_trough = 0.3, 0.0, 0.5     # Highlight for troughs (RGB)
```

---

## Citation

If you use this code or results derived from this model, please cite the associated paper:

```bibtex
@article{XXX,
  title={Full strain matrix estimation in thin-walled structures with recurrent inpainting model},
  author={Cruz-Alonso, Terroba, Cuesta-Infante},
  journal={XXX},
  year={2025}
}
```

---

For any technical questions, contact the lead author or open an issue in the repository.
