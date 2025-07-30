# Recurrent Inpainting Model (RIM)

Este repositorio contiene la implementación del modelo **RIM** (*Recurrent Inpainting Model*) aplicado a la reconstrucción de mapas de deformación o campos físicos a partir de datos incompletos obtenidos mediante sensores distribuidos. La arquitectura se basa en un proceso de difusión inversa condicionado, con un modelo de denoising entrenado sobre mapas en espacio RGB codificados.

Este código está diseñado como soporte reproducible para los experimentos presentados en el artículo:
> **Full strain matrix estimation in thin-walled structures with recurrent inpainting model**  
> *Cruz-Alonso, Terroba, Cuesta-Infante*  
> [Nombre de la conferencia o revista, año]

---

## Estructura del proyecto

```
rim/
├── main_rim.py                # Script principal de entrenamiento
├── config/
│   └── models.ini              # Configuración del modelo y entrenamiento
├── models/
│   ├── rim.py                 # Implementación del modelo RIM
│   └── unet.py                 # Modelo UNet como red de denoising
├── utils/
│   ├── metrics.py              # Funciones de pérdida y métricas
│   ├── generate_mask.py       # Generación de máscaras y datos
│   └── mask_cube.npy                  # (Opcional) Máscara predefinida de sensores
├── train/                      # Carpeta esperada para datos de entrenamiento (imágenes)
└── test/                       # Carpeta esperada para datos de validación/test

```

---

## Requisitos

- Python ≥ 3.8
- TensorFlow ≥ 2.10
- NumPy, Pillow, matplotlib, OpenCV, wandb (opcional)

Instalación recomendada (entorno virtual):

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> Asegúrate de registrar las funciones de pérdida personalizadas en `utils/metrics.py`, como `loss_inpainting`, `gradient_diff_loss`, etc.

---

## Entrenamiento

Coloca tus datos de entrada en las carpetas `train/` y `test/`. Asegúrate de que sean imágenes `.png` o `.jpg` con dimensiones cuadradas.

Luego ejecuta:

```bash
python main_rim.py
```

El modelo se entrenará usando la configuración definida en `config/models.ini` y se guardará en la carpeta `checkpoints/`.

---

## Configuración (`config/models.ini`)

Algunos de los parámetros clave son:

```ini
[RIM]
timesteps = 5
beta_min = 0.05
beta_max = 0.2
intensity_realce = 0.03
intensity_smooth = 0.2
umbral_realce = 20
color_peak = 1.0, 1.0, 0.0       # Realce para picos (RGB)
color_trough = 0.3, 0.0, 0.5     # Realce para valles (RGB)
```

---

## Cita

Si utilizas este código o los resultados derivados de este modelo, por favor cita el paper asociado:

```bibtex
@article{XXX,
  title={Full strain matrix estimation in thin-walled structures with recurrent inpainting model},
  author={Cruz-Alonso, Terroba, Cuesta-Infante},
  journal={XXX},
  year={2025}
}
```
---

Para cualquier duda técnica, contacto con el autor principal o crea una issue en el repositorio.
