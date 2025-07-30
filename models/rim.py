from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from models.unet import Unet


class Rim(Model):
    def __init__(self, denoising_model=Unet(), padding_zone=None, rim_cfg=None, **kwargs):        
        if rim_cfg is None:
            raise ValueError("Debe proporcionarse un archivo de configuración (.ini) correctamente leído con configparser.")

        # ==== Hiperparámetros del modelo desde config ====
        self.timesteps = int(rim_cfg["timesteps"])
        self.beta_min = float(rim_cfg["beta_min"])
        self.beta_max = float(rim_cfg["beta_max"])
        self.intensity_realce = float(rim_cfg["intensity_realce"])
        self.intensity_smooth = float(rim_cfg["intensity_smooth"])
        self.threshold_relative = float(rim_cfg["threshold_relative"])
        self.umbral_realce = int(rim_cfg["umbral_realce"])
        self.peak_color = self._parse_rgb(rim_cfg["color_peak"])
        self.trough_color = self._parse_rgb(rim_cfg["color_trough"])
        self.padding_zone = padding_zone

        # Luego llamas al super (ya sin 'config' en kwargs)
        super(Rim, self).__init__(**kwargs)

        # ==== Red de denoising (por defecto: UNet) ====
        self.denoising_model = denoising_model.prepare_model()
        self.denoising_class = denoising_model

        # ==== Parámetros de difusión ====
        self.betas = tf.linspace(self.beta_min, self.beta_max, self.timesteps)
        self.alphas = 1.0 - self.betas

    def _parse_rgb(self, rgb_string):
        values = [float(x.strip()) for x in rgb_string.split(",")]
        if len(values) != 3:
            raise ValueError(f"Se esperaban 3 valores RGB, pero se recibieron: {rgb_string}")
        return values

    def reverse_diffusion_step(self, x_t, t, pred_noise, sensor_data, pad):
        alpha_t = tf.gather(self.alphas, t)
        beta_t = tf.gather(self.betas, t)
        x_t_minus_1 = (x_t - tf.sqrt(beta_t) * pred_noise) / tf.sqrt(alpha_t)
        x_t_minus_1 = tf.where(sensor_data != 0, sensor_data, x_t_minus_1)

        if pad is not None:
            x_t_minus_1 = tf.where(pad != 0, x_t_minus_1, 0)
        return x_t_minus_1

    def enhance_peaks_and_troughs(self, image, intensity=0.05):
        luminance = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
        p_high = 1.0 - self.umbral_realce / 100
        p_low = self.umbral_realce / 100

        peaks = tf.where(luminance > p_high, luminance, 0)
        troughs = tf.where(luminance < p_low, luminance, 0)

        peak_boost = tf.expand_dims(peaks, axis=-1) * tf.constant(self.peak_color, dtype=tf.float32)
        trough_boost = tf.expand_dims(troughs, axis=-1) * tf.constant(self.trough_color, dtype=tf.float32)

        enhanced_image = image + intensity * (peak_boost - trough_boost)
        return tf.clip_by_value(enhanced_image, 0, 1)

    def apply_spatial_smoothing(self, image, kernel_size=5, sigma=1.5, intensity=0.05):
        def gaussian_kernel(size, sigma, channels):
            ax = tf.range(-(size // 2), (size // 2) + 1, dtype=tf.float32)
            xx, yy = tf.meshgrid(ax, ax)
            gaussian = tf.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
            gaussian /= tf.reduce_sum(gaussian)
            kernel = tf.expand_dims(gaussian, axis=-1)
            kernel = tf.repeat(kernel, channels, axis=-1)
            return tf.expand_dims(kernel, axis=-1)

        channels = image.shape[-1]
        kernel = gaussian_kernel(kernel_size, sigma, channels)
        smoothed = tf.nn.depthwise_conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')
        return image + intensity * (smoothed - image)

    def refinement_pass(self, noisy_image, sensor_data, training=False, pad=None):
        threshold = self.threshold_relative * self.timesteps
        for t in range(self.timesteps - 1, -1, -1):
            pred_noise = self.denoising_model(noisy_image, training=training)
            noisy_image = self.reverse_diffusion_step(noisy_image, t, pred_noise, sensor_data, pad)
            if t < threshold:
                realce_intensity = self.intensity_realce * (threshold - t)
                noisy_image = self.enhance_peaks_and_troughs(noisy_image, intensity=realce_intensity)
            noisy_image = self.apply_spatial_smoothing(noisy_image, kernel_size=7, sigma=2.0, intensity=self.intensity_smooth)
        return noisy_image

    def call(self, sensor_data, training=False):
        noisy_image = sensor_data
        shape = tf.shape(sensor_data)

        if self.padding_zone is not None:
            pad = tf.cast(self.padding_zone, tf.float32)   # (H, W)
            pad = tf.expand_dims(pad, axis=0)              # (1, H, W)
            pad = tf.expand_dims(pad, axis=-1)             # (1, H, W, 1)
            pad = tf.tile(pad, [shape[0], 1, 1, shape[3]])  # (B, H, W, C)
            
        noisy_image = self.refinement_pass(noisy_image, sensor_data, training, pad)
        return tf.clip_by_value(noisy_image, 0, 1)

    def train_step(self, data):
        sensor_data, target = data
        sensor_data = tf.cast(sensor_data, tf.float32)
        target = tf.cast(target, tf.float32)

        with tf.GradientTape() as tape:
            output = self.call(sensor_data, training=True)
            loss = tf.reduce_mean(self.compiled_loss(target, output))

        grads = tape.gradient(loss, self.denoising_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.denoising_model.trainable_weights))
        self.compiled_metrics.update_state(target, output)

        results = {m.name: m.result() for m in self.metrics[1].metrics}
        results["loss"] = loss
        return results

    def test_step(self, data):
        sensor_data, target = data
        sensor_data = tf.cast(sensor_data, tf.float32)
        target = tf.cast(target, tf.float32)

        output = self.call(sensor_data, training=False)
        loss = tf.reduce_mean(self.compiled_loss(target, output))
        self.compiled_metrics.update_state(target, output)

        results = {m.name: m.result() for m in self.metrics[1].metrics}
        results["loss"] = loss
        return results
