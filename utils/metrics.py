import tensorflow as tf
from tensorflow import keras

@tf.keras.utils.register_keras_serializable()
def loss_min_max(y_true, y_pred, epsilon=0.3, lambda_weight=10):
    """
    Función de pérdida personalizada para mapas de calor con gradientes de color (morado a amarillo).
    Se enfoca en errores en regiones moradas (bajos valores) y amarillas (altos valores).

    Parámetros:
    - y_true: Tensor con las imágenes objetivo (shape: [batch, height, width, 3]).
    - y_pred: Tensor con las predicciones del modelo (shape: [batch, height, width, 3]).
    - epsilon: Rango de tolerancia para definir cercanía a morado y amarillo.
    - lambda_weight: Peso adicional para los errores en regiones relevantes.

    Retorna:
    - Pérdida escalar.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # Máscara para valores cercanos a "morado"
    # Valores bajos en rojo y verde, valores más altos en azul
    mask_morado = tf.reduce_all(tf.stack([
        y_true[..., 0] <= epsilon,       # Rojo bajo
        y_true[..., 1] <= epsilon,       # Verde bajo
        y_true[..., 2] >= (1 - epsilon)  # Azul alto
    ], axis=-1), axis=-1)

    # Máscara para valores cercanos a "amarillo"
    # Valores altos en rojo y verde, valores bajos en azul
    mask_amarillo = tf.reduce_all(tf.stack([
        y_true[..., 0] >= (1 - epsilon),  # Rojo alto
        y_true[..., 1] >= (1 - epsilon),  # Verde alto
        y_true[..., 2] <= epsilon         # Azul bajo
    ], axis=-1), axis=-1)
    
    # Combinar máscaras
    weighted_mask = tf.cast(mask_morado | mask_amarillo, tf.float32)
    
    # Expandir la máscara para aplicarla por canal
    weighted_mask = tf.expand_dims(weighted_mask, axis=-1)

    # Error ponderado en regiones moradas y amarillas
    weighted_loss = tf.reduce_mean(weighted_mask * tf.square(y_true - y_pred))
        
    return lambda_weight * weighted_loss

@tf.keras.utils.register_keras_serializable()
def loss_inpainting(y_true, y_pred):
    return gradient_diff_loss(y_true, y_pred) + tf.reduce_mean(tf.keras.losses.mse(y_true, y_pred)) + loss_min_max(y_true, y_pred)

@tf.keras.utils.register_keras_serializable()
def gradient_diff_loss(y_true, y_pred):
  def compute(img):
    dy = img[:, 1:, :] - img[:, :-1, :]
    dx =  img[:, :, 1:] - img[:, :, :-1]
    return dx, dy
  y_true = tf.squeeze(tf.cast(y_true, tf.float32))
  y_pred = tf.squeeze(tf.cast(y_pred, tf.float32))
  dx_pred, dy_pred = compute(y_pred)
  dx_true, dy_true = compute(y_true)
  
  diff_x = tf.abs(dx_pred - dx_true)
  diff_y = tf.abs(dy_pred - dy_true)
  
  loss_x = tf.reduce_mean(diff_x)
  loss_y = tf.reduce_mean(diff_y)
  
  return (loss_x + loss_y)/2