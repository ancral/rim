import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
def loss_min_max(y_true, y_pred, epsilon=0.3, lambda_weight=10):
    """
    Custom loss function for heatmaps with color gradients (purple to yellow).
    Focuses on errors in purple regions (low values) and yellow regions (high values).

    Parameters:
    - y_true: Tensor with target images (shape: [batch, height, width, 3]).
    - y_pred: Tensor with model predictions (shape: [batch, height, width, 3]).
    - epsilon: Tolerance range to define proximity to purple and yellow.
    - lambda_weight: Additional weight for errors in relevant regions.

    Returns:
    - Scalar loss.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # Mask for values close to "purple"
    # Low values in red and green, higher values in blue
    mask_purple = tf.reduce_all(tf.stack([
        y_true[..., 0] <= epsilon,       # Low red
        y_true[..., 1] <= epsilon,       # Low green
        y_true[..., 2] >= (1 - epsilon)  # High blue
    ], axis=-1), axis=-1)

    # Mask for values close to "yellow"
    # High values in red and green, low values in blue
    mask_yellow = tf.reduce_all(tf.stack([
        y_true[..., 0] >= (1 - epsilon),  # High red
        y_true[..., 1] >= (1 - epsilon),  # High green
        y_true[..., 2] <= epsilon          # Low blue
    ], axis=-1), axis=-1)
    
    # Combine masks
    weighted_mask = tf.cast(mask_purple | mask_yellow, tf.float32)
    
    # Expand mask to apply per channel
    weighted_mask = tf.expand_dims(weighted_mask, axis=-1)

    # Weighted error in purple and yellow regions
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
  
  return (loss_x + loss_y) / 2