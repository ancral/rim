import tensorflow as tf
from tensorflow import keras
  
class Unet:
    """
    Build UNET like model for image inpainting task.
    """
    def __init__(self):
      self.prepare_model()
      
    def prepare_model(self, input_size=(32, 32, 3)):  # Cambiado a 32x32x3
      inputs = keras.layers.Input(input_size)

      # Encoder
      conv1, pool1 = self.__ConvBlock(32, (3,3), (2,2), "relu", "same", inputs)
      conv2, pool2 = self.__ConvBlock(64, (3,3), (2,2), "relu", "same", pool1)
      conv3, pool3 = self.__ConvBlock(128, (3,3), (2,2), "relu", "same", pool2)
      conv4, pool4 = self.__ConvBlock(256, (3,3), (2,2), "relu", "same", pool3)

      # Decoder
      conv5, up6 = self.__UpConvBlock(512, 256, (3,3), (2,2), (2,2), "relu", "same", pool4, conv4)
      conv6, up7 = self.__UpConvBlock(256, 128, (3,3), (2,2), (2,2), "relu", "same", up6, conv3)
      conv7, up8 = self.__UpConvBlock(128, 64, (3,3), (2,2), (2,2), "relu", "same", up7, conv2)
      conv8, up9 = self.__UpConvBlock(64, 32, (3,3), (2,2), (2,2), "relu", "same", up8, conv1)

      conv9 = self.__ConvBlock(32, (3,3), (2,2), "relu", "same", up9, False)

      # La salida final tiene 3 canales para las imágenes RGB
      outputs = keras.layers.Conv2D(3, (3, 3), activation="linear", padding="same")(conv9)

      model = keras.models.Model(inputs=[inputs], outputs=[outputs])
      model.summary()
      return model

    def __ConvBlock(self, filters, kernel_size, pool_size, activation, padding, connecting_layer, pool_layer=True):
      conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(connecting_layer)
      conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(conv)
      if pool_layer:
          pool = keras.layers.MaxPooling2D(pool_size)(conv)
          return conv, pool
      else:
          return conv

    def __UpConvBlock(self, filters, up_filters, kernel_size, up_kernel, up_stride, activation, padding, connecting_layer, shared_layer):
      conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(connecting_layer)
      conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(conv)
      up = keras.layers.Conv2DTranspose(filters=up_filters, kernel_size=up_kernel, strides=up_stride, padding=padding)(conv)
      up = keras.layers.concatenate([up, shared_layer], axis=3)

      return conv, up