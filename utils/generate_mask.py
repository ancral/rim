from tensorflow import keras
import numpy as np
import cv2

class createAugment(keras.utils.Sequence):
  def __init__(self, X, y, batch_size=32, dim=(32, 32), mask = None, shuffle=True, num_sensors=16, scheme_mask=2, type_scheme_mask=1, delete_data=False):
      """
        Scheme Mask: 0 (Lines), 1 (Random points), 2 (Fixed points)
        Type scheme mask (for scheme_mask=2): 0 (r > c), 1 (r < c), 2 (cross), 3 (waves), 4 (flower)
      """
      self.batch_size = batch_size
      self.X = X.astype(np.uint8)
      self.y = y.astype(np.uint8)
      self.dim = dim
      self.shuffle = shuffle
      self.num_sensors = num_sensors
      self.full_squares = self.num_sensors % 4 == 0
      self.delete_data = delete_data
      self.is_rgb = len(X.shape) == 4 and X.shape[-1] == 3  # Check if images are RGB
      if not mask is None:
        self.mask = mask.astype(np.uint8)
      else:
        self.mask = mask

      if scheme_mask == 0:
        self.__createMask = self.__createMaskLines
      elif scheme_mask == 1:
        self.__createMask = self.__createMaskRandom
      if scheme_mask == 2:
        self.__createMask = self.__createMaskFixedMatrix
        col = 4 if (self.num_sensors % 4 == 0 and self.num_sensors / 4 != 1) else 2
        row = int(self.num_sensors / col)

        if type_scheme_mask == 0:
          self.pos_x_sensors, self.pos_y_sensors = col, row if row > col else row, col
          self.gap_x_sensors = int(self.dim[0] / (self.pos_x_sensors + 1))
          self.gap_y_sensors = int(self.dim[1] / (self.pos_y_sensors + 1))
        elif type_scheme_mask == 1:
          self.pos_x_sensors, self.pos_y_sensors = col, row if row < col else row, col
          self.gap_x_sensors = int(self.dim[0] / (self.pos_x_sensors + 1))
          self.gap_y_sensors = int(self.dim[1] / (self.pos_y_sensors + 1))
        elif type_scheme_mask == 2:
          self.__createMask = self.__createMaskFixedCross
          self.pos_x_sensors, self.pos_y_sensors = int((self.num_sensors + 2)/2) + 1, int((self.num_sensors + 2)/2) + 1
          if not self.full_squares:
            self.pos_x_sensors, self.pos_y_sensors = int((self.num_sensors)/2) + 2, int((self.num_sensors)/2) + 3

          self.gap_x_sensors = int(self.dim[0] / self.pos_x_sensors)
          self.gap_y_sensors = int(self.dim[1] / self.pos_y_sensors)
        elif type_scheme_mask == 3:
          self.__createMask = self.__createMaskFixedWaves
          self.pos_x_sensors, self.pos_y_sensors = self.num_sensors - 2, 4
          self.gap_x_sensors = int(self.dim[0] / (self.pos_x_sensors))
          self.gap_y_sensors = int(self.dim[1] / (self.pos_y_sensors + 1))
        else:
          self.__createMask = self.__createMaskFixedFlower
          self.pos_x_sensors, self.pos_y_sensors = int(self.num_sensors/2) + 1, int(self.num_sensors/2) + 1
          self.gap_x_sensors = int(self.dim[0] / (self.pos_x_sensors + 1))
          self.gap_y_sensors = int(self.dim[1] / (self.pos_y_sensors + 1))

        print(f"[Info] Gap_x: {self.gap_x_sensors}, Gap_y: {self.gap_y_sensors}")
        print(f"[Info] Pos_x: {self.pos_x_sensors}, Pos_y: {self.pos_y_sensors}")

      self.on_epoch_end()

  def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(len(self.X) / self.batch_size))

  @property
  def shape(self):
      return (len(self), self.dim[0], self.dim[1], 3 if self.is_rgb else self.dim[0:2])

  def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

      # Generate data
      return self.__data_generation(indexes)

  def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.X))
      if self.shuffle:
          np.random.shuffle(self.indexes)

  def get_position_mask(self):
    if self.mask is None:
      mask = self.__createMask() 
    else:
      mask = self.mask * 255
    
    nonzero_indices = np.argwhere(mask==255)
    print(nonzero_indices)
    for delete_ind in self.delete_data:
      fila, col = nonzero_indices[delete_ind]
      mask[fila, col] = 0 
    return np.argwhere(mask==255)

  def __data_generation(self, idxs):
    # X_batch is a matrix of masked images used as input. Masked image
    # y_batch is a matrix of original images used for computing error from reconstructed image. Original image
    if self.is_rgb:
        X_batch = np.zeros((self.batch_size, self.dim[0], self.dim[1], 3))
        y_batch = np.zeros((self.batch_size, self.dim[0], self.dim[1], 3))
    else:
        X_batch = np.zeros((self.batch_size, self.dim[0], self.dim[1]))
        y_batch = np.zeros((self.batch_size, self.dim[0], self.dim[1]))

    ## Iterate through random indexes
    for i, idx in enumerate(idxs):
      image_copy = self.X[idx].copy()

      ## Get mask (associated to that image) and apply mask
      if self.mask is None:        mask = self.__createMask() 
      else:
        mask = self.mask * 255
      
      nonzero_indices = np.argwhere(mask==255)
      for delete_ind in self.delete_data:  
        fila, col = nonzero_indices[delete_ind]
        mask[fila, col] = 0 
           
      mask = mask.squeeze() if not self.is_rgb else cv2.merge([mask, mask, mask])
            
      masked_image = cv2.bitwise_and(image_copy, mask)
      
      X_batch[i,] = masked_image.astype(np.float32)/255.0
      y_batch[i] = self.y[idx].astype(np.float32)/255.0

    return X_batch, y_batch

  def __createMaskFixedFlower(self):
    mask = np.full((self.dim[0], self.dim[1]), 0, np.uint8)

    for i in range(1, int(self.num_sensors/4) + 1):
      # Index = 1: Square; 2: Diamond
      offset_diamond = ((i+1) % 2)*int((self.dim[0] - i*2*self.gap_x_sensors)/2)
      # A
      mask[i*self.gap_y_sensors, i*self.gap_x_sensors + offset_diamond] = 255
      # B
      mask[i*self.gap_y_sensors + offset_diamond, self.dim[0] - i*self.gap_x_sensors] = 255
      # C
      mask[self.dim[1] - i*self.gap_y_sensors - offset_diamond, i*self.gap_x_sensors] = 255
      # D
      mask[self.dim[1] - i*self.gap_y_sensors, self.dim[0] - i*self.gap_x_sensors - offset_diamond] = 255

    if not self.full_squares:
      mask[int(self.dim[1]/2), int(self.dim[0]/2)] = 255

    return mask


  def __createMaskFixedWaves(self):
      mask = np.full((self.dim[0], self.dim[1]), 0, np.uint8)
      # Square
      mask[self.gap_y_sensors, self.gap_x_sensors] = 255
      mask[self.gap_y_sensors, self.dim[0] - self.gap_x_sensors] = 255
      mask[self.dim[1] - self.gap_y_sensors, self.gap_x_sensors] = 255
      mask[self.dim[1] - self.gap_y_sensors, self.dim[0] - self.gap_x_sensors] = 255

      for i in range(2, self.num_sensors - 2 - 1):
        ini_elev = (i % 2)*(self.gap_y_sensors)
        mask[ini_elev + 2*self.gap_y_sensors, i*self.gap_x_sensors] = 255

      if self.num_sensors % 2 == 0:
        mask[4*self.gap_y_sensors, int(self.pos_x_sensors/2)*self.gap_x_sensors] = 255

      return mask

  def __createMaskFixedCross(self):
      mask = np.full((self.dim[0], self.dim[1]), 0, np.uint8)
      for i in range(1, int(self.num_sensors/4) + 1):
        # Square
        mask[i*self.gap_x_sensors, i*self.gap_y_sensors] = 255
        mask[i*self.gap_x_sensors, (self.pos_y_sensors - i)*self.gap_y_sensors] = 255
        mask[(self.pos_x_sensors - i)*self.gap_x_sensors, i*self.gap_y_sensors] = 255
        mask[(self.pos_x_sensors - i)*self.gap_x_sensors, (self.pos_y_sensors - i)*self.gap_y_sensors] = 255

      if not self.full_squares:
        mask[(int(self.pos_x_sensors/2))*self.gap_x_sensors, (int(self.pos_y_sensors/2) - 1)*self.gap_y_sensors] = 255
        mask[(int(self.pos_x_sensors/2)+ 1)*self.gap_x_sensors, (int(self.pos_y_sensors/2) + 1)*self.gap_y_sensors] = 255

      return mask

  def __createMaskFixedMatrix(self):
    mask = np.full((self.dim[0], self.dim[1]), 0, np.uint8)
    for i in range(1, self.pos_x_sensors + 1):
      for j in range(1, self.pos_y_sensors + 1):
        mask[i*self.gap_x_sensors, j*self.gap_y_sensors] = [255, 255, 255]
        
    return mask

  def __createMaskRandom(self):
    mask = np.full((32, 32), 0, np.uint8)
    for _ in range(self.num_sensors):
      x, y = np.random.randint(1, 32), np.random.randint(1, 32)
      mask[x, y] = [255, 255, 255]

    return mask

  def __createMaskLines(self):
      mask = np.full((self.dim[0], self.dim[1]), 0, np.uint8)
      if self.full_squares:
        for i in range(1, int(self.num_sensors / 4) + 1):
          mask[i*self.gap_y_sensors, :] = 255
          mask[:, i*self.gap_x_sensors] = 255
      else:
        for i in range(1, int(self.num_sensors / 2) + 1):
          mask[i*self.gap_y_sensors, :] = 255
      return mask
  
def rearrange_image_and_create_mask(original_image, target_size):
    if len(original_image.shape) == 3:
        original_height, original_width, num_channels = original_image.shape
    else:
        original_height, original_width = original_image.shape
        num_channels = 1
    
    num_blocks = original_height // original_width if original_height > original_width else original_width // original_height
    print(num_blocks)

    block_size = original_width // num_blocks
    
    div_h = (original_height // block_size) + 1
    div_w =  (num_blocks // div_h)
    
    # Calculate new dimensions for the rearranged image
    new_height = block_size * div_h
    new_width = block_size * div_w

    # Create a new image with white background (padding) of size new_height x new_width x num_channels
    if num_channels == 1:
        new_img = np.ones((new_height, new_width)) * 255
    else:
        new_img = np.ones((new_height, new_width, num_channels)) * 255
    
    # Iterate over blocks in the original image
    for i in range(num_blocks):
        # Calculate the position to paste the block in the new image
        new_left = (i % div_w) * block_size
        new_upper = (i // div_w) * block_size
        
        # Paste the block in the new image
        if num_channels == 1:
            new_img[new_upper:new_upper+block_size, new_left:new_left+block_size] = original_image[:, i*block_size:(i+1)*block_size]
        else:
            new_img[new_upper:new_upper+block_size, new_left:new_left+block_size, :] = original_image[:, i*block_size:(i+1)*block_size, :]
    
    # Create a final image with white background (padding) of size target_size x target_size x num_channels
    if num_channels == 1:
        final_img = np.zeros((target_size, target_size))
    else:
        final_img = np.zeros((target_size, target_size, num_channels))
    
    # Calculate vertical and horizontal padding to center the new image in the final image
    vertical_padding = (target_size - new_height) // 2
    horizontal_padding = (target_size - new_width) // 2
    
    # Paste the rearranged image in the final image, centered vertically and horizontally
    if num_channels == 1:
        final_img[vertical_padding:vertical_padding+new_height, horizontal_padding:horizontal_padding+new_width] = new_img
    else:
        final_img[vertical_padding:vertical_padding+new_height, horizontal_padding:horizontal_padding+new_width, :] = new_img
    
    # Create a mask indicating padding areas
    mask = np.zeros((target_size, target_size))
    mask[vertical_padding:vertical_padding+new_height, horizontal_padding:horizontal_padding+new_width] = 1
    
    return final_img, mask