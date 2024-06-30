import numpy as np
import struct

def read_idx3_ubyte(file_path):
    with open(file_path, 'rb') as f:
        # Read the header information
        magic_number, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))
        
        if magic_number != 2051:
            raise ValueError(f'Invalid magic number {magic_number}, expected 2051.')
        
        # Read the image data
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, num_rows, num_cols)
    
    return images

# Example usage
# file_path = 'neural_network/t10k-images.idx3-ubyte'
# images = read_idx3_ubyte(file_path)
# print(images.shape)  # Should print (num_images, num_rows, num_cols)
# print(images[0])  # Should print (num_images, num_rows, num_cols)
