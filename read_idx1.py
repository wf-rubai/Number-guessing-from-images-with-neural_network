import numpy as np
import struct

def read_idx1_ubyte(file_path):
    with open(file_path, 'rb') as f:
        # Read the header information
        magic_number, num_items = struct.unpack('>II', f.read(8))
        
        if magic_number != 2049:
            raise ValueError(f'Invalid magic number {magic_number}, expected 2049.')
        
        # Read the label data
        labels = np.fromfile(f, dtype=np.uint8)
    
    return labels

# Example usage
# file_path = 'neural_network/t10k-labels.idx1-ubyte'
# images = read_idx1_ubyte(file_path)
# print(images)  # Should print (num_images, num_rows, num_cols)