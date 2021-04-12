import numpy as np

'''
a thermometer encoding for use to encode architecture information such as number of conv layers or fc layers
example: numbers = [1, 5], max_size=50
output:[1,0,0,0,0,0,0,0,0,,,0,0,,0,1,1,1,1,1,0,0,0,,0,,,....,0]
concatenated array of size #numbers * max_size
max_size indicates the maximum number that is possible (must be consistent)

'''
def thermometer_encoding(numbers, max_size):
    encoding = np.arange(max_size) < np.array(numbers).reshape(-1, 1).astype(float)     #2d array of thermometer
    return encoding.flatten().tolist()


'''
based on max layer
create decimal number between [-1,1] to reflect the number of layers in the architecture
'''
def decimal_encoding(num_layer, max_size):
	increment = 2/max_size
	return (-1+ num_layer*increment)


