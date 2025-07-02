import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python3 {} [data file]".format(sys.argv[0]))
    exit(1)

file = sys.argv[1]
data = np.load(file)

print(data)
print(data.shape) # Example: print the shape of the array
