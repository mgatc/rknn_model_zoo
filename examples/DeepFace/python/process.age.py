import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python3 {} [data file]".format(sys.argv[0]))
    exit(1)

file = sys.argv[1]
data = np.load(file)

print(data)
print(data.shape) # Example: print the shape of the array

sum = 0.0

for item in range(len(data[0])-1):
    print(item * data[0][item])
    sum += item * data[0][item]

print(sum)