import gzip
import numpy as np

f = gzip.open('C:/Users/ginns/Desktop/compressed.gz', 'rb')
file_content = f.read()
f.close()

# print(json.loads(file_content[0:150]))

tensor = bytearray(file_content)

print(tensor[0:150])
print(len(tensor))

inputShape = (150, 150, 256)
inputTensor = np.zeros(inputShape)

print(inputTensor[0:150])
