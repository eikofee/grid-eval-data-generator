import numpy as np
from sklearn.manifold import TSNE
import h5py
import sys

hf = h5py.File("python/tsneDataX.h5", 'r')
data = np.array(hf.get("datakey"))
print(np.shape(data), file = sys.stderr)
model = TSNE()
res = model.fit_transform(data)
for x in res:
    print(str(x[0]) + "," + str(x[1]))
