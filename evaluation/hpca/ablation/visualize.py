import pandas as pd 
import pickle as pkl 
import matplotlib.pyplot as plt
import os 
import numpy as np

'''
noM: no microkernel | top5 time(s)
noF: no fusion | top5 time(s)
noC: no cost model (survey) | geomean time(s) 
'''
d = {}
shape_map = {}
for file in os.listdir('result'):
    with open(os.path.join('result', file), 'rb') as f:
        alg = file.split('.')[0].split('_')[-1]
        data = pkl.load(f)
        d[alg] = {}
        for shape, value in data:
            shape = tuple(shape)
            if shape not in shape_map:
                shape_map[shape] = f'S{len(shape_map)}'
            d[alg][shape_map[shape]] = value['geomean']['time(s)'] if 'C' in alg else value['top5']['time(s)']
df = pd.DataFrame(d)
df.loc['geomean'] = np.exp(np.mean(np.log(df), axis = 0))
print (df)
ax = df.plot.bar(rot = 90)
plt.savefig('ablation.png')