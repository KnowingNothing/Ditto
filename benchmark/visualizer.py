import os 
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
def vis_cpu(dir):
    shape_tags = {}
    data = {}
    for file in os.listdir(dir):
        print('processing', file)
        workload, alg = file.split('.')[0].split('-')
        if workload == 'conv_relu_conv_nchwc':
            workload = 'conv_relu_conv'
        if workload == 'conv_conv_nchwc':
            workload = 'conv_conv'
        if workload not in data:
            data[workload] = {}
        data[workload][alg] = {}
        with open (os.path.join(dir, file), 'rb') as f:
            tmp = pkl.load(f)
            for x in tmp:
                if workload not in shape_tags:
                    shape_tags[workload] = {}
                key = tuple(x[0])
                if key not in shape_tags[workload]:
                    shape_tags[workload][key] = f'S{len(shape_tags[workload])}'
                key = shape_tags[workload][key]
                
                if type(x[1]) == dict:
                    if 'topn' in x[1]:
                        data[workload][alg][key] = x[1]['topn']['time(s)']
                    elif 'top10' in x[1]:
                        data[workload][alg][key] = x[1]['top10']['time(s)']
                    elif 'top5' in x[1]:
                        data[workload][alg][key] = x[1]['top5']['time(s)']
                    elif 'time' in x[1]:
                        data[workload][alg][key] = x[1]['time']
                    else:
                        print(x)
                        raise RuntimeError()
                else:
                    data[workload][alg][key] = x[1][0] 
    return data 
data = vis_cpu('./result')
print(data)
indices = pd.MultiIndex.from_tuples((workload, shape, alg) for workload in data for alg in data[workload] for shape in data[workload][alg])
df = pd.DataFrame(index=indices, data = [data[workload][alg][shape] for workload, shape, alg in indices], columns=['time'])
os.system('mkdir -p pics')
for workload in data.keys():
    tdf = df.loc[workload]['time'].unstack()
    tdf.loc['geomean'] = np.exp(np.mean(np.log(tdf), axis = 0))
    print(workload)
    print (tdf)
    ax = tdf.plot.bar(rot = 45)
    plt.savefig(f'pics/{workload}-cpu.png')

# for workload in df.columns:

# df = pd.DataFrame(data)
# print(df)