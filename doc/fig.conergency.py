import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/times.ttf')
plt.rc('font',family='Times New Roman')
import numpy as np

euconst_data = pd.read_csv('doc/csv/convergency_fuzzys2s_euconst.csv').to_numpy().tolist()
samsum_data = pd.read_csv('doc/csv/convergency_fuzzys2s_samsum.csv').to_numpy().tolist()
hs_data = pd.read_csv('doc/csv/convergency_fuzzys2s_hearthstone.csv').to_numpy().tolist()

def sampling(data, step=10):
    output=[]
    for i in range(len(data)):
        if i % step == 0:
            output.append(data[i][1])
    return np.array(output)

euconst_data_s = sampling(euconst_data)
samsum_data_s = sampling(samsum_data)
hs_data_s = sampling(hs_data)

# f, axes = plt.subplots(1, 3, figsize=(15,5), dpi=300)
plt.figure(figsize=(15,5), dpi=300)
plt.rcParams.update({'font.size': 18})

# sub_graph = axes[0]
data = euconst_data_s
plt.subplot(1, 3, 1)
plt.plot(data, label='Loss',color='skyblue')
plt.title("Training on EUconst Dataset")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# sub_graph = axes[1]
data = samsum_data_s
plt.subplot(1, 3, 2)
plt.plot(data, label='Loss',color='skyblue')
plt.title("Training on SAMSum Dataset")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# sub_graph = axes[2]
data = hs_data_s
plt.subplot(1, 3, 3)
plt.plot(data, label='Loss',color='skyblue')
plt.title("Training on HS Dataset")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplots_adjust(wspace =0.4, hspace =0.3)#调整子图间距 (0,1)
plt.tight_layout(pad=1)#调整整体空白
plt.savefig("doc/fig.convergency.svg",format = "svg")
plt.savefig("doc/fig.convergency.jpg", format = "jpg", dpi=300)
plt.savefig("doc/fig.convergency.tif", format = "tif", dpi=300)
plt.show()