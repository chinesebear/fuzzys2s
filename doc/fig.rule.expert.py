import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/times.ttf')
plt.rc('font',family='Times New Roman')
import numpy as np
import seaborn as sns

data_matrix = []
data_matrix.append(np.mat([
            [0.0, 1, 25],
            [0, 26,0],
            [46, 1, 1]
        ]))
data_matrix.append(np.mat([
            [0.0, 1, 30],
            [0, 35,0],
            [33, 0, 1]]))  
data_matrix.append(np.mat([
            [0.0, 3, 40],
            [0, 26,1],
            [28, 1, 1]])) 
data_matrix.append(np.mat([
            [0.0, 1, 24],
            [0, 42,0],
            [31, 1, 1]]))  

datasets=["WMT14", "Tatoeba", "EUconst", "Ubuntu"]

for i in range(4):
    plt.subplot(2, 2, i+1)    
    sns.heatmap(data_matrix[i],
                    linewidth=0.5,
                    # 将具体的数字写在对应的表格中，%.1f 指定了样式，在较复杂的样式中可以去掉
                    annot=np.array(['%d' % point for point in np.array(data_matrix[i].ravel())[0]]).reshape(np.shape(data_matrix[i])),
                    # 这里必须置空，否则会出现问题
                    fmt='',
                    yticklabels=["Short", "Medium", "Long"],
                    # 如果 usetext=True, 这里可以使用 latex 语法比如 $\leq$ = <
                    xticklabels=["Long", "Medium", "Short"],
                    vmax=100,
                    vmin=0,
                    # cmap 决定了注意力图的色调
                    cmap="YlGnBu")
    plt.title(datasets[i])
    plt.ylabel("Human Expert", labelpad=5)
    plt.xlabel("GenFS", labelpad=5)

plt.subplots_adjust(wspace =0.5, hspace =0.3)#调整子图间距 (0,1)
plt.tight_layout(pad=0)#调整整体空白
plt.savefig("doc/fig.rule expert.svg", format = "svg")
plt.savefig("doc/fig.rule expert.tif", format = "tif",dpi=300)
plt.savefig("doc/fig.rule expert.jpg", format = "jpg",dpi=300)
plt.show()