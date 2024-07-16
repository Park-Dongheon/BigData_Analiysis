from IPython.display import set_matplotlib_formats, display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import mglearn
from cycler import cycler
# pandas, numpy, matplotlib Setting

set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 300 # 해상도 크기
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.rcParams['font.size'] = 9
plt.rcParams['figure.dpi'] = 100
plt.rcParams['image.cmap'] = "viridis"
plt.rcParams['image.interpolation'] = "none"
plt.rcParams['savefig.bbox'] = "tight"
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['legend.numpoints'] = 1
plt.rc('axes', prop_cycle=(
    cycler('color', mglearn.plot_helpers.cm_cycle.colors) +
    cycler('linestyle', ['-', '-', "--", (0, (3, 3)), (0, (1.5, 1.5))])))


np.set_printoptions(precision=3, suppress=True) # 소숫점 이하 3자리까지 

pd.set_option("display.max_columns", 8)
pd.set_option('display.precision', 2)

__all__ = ['np', 'mglearn', 'display', 'plt', 'pd']
