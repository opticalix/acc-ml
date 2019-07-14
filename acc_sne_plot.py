import numpy as np
from tsfeature import feature_core
import acc_pre_proc
import numpy as np
import pandas as pd
import os
import math
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

b = feature_core.sequence_feature(np.array([1,2,3,4,5]), 5, 4)
