import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import _pickle as pickle
from methods import *

np.random.seed(12345)


state = {"n_f" : 492, "n_p" : 4, "n_m" : 4, "p_f" : 10, "p" : 10,}
params = {"sigma_eps" : 0.005, "sigma_mu" : 0.0, "t_c": 0.001, "gamma" : 0.01, 
          "beta": 4, "R": 0.0004, "s": 0.75, "alpha_1": 0.6, "alpha_2": 1.5, 
          "alpha_3": 1, "v_1": 2, "v_2": 0.6, "dt": 0.002}
totalT = 1e7

model = LuxMarchesiModel(state, params)
history = model.simulate(totalT, 1)

# Save the data
with open("./files/simul2.pkl", "wb") as f:
    pickle.dump(history, f)
