import numpy as np
import pandas as pd
from scipy.io import loadmat

def read_mex_data():

    path = "files/"
    #df = pd.read_csv(path + 'MEXswLatest.mat')

    data = loadmat(path + 'MEXswLatest.mat')
    print(data.keys())          # variables stored in the file
    time = data["time"]               # access a variable by name
    proton_density = data["NH"]
    alpha_density = data["NA"]
    proton_speed = data["VH"]
    alpha_speed = data["VA"]
    proton_temperature = data["TH"]
    alpha_temperature = data["TA"]
    t = np.datetime64("1970-01-01T00:00:00") + ((time - 719529) * 86400e3).astype("timedelta64[ms]")


    df = pd.DataFrame({ "datetime": t.ravel(),
                       "proton_density": proton_density.ravel(),
                       "alpha_density": alpha_density.ravel(),
                       "proton_speed": proton_speed.ravel(),
                       "alpha_speed": alpha_speed.ravel(),
                       "proton_temperature": proton_temperature.ravel(),
                       "alpha_temperature": alpha_temperature.ravel()
    })
    return df



if __name__ == "__main__":

    read_mex_data()