
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO



def read_maven_sep_flux_data(filename):
    """
    Reads files from
    https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi

    Parameters
    ----------
    filename: str


    Returns
    -------
    pd.DataFrame
        Contains the MAVEN/SEP flux data
        
    """
    path = "omniweb/"  # Change filepath to where files are located
    fmt_filename = filename + '.fmt'
    data_filename = filename + '.lst'

    fmt_file = path + fmt_filename
    data_file = path + data_filename

    with open(fmt_file, 'r') as f:
        # Skipping the rows before the relevant info
        lines = f.readlines()[4:]
    headers = []
    for line in lines:
        elements = line.strip().split()
        if "I" in elements[-1:][0]:
                # Year, DOY, HOUR are integers
            headers.append(elements[1])
        elif len(elements) == 5:
                energy_range = "".join(elements[1:-2])
                unit = elements[-2:-1][0]
                item_header = str(energy_range) + " " + str(unit)
                headers.append(item_header)
        elif len(elements) == 4:
                item_header = elements[1] + " " + elements[-2]
                headers.append(item_header)  
        else:
            raise Exception("MAVEN/SEP format error")
    with open(data_file, 'r') as f:
        data = f.read()
    df = pd.read_csv(StringIO(data), 
                     sep='\\s+',
                     names=headers)

    df['date'] = pd.to_datetime(df['YEAR'].astype(str) + df['DOY'].astype(str), format='%Y%j')
    df['datetime'] = df['date'] + pd.to_timedelta(df['Hour'], unit='h')
    return df



if __name__ == "__main__":
    filename = "maven_f_flux_hr_aurora_attempt_1"
    df = read_maven_sep_flux_data(filename)
    print(df)