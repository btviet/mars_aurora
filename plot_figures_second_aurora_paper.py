import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import numpy as np

from matplotlib.colors import LogNorm
import seaborn as sns

from parameters import DATA_DIR
from read_mex_data import read_mex_data

FONTSIZE_AXES_TICKS = 10
FONTSIZE_AXES_LABELS = 10


obs_times = [
        datetime.strptime("2023-05-05 18:35:00", "%Y-%m-%d %H:%M:%S"),
        datetime.strptime("2023-05-10 22:56:00", "%Y-%m-%d %H:%M:%S"),
        datetime.strptime("2023-09-01 21:04:00", "%Y-%m-%d %H:%M:%S"),
        datetime.strptime("2024-03-18 06:45:00", "%Y-%m-%d %H:%M:%S"), # detection
    datetime.strptime("2024-05-18 00:33:00", "%Y-%m-%d %H:%M:%S"), # detection
    datetime.strptime("2024-05-20 01:15:00", "%Y-%m-%d %H:%M:%S"),
    datetime.strptime("2024-06-17 18:00:00", "%Y-%m-%d %H:%M:%S"),
    datetime.strptime("2024-08-04 00:21:00", "%Y-%m-%d %H:%M:%S"),
] #  Actual observation times 


donki_peak_times = [
        datetime.strptime("2023-05-05 21:00:00", "%Y-%m-%d %H:%M:%S"),
        datetime.strptime("2023-05-10 13:00:00", "%Y-%m-%d %H:%M:%S"),
        datetime.strptime("2023-09-01 17:00:00", "%Y-%m-%d %H:%M:%S"),
        datetime.strptime("2024-03-18 04:00:00", "%Y-%m-%d %H:%M:%S"), # detection
    datetime.strptime("2024-05-17 17:00:00", "%Y-%m-%d %H:%M:%S"), # detection
    datetime.strptime("2024-05-19 12:00:00", "%Y-%m-%d %H:%M:%S"),
    datetime.strptime("2024-06-16 18:00:00", "%Y-%m-%d %H:%M:%S"),
    datetime.strptime("2024-08-04 00:00:00", "%Y-%m-%d %H:%M:%S"),
] 


def read_maven_sep_flux_data():

    """
    Reads the merged processed data from the s1 sensor
    in the forward look direction on MAVEN/SEP
    """

    df = pd.read_csv(DATA_DIR / "processed_maven_sep_all.txt",
                     sep="\t", parse_dates=['date'])

    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def monster_figure():

    df = read_maven_sep_flux_data()
    current_date = datetime.strptime("2024-03-18 07:00:00", "%Y-%m-%d %H:%M:%S")
    start_date = current_date - pd.Timedelta(days=14)
    start_date = max(start_date, df.iloc[0]["date"])
    end_date = current_date + pd.Timedelta(days=14)
    

    df = df[(df['datetime']>=start_date) & (df['datetime'] <= end_date)]
   
    df = df.reset_index(drop=True)
    df_ion_flux = df.iloc[:,3:31]
    df_electron_flux = df.iloc[:,31:-2]
    df_datetime = df["datetime"]
    
    df_ion = pd.concat([df_datetime, df_ion_flux], axis=1)
    df_electron = pd.concat([df_datetime, df_electron_flux], axis=1)

    df_ion.set_index('datetime', inplace=True)
    df_electron.set_index('datetime', inplace=True)
    df_ion = df_ion.iloc[:, ::-1]
    df_electron = df_electron.iloc[:, ::-1]

    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    heatmap_ion = sns.heatmap(df_ion.T, cmap='plasma', cbar=False, norm=LogNorm(vmin=0.1, vmax=1e4), ax=axes[0])
    sns.heatmap(df_electron.T, cmap='plasma', cbar=False, norm=LogNorm(vmin=0.1, vmax=1e4), ax=axes[1])
    
    axes[1].legend()
    plt.subplots_adjust(hspace=0.05) 
    cbar = fig.colorbar(heatmap_ion.collections[0], ax=axes, orientation='vertical')
    cbar.set_label(r"$1/\mathrm{sec} \cdot \mathrm{cm}^2 \cdot \mathrm{ster} \cdot \mathrm{KeV}$", 
                   fontsize=FONTSIZE_AXES_LABELS)
    cbar.ax.tick_params(labelsize=FONTSIZE_AXES_TICKS) 


    axes[0].set_ylabel('Ion energy [keV]', fontsize=FONTSIZE_AXES_LABELS)
    axes[1].set_ylabel('Electron energy [keV]', fontsize=FONTSIZE_AXES_LABELS)

    # ADJUST THE TICKS HERE
    tick_index_list = []
    tick_label_list = []
    interval_length = end_date - start_date

    if interval_length >= pd.Timedelta(days=24):
        days_interval_ticks = 5
    else:
        days_interval_ticks = 4
    tick_date_range = pd.date_range(current_date-pd.Timedelta(days=2*days_interval_ticks), 
                                    current_date+pd.Timedelta(days=2*days_interval_ticks),
                                    freq=f'{days_interval_ticks}D')

   
    for tick_date in tick_date_range:
        midnight_tick_date = datetime.combine(tick_date, datetime.min.time())
        midnight_tick_index = df.index[df['datetime'] == midnight_tick_date].tolist()[0]
        tick_label_list.append(midnight_tick_date.date())
        tick_index_list.append(midnight_tick_index)



    ion_tick_labels = tick_label_list # df_ion.index[tick_index_].date  
    ion_tick_indices= tick_index_list


    axes[1].set_xticks(ion_tick_indices)
    axes[1].set_xticklabels(ion_tick_labels, fontsize=FONTSIZE_AXES_TICKS, 
                            rotation=10) 

    minor_tick_index_list = []
    minor_tick_date_range =  pd.date_range(start_date, end_date,freq='D')

    axes[0].set_xlabel('')  
    axes[1].set_xlabel('Date', fontsize=FONTSIZE_AXES_LABELS)

    ion_upper_bounds = [float(col.split('-')[1].split()[0]) for col in df_ion.columns]

    ion_axis_ticks = ion_upper_bounds[::3]
    ion_tick_indices = [i for i, value in enumerate(ion_upper_bounds) if value in ion_axis_ticks]
    ion_tick_labels = [int(value) if value.is_integer() else value for value in ion_axis_ticks]  
    axes[0].set_yticks(ion_tick_indices)
    axes[0].set_yticklabels(ion_tick_labels, fontsize=FONTSIZE_AXES_TICKS)

    # electron_lower_bounds = [float(col.split('-')[0]) for col in df_electron.columns]
    electron_upper_bounds = [float(col.split('-')[1].split()[0]) for col in df_electron.columns]
    electron_axis_ticks = electron_upper_bounds[::2]
    electron_tick_indices = [i for i, value in enumerate(electron_upper_bounds) if value in electron_axis_ticks]
    electron_tick_labels =  [int(value) if value.is_integer() else value for value in electron_axis_ticks] 
  
    axes[1].set_yticks(electron_tick_indices)
    axes[1].set_yticklabels(electron_tick_labels, fontsize=FONTSIZE_AXES_TICKS)
    axes[0].tick_params(which='major', axis='y', length=10)  
    axes[0].tick_params(which='minor', axis='y', length=6) 

    axes[1].tick_params(which='major', axis='y', length=10)  
    axes[1].tick_params(which='minor', axis='y', length=6) 

    axes[1].tick_params(which='minor', axis='x', length=6) 
    axes[1].tick_params(which='major', axis='x', length=10) 

    plt.show()



def single_figure():

    count=6
    # donki_predict_date = donki_peak_times[count]
    # count += 1
    df = read_maven_sep_flux_data()
    df_mex = read_mex_data()
    current_date = datetime.strptime("2024-05-20 01:15:00", "%Y-%m-%d %H:%M:%S")
    # Slice in time
    start_date = current_date - pd.Timedelta(days=2)
    #start_date = datetime.combine(start_date, datetime.min.time())
    start_date = max(start_date, df.iloc[0]["date"])
    end_date = current_date + pd.Timedelta(days=2)
    #end_date = datetime.combine(end_date, datetime.min.time())
    df = df[(df['datetime']>=start_date) & (df['datetime'] <= end_date)]


    df_mex = df_mex[(df_mex['datetime']>=start_date) & (df_mex['datetime'] <= end_date)]

    df = df.reset_index(drop=True)
    df_mex = df_mex.reset_index(drop=True)
    df_ion_flux = df.iloc[:,3:31] # Ion energy channels
    df_electron_flux = df.iloc[:,31:-2] # Electron energy channels
    df_datetime = df["datetime"] # Datetime column
    
    df_ion = pd.concat([df_datetime, df_ion_flux], axis=1)
    df_electron = pd.concat([df_datetime, df_electron_flux], axis=1)

    df_ion.set_index('datetime', inplace=True)
    df_electron.set_index('datetime', inplace=True)
    df_mex.set_index('datetime', inplace=True)


    # df_ion = df_ion.iloc[:, ::-1]
    # df_electron = df_electron.iloc[:, ::-1]

    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

    x_ion = df_ion.index
    y_ion = np.arange(df_ion.shape[1])
    mesh = axes[0].pcolormesh(x_ion, y_ion, df_ion.T.to_numpy(),
                            cmap="nipy_spectral", norm=LogNorm(vmin=0.01, vmax=1e4), shading="nearest")

    x_el = df_electron.index
    y_el = np.arange(df_electron.shape[1])
    axes[1].pcolormesh(x_el, y_el, df_electron.T.to_numpy(),
                    cmap="nipy_spectral", norm=LogNorm(vmin=0.01, vmax=1e4), shading="nearest")

    axes[2].plot(df_mex.index, df_mex["proton_density"], label=r'$H^+$',
                    marker='o', markersize=3)
    axes[2].plot(df_mex.index, df_mex["alpha_density"], label=r'$He^{2+}$',
                    marker='o', markersize=3)


    axes[2].axhline(15,
            linestyle='solid',
            color='green',
            label='Criteria')
    axes[2].set_yscale("log")
    axes[2].set_ylim([1e-2, 1e2])


    axes[3].plot(df_mex.index, df_mex["proton_speed"], label=r'$H^+$',
                    marker='o', markersize=3)
    axes[3].plot(df_mex.index, df_mex["alpha_speed"], label=r'$He^{2+}$',
                    marker='o', markersize=3)

    axes[3].axhline(550,
            linestyle='solid',
            color='green',
            label='Criteria')
    
    axes[3].set_ylim([0, 1000])

    axes[4].plot(df_mex.index, df_mex["proton_temperature"], label=r'$H^+$',
                    marker='o', markersize=3)
    
    axes[4].plot(df_mex.index, df_mex["alpha_temperature"], label=r'$He^{2+}$',
                    marker='o', markersize=3)
    axes[4].axhline(200000,
            linestyle='solid',
            color='green',
            label='Criteria')
    axes[4].set_yscale("log")
    axes[4].set_ylim([1e3, 1e7])


    for ax in axes:
        ax.axvline(x=current_date,
                linestyle='solid',
                color='black',
                label='')
        
        #ax.axvline(x=donki_predict_date,
        #        linestyle='dashed',
        #        color='black',
        #        label='')

        donki_date = datetime.strptime("2024-05-20 20:00:00", "%Y-%m-%d %H:%M:%S"),
        ax.axvline(x=donki_date,
                linestyle='dashed',
                color='black',
                label='')

        # Colorbar
        cax = axes[0].inset_axes([1.02, -1, 0.02, 2])   # [x, y, width, height] in axes coords
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label(r"$1/\mathrm{sec} \cdot \mathrm{cm}^2 \cdot \mathrm{ster} \cdot \mathrm{KeV}$")

        fig.subplots_adjust(right=0.9)   # leave room on the right so it isn't clipped

        #cbar = fig.colorbar(mesh, ax=axes[0:2], orientation="vertical")


        ion_upper_bounds = [float(col.split('-')[1].split()[0]) for col in df_ion.columns]
        # Date ticks: one major tick per day at midnight
        #axes[2].xaxis.set_major_locator(mdates.DayLocator())
        #axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        #axes[2].set_xlim(start_date, end_date)

        #ion_upper_bounds = [float(col.split('-')[1].split()[0]) for col in df_ion.columns]

        ion_axis_ticks = ion_upper_bounds[::3]
        ion_tick_indices = [i for i, value in enumerate(ion_upper_bounds) if value in ion_axis_ticks]
        ion_tick_labels = [int(value) if value.is_integer() else value for value in ion_axis_ticks]  
        axes[0].set_yticks(ion_tick_indices)
        axes[0].set_yticklabels(ion_tick_labels, fontsize=FONTSIZE_AXES_TICKS)

        # electron_lower_bounds = [float(col.split('-')[0]) for col in df_electron.columns]
        electron_upper_bounds = [float(col.split('-')[1].split()[0]) for col in df_electron.columns]
        electron_axis_ticks = electron_upper_bounds[::2]
        electron_tick_indices = [i for i, value in enumerate(electron_upper_bounds) if value in electron_axis_ticks]
        electron_tick_labels =  [int(value) if value.is_integer() else value for value in electron_axis_ticks] 
    
        axes[1].set_yticks(electron_tick_indices)
        axes[1].set_yticklabels(electron_tick_labels, fontsize=FONTSIZE_AXES_TICKS)
        axes[0].tick_params(which='major', axis='y', length=10)  
        axes[0].tick_params(which='minor', axis='y', length=6) 

        axes[1].tick_params(which='major', axis='y', length=10)  
        axes[1].tick_params(which='minor', axis='y', length=6) 

        #axes[1].tick_params(which='minor', axis='x', length=6) 
        #axes[1].tick_params(which='major', axis='x', length=10) 


        axes[0].set_ylabel('Ion energy [keV]', fontsize=FONTSIZE_AXES_LABELS)
        axes[1].set_ylabel('Electron energy [keV]', fontsize=FONTSIZE_AXES_LABELS)
        axes[2].set_ylabel('Number density [cm^-3]', fontsize=FONTSIZE_AXES_LABELS)
        axes[3].set_ylabel('Bulk speed [km/s]', fontsize=FONTSIZE_AXES_LABELS)
        axes[4].set_ylabel('Temperature [K]', fontsize=FONTSIZE_AXES_LABELS)
        axes[4].set_xlabel('Date', fontsize=FONTSIZE_AXES_LABELS)
        axes[2].legend(loc='upper left')
        axes[3].legend(loc='upper left')
        axes[4].legend(loc='upper left')

        axes[3].tick_params(
            right=True,
            labelright=True,

        )

        axes[3].yaxis.set_label_position("right")

        plt.subplots_adjust(left=0.1, bottom=0.05, 
                            hspace=0.1)

    fig.suptitle(f'Attempt #{count}',
                    fontsize=18,
                    y=0.91)
    plt.show()
        #plt.savefig(f'attempt_{count}_complete.png', 
        #                    dpi=300, transparent=False)
        


def test_plot():
    df = read_mex_data()
    current_date = datetime.strptime("2024-03-18 07:00:00", "%Y-%m-%d %H:%M:%S")
    # Slice in time
    start_date = current_date - pd.Timedelta(days=2)
    start_date = datetime.combine(start_date, datetime.min.time())

    end_date = current_date + pd.Timedelta(days=2)
    end_date = datetime.combine(end_date, datetime.min.time())
    df = df[(df['datetime']>=start_date) & (df['datetime'] <= end_date)]

    print(df)
    fig, ax1 = plt.subplots(figsize=(10, 7.5))
    #ax2=ax1.twinx()
    ax1.plot(df["datetime"], df["proton_density"])
    ax1.plot(df["datetime"], df["alpha_density"])
    ax1.set_yscale('log')
    ax1.set_ylim([1e-2, 1e2])
    plt.show()

if __name__ == "__main__":
    # monster_figure()
    single_figure()
    # df = read_maven_sep_flux_data()
    # print(df)