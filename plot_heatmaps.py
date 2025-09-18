import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm
from datetime import datetime
import seaborn as sns
import matplotlib.patches as patches

from read_maven_sep import read_maven_sep_flux_data
from parameters import (
    FONTSIZE_AXES_LABELS,
    FONTSIZE_AXES_TICKS,
    FONTSIZE_TITLE
)

from obs_times import obs_times, obs_closest_times

def plot_heatmaps_all_attempts():
    """
    In one figure,
    plots heatmaps of all 8 attempts, for both ion and electron channels
    """
    
    fig, axes = plt.subplots(8, 2, figsize=(20, 20), sharex=True)
    electron_column = axes[:, 0]
    ion_column = axes[:, 1]
    for i in range(8): # for each of the attempts
        obs_start =  obs_times[i] - pd.Timedelta(hours=1)
        obs_end = obs_times[i] + pd.Timedelta(hours=1)
        filename = 'maven_f_flux_hr_aurora_attempt_' + str(i+1)
        temp_df =  read_maven_sep_flux_data(filename) 
        # temp_df = temp_df[temp_df["20.1-21.5 keV/n,Ion"]<99900000.0]

        numeric_cols = temp_df.select_dtypes(include='number').columns # only numeric columns
        temp_df[numeric_cols] = temp_df[numeric_cols].mask(temp_df[numeric_cols] > 999999)

        indices= temp_df[(temp_df['datetime']>=obs_start) & (temp_df['datetime'] <= obs_end)].index

        closest_obs_sample_index = indices[-1]
        index_range = [closest_obs_sample_index -24*5, closest_obs_sample_index +24*5+1]
        sliced_df = temp_df.iloc[index_range[0]:index_range[1]].reset_index()
        
        df_ion_flux = sliced_df.iloc[:,4:32]
        print("df_ion_flux: \n", df_ion_flux.columns)
        df_electron_flux = sliced_df.iloc[:,32:-2]
        print("df_electron_flux: \n", df_electron_flux.columns)
        df_datetime = sliced_df["datetime"]


        df_ion = pd.concat([df_datetime, df_ion_flux], axis=1)
        df_electron = pd.concat([df_datetime, df_electron_flux], axis=1)

        df_ion.set_index('datetime', inplace=True)
        df_electron.set_index('datetime', inplace=True)

        df_ion = df_ion.iloc[:, ::-1]
        df_electron = df_electron.iloc[:, ::-1]
        heatmap_electron = sns.heatmap(df_electron.T, 
                                       cmap='nipy_spectral', #nipy_spectral'
                                       cbar=False, 
                                       norm=LogNorm(vmin=0.01, vmax=1e4), 
                                       mask=df_electron.T.isna(),
                                       ax=electron_column[i])
        
        heatmap_ion = sns.heatmap(df_ion.T, 
                                       cmap='nipy_spectral', 
                                       cbar=False, 
                                       norm=LogNorm(vmin=0.01, vmax=1e4), 
                                       mask=df_ion.T.isna(),
                                       ax=ion_column[i])


        # Get axis limits
        x0, x1 = ion_column[i].get_xlim()
        y0, y1 = ion_column[i].get_ylim()

        # Draw rectangle border
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                linewidth=2, edgecolor='black',
                                facecolor='none', zorder=10)
        ion_column[i].add_patch(rect)

        # electron panel border
        x0, x1 = electron_column[i].get_xlim()
        y0, y1 = electron_column[i].get_ylim()


        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                linewidth=2, edgecolor='black',
                                facecolor='none', zorder=10)
        electron_column[i].add_patch(rect)


        x_position = list(df_ion.index).index(obs_closest_times[i])
    
        ion_column[i].axvline(x=x_position, color='white', 
                              linestyle='dashed', linewidth=2)
        
        x_position = list(df_electron.index).index(obs_closest_times[i])
    
        electron_column[i].axvline(x=x_position, color='white', 
                              linestyle='dashed', linewidth=2)
        
        #ion_column[i].axvline(x=obs_times[i],
        #        linestyle='dashed',
        #        linewidth=3,
        #        color='white')
        
        electron_upper_bounds = [float(col.split('-')[1].split()[0]) 
                                 for col in df_electron.columns]
        electron_axis_ticks = electron_upper_bounds[::2]
        electron_tick_indices = [i for i, value in enumerate(electron_upper_bounds) 
                                 if value in electron_axis_ticks]

        electron_tick_labels =  [int(value) if value.is_integer() else value for value in electron_axis_ticks] 
        electron_column[i].set_yticks(electron_tick_indices)
        electron_column[i].set_yticklabels(electron_tick_labels, fontsize=FONTSIZE_AXES_TICKS)
        electron_column[i].set_xlabel("")
        electron_column[i].tick_params(which='major', axis='y', length=10)  
        electron_column[i].tick_params(which='minor', axis='y', length=6) 
        electron_column[i].tick_params(which='major', axis='x', length=10)  
        electron_column[i].tick_params(which='minor', axis='x', length=6) 

        ion_upper_bounds = [float(col.split('-')[1].split()[0]) 
                                 for col in df_ion.columns]
        ion_axis_ticks = ion_upper_bounds[::4]
        ion_tick_indices = [i for i, value in enumerate(ion_upper_bounds) 
                                 if value in ion_axis_ticks]

        ion_tick_labels =  [int(value) if value.is_integer() else value for value in ion_axis_ticks]

    
        ion_column[i].set_yticks(ion_tick_indices)
        ion_column[i].set_yticklabels(ion_tick_labels, fontsize=FONTSIZE_AXES_TICKS)
        ion_column[i].set_xlabel("")
        ion_column[i].tick_params(which='major', axis='y', length=10)  
        ion_column[i].tick_params(which='minor', axis='y', length=6) 
        ion_column[i].tick_params(which='major', axis='x', length=10)  
        ion_column[i].tick_params(which='minor', axis='x', length=6) 


    #heatmap_ion = sns.heatmap(df_ion.T, cmap='plasma', cbar=False, norm=LogNorm(vmin=0.1, vmax=1e4), ax=axes[0])
    


    #axes[1].legend()

    plt.subplots_adjust(hspace=0.1) 
    fig.subplots_adjust(wspace=0.4) 
    ### COLORBAR
    #cbar = fig.colorbar(heatmap_electron.collections[0], ax=axes, orientation='vertical')
    #cbar.set_label(r"$1/\mathrm{sec} \cdot \mathrm{cm}^2 \cdot \mathrm{ster} \cdot \mathrm{KeV}$", 
    #               fontsize=FONTSIZE_AXES_LABELS)
    #cbar.ax.tick_params(labelsize=FONTSIZE_AXES_TICKS) 
    
    cbar_ax = fig.add_axes([0.95, 0.12, 0.02, 0.75])  # Adjust width here (3rd value)

    cbar = fig.colorbar(heatmap_electron.collections[0], cax=cbar_ax, orientation='vertical')
    cbar.set_label(r"$1/\mathrm{sec} \cdot \mathrm{cm}^2 \cdot \mathrm{ster} \cdot \mathrm{KeV}$",
               fontsize=FONTSIZE_AXES_LABELS)
    cbar.ax.tick_params(labelsize=FONTSIZE_AXES_TICKS)



    x_axis = df_electron.index  # these become columns after .T
    tick_indices = list(range(len(x_axis)))
    tick_labels = [dt.strftime("%Y-%m-%d\n%H:%M") for dt in x_axis]
    tick_indices_thinned = tick_indices[::24]
    tick_labels_thinned = tick_labels[::24]
    tick_labels_thinned = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    electron_column[-1].set_xticks(tick_indices_thinned)
    electron_column[-1].set_xticklabels(tick_labels_thinned, rotation=0, ha='center', fontsize=FONTSIZE_AXES_TICKS)


    x_axis = df_ion.index  # these become columns after .T
    tick_indices = list(range(len(x_axis)))
    tick_labels = [dt.strftime("%Y-%m-%d\n%H:%M") for dt in x_axis]
    tick_indices_thinned = tick_indices[::24]
    tick_labels_thinned = tick_labels[::24]
    tick_labels_thinned = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    ion_column[-1].set_xticks(tick_indices_thinned)
    ion_column[-1].set_xticklabels(tick_labels_thinned, rotation=0, ha='center', fontsize=FONTSIZE_AXES_TICKS)


    electron_column[4].set_ylabel('Electron energy [keV]', fontsize=FONTSIZE_AXES_LABELS)
    ion_column[4].set_ylabel('Ion energy [keV]', fontsize=FONTSIZE_AXES_LABELS)
    #fig.supylabel("Electron energy [keV]", fontsize=FONTSIZE_AXES_LABELS)
    #fig.text(0.96, 0.5, "Another Y Label", va='center', rotation='vertical',
    #     fontsize=FONTSIZE_AXES_LABELS)
    electron_column[-1].set_xlabel('Day', fontsize=FONTSIZE_AXES_LABELS)
    ion_column[-1].set_xlabel('Day', fontsize=FONTSIZE_AXES_LABELS)



    fig.suptitle("MAVEN/SEP ion and electron fluxes for all aurora attempts",
                  fontsize=FONTSIZE_TITLE, y=0.90)
    plt.gca().grid(False)

    plt.savefig("figures/heatmap_all_attempts.png", bbox_inches='tight')
    plt.show()


def plot_heatmaps_all_attempts_electrons_only():
    """
    In one figure,
    plots heatmaps of all 8 attempts, for electron channels only
    """
    
    color_ion_low = '#4daf4a' #"green"
    color_ion_high = '#377eb8' #"blue"
    color_electron_low = '#ff7f00' #"orange"
    color_electron_high = '#e41a1c' #"red"
    
    colors_list = [color_ion_low, 
                   color_ion_high, 
                   color_electron_low, 
                   color_electron_high]
    obs_times = [
        datetime.strptime("2023-05-05 18:35:00", "%Y-%m-%d %H:%M:%S"),
        datetime.strptime("2023-05-10 22:56:00", "%Y-%m-%d %H:%M:%S"),
        datetime.strptime("2023-09-01 21:04:00", "%Y-%m-%d %H:%M:%S"),
        datetime.strptime("2024-03-18 06:45:00", "%Y-%m-%d %H:%M:%S"),
    datetime.strptime("2024-05-18 00:33:00", "%Y-%m-%d %H:%M:%S"),
    datetime.strptime("2024-05-20 01:15:00", "%Y-%m-%d %H:%M:%S"),
    datetime.strptime("2024-06-17 18:00:00", "%Y-%m-%d %H:%M:%S"),
    datetime.strptime("2024-08-04 00:21:00", "%Y-%m-%d %H:%M:%S"),
    ]

    # filename_1 = 'maven_f_flux_hr_aurora_attempt_1' 

    # start_date_1 = current_date_1 - pd.Timedelta(days=5)
    #end_date_1 = current_date_1 + pd.Timedelta(days=5)
    fig, axes = plt.subplots(8, 1, figsize=(12, 20), sharex=True)
    for i in range(8):
        obs_start =  obs_times[i] - pd.Timedelta(hours=1)
        obs_end = obs_times[i] + pd.Timedelta(hours=1)
        filename = 'maven_f_flux_hr_aurora_attempt_' + str(i+1)
        temp_df =  read_maven_sep_flux_data(filename) 
        # temp_df = temp_df[temp_df["20.1-21.5 keV/n,Ion"]<99900000.0]

        numeric_cols = temp_df.select_dtypes(include='number').columns # only numeric columns
        temp_df[numeric_cols] = temp_df[numeric_cols].mask(temp_df[numeric_cols] > 999999)

        indices= temp_df[(temp_df['datetime']>=obs_start) & (temp_df['datetime'] <= obs_end)].index

        closest_obs_sample_index = indices[-1]
        index_range = [closest_obs_sample_index -24*5, closest_obs_sample_index +24*5+1]
        sliced_df = temp_df.iloc[index_range[0]:index_range[1]].reset_index()
        print("sliced_df: ", sliced_df)
        print(sliced_df.columns)
        df_ion_flux = sliced_df.iloc[:,3:31]
        df_electron_flux = sliced_df.iloc[:,32:-2]
        df_datetime = sliced_df["datetime"]


        df_ion = pd.concat([df_datetime, df_ion_flux], axis=1)
        df_electron = pd.concat([df_datetime, df_electron_flux], axis=1)

        df_ion.set_index('datetime', inplace=True)
        df_electron.set_index('datetime', inplace=True)

        df_ion = df_ion.iloc[:, ::-1]
        df_electron = df_electron.iloc[:, ::-1]
        print(df_electron.columns, "the columns")
        heatmap_electron = sns.heatmap(df_electron.T, 
                                       cmap='nipy_spectral', 
                                       cbar=False, 
                                       norm=LogNorm(vmin=0.01, vmax=1e4), 
                                       mask=df_electron.T.isna(),
                                       ax=axes[i])
        electron_upper_bounds = [float(col.split('-')[1].split()[0]) for col in df_electron.columns]
        electron_axis_ticks = electron_upper_bounds[::2]
        electron_tick_indices = [i for i, value in enumerate(electron_upper_bounds) if value in electron_axis_ticks]

        electron_tick_labels =  [int(value) if value.is_integer() else value for value in electron_axis_ticks] 
        axes[i].set_yticks(electron_tick_indices)
        axes[i].set_yticklabels(electron_tick_labels, fontsize=FONTSIZE_AXES_TICKS)
        axes[i].set_xlabel("")
        axes[i].tick_params(which='major', axis='y', length=10)  
        axes[i].tick_params(which='minor', axis='y', length=6) 
        axes[i].tick_params(which='major', axis='x', length=10)  
        axes[i].tick_params(which='minor', axis='x', length=6) 
    
    #heatmap_ion = sns.heatmap(df_ion.T, cmap='plasma', cbar=False, norm=LogNorm(vmin=0.1, vmax=1e4), ax=axes[0])
    


    #axes[1].legend()

    plt.subplots_adjust(hspace=0.1) 
    ### COLORBAR
    cbar = fig.colorbar(heatmap_electron.collections[0], ax=axes, orientation='vertical')
    cbar.set_label(r"$1/\mathrm{sec} \cdot \mathrm{cm}^2 \cdot \mathrm{ster} \cdot \mathrm{KeV}$", 
                   fontsize=FONTSIZE_AXES_LABELS)
    cbar.ax.tick_params(labelsize=FONTSIZE_AXES_TICKS) 


    x_axis = df_electron.index  # these become columns after .T
    tick_indices = list(range(len(x_axis)))
    tick_labels = [dt.strftime("%Y-%m-%d\n%H:%M") for dt in x_axis]
    tick_indices_thinned = tick_indices[::24]
    tick_labels_thinned = tick_labels[::24]
    tick_labels_thinned = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    axes[-1].set_xticks(tick_indices_thinned)
    axes[-1].set_xticklabels(tick_labels_thinned, rotation=0, ha='center', fontsize=FONTSIZE_AXES_TICKS)

    
    #axes[4].set_ylabel('Electron energy [keV]', fontsize=FONTSIZE_AXES_LABELS)
    fig.supylabel("Electron energy [keV]", fontsize=FONTSIZE_AXES_LABELS)
    axes[-1].set_xlabel('Day', fontsize=FONTSIZE_AXES_LABELS)





    fig.suptitle("MAVEN/SEP electron fluxes for all aurora attempts",
                  fontsize=FONTSIZE_TITLE, y=0.90)
    plt.gca().grid(False)

    plt.savefig("figures/heatmap_all_attempts_electrons.png",bbox_inches='tight')
    plt.show()



def plot_heatmap_one_attempt(filename):
    """
    In one figure, plot a heatmap for ion channels, 
    and a heatmap for electron channels,
    for one observation attempt
    """
    df = read_maven_sep_flux_data(filename)
    
    current_date = obs_times[0] #  NEED TO CHANGE THIS TO CORRESPOND WITH FILENAME
    start_date = current_date - pd.Timedelta(days=5)
    end_date = current_date + pd.Timedelta(days=5)
    df = df[(df['datetime']>=start_date) & (df['datetime'] <= end_date)]

    df_ion_flux = df.iloc[:,3:31]
    df_electron_flux = df.iloc[:,31:-2]
    df_datetime = df["datetime"]
    
    df_ion = pd.concat([df_datetime, df_ion_flux], axis=1)
    df_electron = pd.concat([df_datetime, df_electron_flux], axis=1)

    df_ion.set_index('datetime', inplace=True)
    df_electron.set_index('datetime', inplace=True)
    df_ion = df_ion.iloc[:, ::-1]
    df_electron = df_electron.iloc[:, ::-1]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
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
    ion_tick_indices = np.arange(0, len(df_ion.index), 48)
    ion_tick_labels = df_ion.index[ion_tick_indices].date 

    axes[1].set_xticks(ion_tick_indices)
    axes[1].set_xticklabels(ion_tick_labels, fontsize=FONTSIZE_AXES_TICKS, 
                            rotation=10)  


    axes[0].set_xlabel('')  
    axes[1].set_xlabel('Date', fontsize=FONTSIZE_AXES_LABELS)

    ion_upper_bounds = [float(col.split('-')[1].split()[0]) for col in df_ion.columns]

    ion_axis_ticks = ion_upper_bounds[::3]
    ion_tick_indices = [i for i, value in enumerate(ion_upper_bounds) if value in ion_axis_ticks]
    ion_tick_labels = [int(value) if value.is_integer() else value for value in ion_axis_ticks]  
    axes[0].set_yticks(ion_tick_indices)
    axes[0].set_yticklabels(ion_tick_labels, fontsize=FONTSIZE_AXES_TICKS)

    electron_lower_bounds = [float(col.split('-')[0]) for col in df_electron.columns]
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


    # CHANGE FIGURE TITLE HERE
    fig.suptitle("MAVEN/SEP hourly fluxes for attempt #8",
                  fontsize=FONTSIZE_TITLE, y=0.95)
    plt.gca().grid(False)

    # CHANGE FIGURE FILENAME HERE
    plt.savefig('figures/aurora_attempt_8.png',
                 dpi=300, transparent=False)
    plt.show()



if __name__ == "__main__":
    # plot_heatmaps_all_attempts()
    # plot_heatmaps_all_attempts_electrons_only()
    filename = "maven_f_flux_hr_aurora_attempt_1"
    plot_heatmap_one_attempt(filename)