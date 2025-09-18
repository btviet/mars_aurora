
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from datetime import datetime

from read_maven_sep import read_maven_sep_flux_data
from parameters import (
    FONTSIZE_AXES_LABELS,
    FONTSIZE_AXES_TICKS,
    FONTSIZE_TITLE
)

from obs_times import obs_times
def plot_integrated_electron_channels():
    """
    Sums all the electron flux channels
    for the May event and the March event.
    Plots the summed channels in one plot
        
    """

    fig, ax1 = plt.subplots(figsize=(12, 6))
    # Aurora detected event in May, observation, 5th attempt
    obs_start =  obs_times[4] - pd.Timedelta(hours=1)
    obs_end = obs_times[4] + pd.Timedelta(hours=1)
    filename = 'maven_f_flux_hr_aurora_attempt_5'
    may_df =  read_maven_sep_flux_data(filename) 
    numeric_cols = may_df.select_dtypes(include='number').columns # only numeric columns
    may_df[numeric_cols] = may_df[numeric_cols].mask(may_df[numeric_cols] > 999999)
    indices= may_df[(may_df['datetime']>=obs_start) & (may_df['datetime'] <= obs_end)].index
    closest_obs_sample_index = indices[-1]
    print("closest sample:", may_df.iloc[closest_obs_sample_index])
    index_range = [closest_obs_sample_index -24*5, closest_obs_sample_index +24*5+1]
    #index_range = [closest_obs_sample_index -24*0, closest_obs_sample_index +24*2]
    sliced_df = may_df.iloc[index_range[0]:index_range[1]].reset_index()
    df_electron_flux = sliced_df.iloc[:,32:-2]
    df_ion_flux = sliced_df.iloc[:,4:32]
    df_datetime = sliced_df["datetime"]
    df_electron_may = pd.concat([df_datetime, df_electron_flux], axis=1)
    df_ion_may = pd.concat([df_datetime, df_ion_flux], axis=1)

    df_electron_may.set_index('datetime', inplace=True)
    df_electron_may = df_electron_may.iloc[:, ::-1]
        
    df_ion_may.set_index('datetime', inplace=True)
    df_ion_may = df_ion_may.iloc[:, ::-1]    


    df_electron_may["sum"] = df_electron_may.sum(axis=1)
    df_electron_may.reset_index(inplace=True)


    # 4th attempt, aurora march observation
    obs_start =  obs_times[3] - pd.Timedelta(hours=1)
    obs_end = obs_times[3] + pd.Timedelta(hours=1)
    filename = 'maven_f_flux_hr_aurora_attempt_4'
    march_df =  read_maven_sep_flux_data(filename) 
    numeric_cols = march_df.select_dtypes(include='number').columns # only numeric columns
    march_df[numeric_cols] = march_df[numeric_cols].mask(march_df[numeric_cols] > 999999)
    indices= march_df[(march_df['datetime']>=obs_start) & (march_df['datetime'] <= obs_end)].index
    closest_obs_sample_index = indices[-1]
    print("closest sample:", march_df.iloc[closest_obs_sample_index])
    index_range = [closest_obs_sample_index -24*5, closest_obs_sample_index +24*5+1]

    # index_range = [closest_obs_sample_index -24*0, closest_obs_sample_index +24*2]
    sliced_df = march_df.iloc[index_range[0]:index_range[1]].reset_index()
    df_electron_flux = sliced_df.iloc[:,32:-2]
    df_ion_flux = sliced_df.iloc[:,4:32]
    df_datetime = sliced_df["datetime"]
    df_electron_march = pd.concat([df_datetime, df_electron_flux], axis=1)
    df_ion_march = pd.concat([df_datetime, df_ion_flux], axis=1)

    df_electron_march.set_index('datetime', inplace=True)
    df_electron_march = df_electron_march.iloc[:, ::-1]
        
    df_ion_march.set_index('datetime', inplace=True)
    df_ion_march = df_ion_march.iloc[:, ::-1]

    df_electron_march["sum"] = df_electron_march.sum(axis=1)
    df_electron_march.reset_index(inplace=True)


    print(df_electron_march[["datetime", "sum"]].iloc[115:125])
    print(df_electron_may[["datetime", "sum"]].iloc[115:125])
    ax1.plot(df_electron_may.index, df_electron_may["sum"], label="May 2024")
    ax1.plot(df_electron_march.index, df_electron_march["sum"], label="March 2024")


    ax1.axvline(x=120,
                linestyle='dashed',
                color='black',
                label='Observation time (next full hour)')


    x_axis = df_electron_may["datetime"]
    tick_indices = list(range(len(x_axis)))
    tick_labels = [dt.strftime("%Y-%m-%d\n%H:%M") for dt in x_axis]
    tick_indices_thinned = tick_indices[::24]
    tick_labels_thinned = tick_labels[::24]
    tick_labels_thinned = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    ax1.set_xticks(tick_indices_thinned)
    ax1.set_xticklabels(tick_labels_thinned, rotation=0, ha='center', fontsize=FONTSIZE_AXES_TICKS)
    

    major_y_locator = MultipleLocator(1000)
    ax1.yaxis.set_major_locator(major_y_locator)
    minor_y_locator = MultipleLocator(500)
    ax1.yaxis.set_minor_locator(minor_y_locator)
    ax1.set_xlabel("Day", fontsize=FONTSIZE_AXES_LABELS)
    ax1.set_ylabel("Summed electron fluxes", fontsize=FONTSIZE_AXES_LABELS)
    
    ax1.tick_params(which='major', axis='y', length=10, labelsize=FONTSIZE_AXES_TICKS)  
    ax1.tick_params(which='minor', axis='y', length=6) 
    ax1.tick_params(which='major', axis='x', length=10, labelsize=FONTSIZE_AXES_TICKS)  
    ax1.tick_params(which='minor', axis='x', length=6) 

    ax1.legend()

    plt.show()



def calculate_scaling_factor():
    may_rayleighs=50
    march_rayleighs=100

    # May event
    obs_start =  obs_times[4] - pd.Timedelta(hours=1)
    obs_end = obs_times[4] + pd.Timedelta(hours=1)
    filename = 'maven_f_flux_hr_aurora_attempt_5'
    may_df =  read_maven_sep_flux_data(filename) 
    numeric_cols = may_df.select_dtypes(include='number').columns # only numeric columns
    may_df[numeric_cols] = may_df[numeric_cols].mask(may_df[numeric_cols] > 999999)
    indices= may_df[(may_df['datetime']>=obs_start) & (may_df['datetime'] <= obs_end)].index
    closest_obs_sample_index = indices[-1]
    #index_range = [closest_obs_sample_index -24*5, closest_obs_sample_index +24*5+1]
    sliced_df = may_df.iloc[indices].reset_index()
    df_electron_flux = sliced_df.iloc[:,32:-2]
    df_ion_flux = sliced_df.iloc[:,4:32]
    df_datetime = sliced_df["datetime"]
    df_electron_may = pd.concat([df_datetime, df_electron_flux], axis=1)
    df_ion_may = pd.concat([df_datetime, df_ion_flux], axis=1)

    df_electron_may.set_index('datetime', inplace=True)
    df_electron_may = df_electron_may.iloc[:, ::-1]
        
    df_ion_may.set_index('datetime', inplace=True)
    df_ion_may = df_ion_may.iloc[:, ::-1]    


    df_electron_may["sum"] = df_electron_may.sum(axis=1)
    df_electron_may.reset_index(inplace=True)
    print("May 18th observation")
    print(df_electron_may[["datetime", "sum"]])

    print(f'Observation time: {obs_times[4]}')

    diff = df_electron_may["sum"].iloc[1]-df_electron_may["sum"].iloc[0]
    diff_per_minute = diff/60 
    interpolated_sum_value_may = df_electron_may["sum"].iloc[0]+diff_per_minute*33
    print(f'Interpolated value of sum at 00:33: {interpolated_sum_value_may}')

    may_scaling_factor = may_rayleighs/interpolated_sum_value_may
    print(f'Scaling factor May: {may_scaling_factor}')


    # 4th attempt, aurora march observation

    obs_start =  obs_times[3] - pd.Timedelta(hours=1)
    obs_end = obs_times[3] + pd.Timedelta(hours=1)
    filename = 'maven_f_flux_hr_aurora_attempt_4'
    march_df =  read_maven_sep_flux_data(filename) 
    numeric_cols = march_df.select_dtypes(include='number').columns # only numeric columns
    march_df[numeric_cols] = march_df[numeric_cols].mask(march_df[numeric_cols] > 999999)
    indices= march_df[(march_df['datetime']>=obs_start) & (march_df['datetime'] <= obs_end)].index

    sliced_df = march_df.iloc[indices].reset_index()
    df_electron_flux = sliced_df.iloc[:,32:-2]
    df_ion_flux = sliced_df.iloc[:,4:32]
    df_datetime = sliced_df["datetime"]
    df_electron_march = pd.concat([df_datetime, df_electron_flux], axis=1)
    df_ion_march = pd.concat([df_datetime, df_ion_flux], axis=1)

    df_electron_march.set_index('datetime', inplace=True)
    df_electron_march = df_electron_march.iloc[:, ::-1]
        
    df_ion_march.set_index('datetime', inplace=True)
    df_ion_march = df_ion_march.iloc[:, ::-1]

    df_electron_march["sum"] = df_electron_march.sum(axis=1)
    df_electron_march.reset_index(inplace=True)

    print("March 18th observation")
    print(df_electron_march[["datetime", "sum"]])
    print(f'Observation time: {obs_times[3]}')

    diff = df_electron_march["sum"].iloc[1]-df_electron_march["sum"].iloc[0]
    diff_per_minute = diff/60 
    interpolated_sum_value_march = df_electron_march["sum"].iloc[0]+diff_per_minute*45

    print(f'Interpolated value of sum at 06:45: {interpolated_sum_value_march}')

    march_scaling_factor = march_rayleighs/interpolated_sum_value_march
    print(f'Scaling factor March: {march_scaling_factor}')


    print("-------------------------")
    prediction_may_rayleighs =  march_scaling_factor*interpolated_sum_value_may
    print(f'Prediction for May Rayleighs: {prediction_may_rayleighs}')

if __name__ == "__main__":
    plot_integrated_electron_channels()
    calculate_scaling_factor()