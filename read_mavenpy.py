from mavenpy import load, retrieve
from parameters import DATA_DIR
import cdflib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
level = ['l1','l2']


sensors = ['s1', 's2']

types = ['raw', 'cal']
particles = ['ion', 'elec']

look_dirs = ['f', 'r']


f_ion_flux_column_labels = ['20.1-21.5 keV/n,Ion', '21.6-23.0 keV/n,Ion',
       '23.0-24.4 keV/n,Ion', '24.4-27.2 keV/n,Ion', '27.2-30.0 keV/n,Ion',
       '30.1-34.3 keV/n,Ion', '34.3-39.9 keV/n,Ion', '40.0-47.0 keV/n,Ion',
       '47.0-57.0 keV/n,Ion', '57.0-69.8 keV/n,Ion', '69.7-86.7 keV/n,Ion',
       '86.8-109.4 keV/n,Ion', '109.1-138.9 keV/n,Ion',
       '138.8-177.2 keV/n,Ion', '177.5-228.5 keV/n,Ion',
       '228.7-295.3 keV/n,Ion', '295.4-384.6 keV/n,Ion',
       '385.0-501.0 keV/n,Ion', '500.5-653.5 keV/n,Ion',
       '653.5-856.5 keV/n,Ion', '856.5-1123.5 keV/n,Ion',
       '1124.0-1476.0 keV/n,Ion', '1478.0-1942.0 keV/n,Ion',
       '1935.0-2545.0 keV/n,Ion', '2549.0-3351.0 keV/n,Ion',
       '3350.0-4410.0 keV/n,Ion', '4405.0-5815.0 keV/n,Ion',
       '5820.0-7560.0 keV/n,Ion']

f_elec_flux_column_labels = ['20.3-21.7 keV,e', '21.8-23.2 keV,e', 
                             '23.2-24.6 keV,e', '24.7-27.5 keV,e', 
                             '27.6-30.4 keV,e', '30.4-34.8 keV,e', 
                             '34.8-40.6 keV,e', '40.6-47.8 keV,e', 
                             '47.9-57.9 keV,e', '57.9-70.9 keV,e', 
                             '70.9-88.3 keV,e', '88.4-111.5 keV,e', 
                             '111.8-142.2 keV,e', '141.4-180.6 keV,e',
                             '180.9-233.1 keV,e']


def retrieve_data():

    instrument = 'sep'
    start_date = '2024-03-19'
    end_date = '2024-03-19'
    level='l2'
    resolution = '30sec'

    dataset_name = 's(.*)-svy-full'
    retrieve.sdc_retrieve(
        instrument=instrument, destination_dir=DATA_DIR,
        ext='cdf', res=resolution, level=level, coord='pl',
        dataset_name = dataset_name,
        start_date=start_date, end_date=end_date)



def explore_data():
    instrument = 'sep'
    start_date = '2024-03-19'
    end_date = '2024-03-19'
    level='l2'
    resolution = '30sec'
    cdf_file_1 = cdflib.CDF(DATA_DIR / "maven/data/sci/sep/l2/2015/03/mvn_sep_l2_s1-cal-svy-full_20150301_v04_r03.cdf")
    timestamps = cdf_file_1.varget('epoch')

    info = cdf_file_1.cdf_info()
    print(info.zVariables)
    attrs = cdf_file_1.varattsget('f_ion_flux_tot')
    print(attrs)
    #print(attrs['DEPEND_1'])      



    attrs = cdf_file_1.varattsget('f_ion_flux')
    print(attrs['DEPEND_1'])        # should print 'f_ion_energy'

    energies = cdf_file_1.varget('f_ion_energy')
    denergies = cdf_file_1.varget('f_ion_denergy')
    print(energies.shape, energies)
    print(cdf_file_1.varattsget('f_ion_energy')['UNITS'])
    print(denergies)
    
    #denergies = cdf_file_1.varget('f_ion_')
    #print(energies.shape, energies)
    #print(cdf_file_1.varattsget('f_ion_energy')['UNITS'])

    #fluxes = cdf_file_1.varget('f_ion_flux')
    #attrs = cdf_file_1.varattsget('f_ion_flux')
    #print(attrs)    
    #energy_var = attrs['DEPEND_1']
    #print(energy_var)
    #print(energy_var)

    # variables = cdf_file_1.cdf_info().zVariables
    #print("Variables:", variables)
    #print(fluxes)
    #print(np.shape(fluxes))
    #print(np.shape(timestamps))
    #print(type(fluxes))
    #ar_attrs = cdf_file_1.varattsget('f_ion_flux')
    #print(ar_attrs)


def load_maven_f_particle_flux(start_date, end_date, sensor):

    instrument = 'sep'
    level = 'l2'
    dataset_name = f'{sensor}-svy-full'
    processed = 'cal'
    start_year = start_date[0:4]
    start_month = start_date[5:7]
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    delta = end_date - start_date 

    df_ion_list = []
    df_electron_list = []
    for i in range(delta.days + 1):
        day = start_date + timedelta(days=i)
        date = str(day)[:10].replace("-","")
        filename = f'maven/data/sci/{instrument}/{level}/{start_year}/{start_month}/' \
        f'mvn_{instrument}_{level}_{sensor}-{processed}-svy-full_{date}_v04_r03.cdf'

        cdf_file = cdflib.CDF(DATA_DIR / filename)
        # variables = cdf_file.cdf_info().zVariables

        timestamps = cdf_file.varget('epoch')
        current_ion_flux = cdf_file.varget('f_ion_flux')
        current_electron_flux = cdf_file.varget('f_elec_flux')


        df_f_ion_flux = pd.DataFrame(current_ion_flux, columns=f_ion_flux_column_labels)
        df_f_ion_flux['epoch'] = timestamps
        df_f_ion_flux['datetime'] = pd.to_datetime(cdflib.cdfepoch.to_datetime(df_f_ion_flux['epoch'].to_numpy()))
        df_f_ion_flux.drop(['epoch'], axis=1, inplace=True)
        df_ion_list.append(df_f_ion_flux)


        df_f_electron_flux = pd.DataFrame(current_electron_flux, columns=f_elec_flux_column_labels)
        df_f_electron_flux['epoch'] = timestamps
        df_f_electron_flux['datetime'] = pd.to_datetime(cdflib.cdfepoch.to_datetime(df_f_electron_flux['epoch'].to_numpy()))
        df_f_electron_flux.drop(['epoch'], axis=1, inplace=True)
        df_electron_list.append(df_f_electron_flux)


        # energies = cdf_file.varget('f_elec_energy')
        # denergies = cdf_file.varget('f_elec_denergy')
        # print(energies.shape, energies)
        # print(denergies)

    combined_ion = pd.concat(df_ion_list, ignore_index=True)
    combined_electron = pd.concat(df_electron_list, ignore_index=True)
    return combined_ion, combined_electron
    
def load_maven_f_electron_flux():
    print("yay")

def load_maven_r_ion_flux():
    print("yay")

def load_maven_data():
    instrument = 'sep'
    level = 'l2'
    sensor = 's1'
    dataset_name = f'{sensor}-svy-full'
    variable = 'f_ion_flux'
    processed = 'cal'
    start_date = '2015-03-01'
    end_date = '2015-03-01'
    start_year = start_date[0:4]
    start_month = start_date[5:7]
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")


    delta = end_date - start_date 

    for i in range(delta.days + 1):
        day = start_date + timedelta(days=i)
        date = str(day)[:11].replace("-","")
        filename = f'maven/data/sci/{instrument}/{level}/{start_year}/{start_month}/' \
        f'mvn_{instrument}_{level}_{sensor}-{processed}-svy-full_{date}_v04_r03.cdf'

        cdf_file = cdflib.CDF(DATA_DIR / filename)
        variables = cdf_file.cdf_info().zVariables
        print("Variables:", variables)


        timestamps = cdf_file.varget('epoch')
        fluxes = cdf_file.varget(variable)
        df_f_ion_flux = pd.DataFrame(fluxes, columns=ion_energy_channel_names)
        df_f_ion_flux['epoch'] = timestamps
        df_f_ion_flux['datetime'] = pd.to_datetime(cdflib.cdfepoch.to_datetime(df_f_ion_flux['epoch'].to_numpy()))
    
    
    df_f_ion_flux.drop(['epoch'], axis=1)
    """
    cdf_file_1 = cdflib.CDF(DATA_DIR / "maven/data/sci/sep/l2/2015/03/mvn_sep_l2_s1-cal-svy-full_20150301_v04_r03.cdf")
    timestamps = cdf_file_1.varget('epoch')
    print(timestamps)

    fluxes = cdf_file_1.varget('f_ion_flux')

    variables = cdf_file_1.cdf_info().zVariables
    print("Variables:", variables)
    print(fluxes)
    print(np.shape(fluxes))
    print(np.shape(timestamps))
    print(type(fluxes))
    ar_attrs = cdf_file_1.varattsget('epoch')
    print(ar_attrs)

    df_f_ion_flux = pd.DataFrame(fluxes, columns=ion_energy_channel_names)
    df_f_ion_flux['epoch'] = timestamps
    df_f_ion_flux['datetime'] = pd.to_datetime(cdflib.cdfepoch.to_datetime(df_f_ion_flux['epoch'].to_numpy()))
    df_f_ion_flux.drop(['epoch'], axis=1)
    return df_f_ion_flux
    """
        #start_year = start_date[0:4]
    #start_month = start_date[5:7]
    #start_day = start_date[-2:]


    #end_year = end_date[0:4]
    #end_month = end_date[5:7]
    #end_day = end_date[-2:]
if __name__ == "__main__":
    # retrieve_data()
    # load_maven_data()
    # explore_data()
    
    a, b = load_maven_f_particle_flux(start_date='2024-03-17', end_date='2024-03-18', 
                                      sensor='s1')
