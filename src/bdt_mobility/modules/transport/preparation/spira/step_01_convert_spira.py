import pandas as pd
import digitalhub as dh
from bdt_mobility.utils.config import DATA_PATH
from bdt_mobility.preparation.config import PROJECT_NAME

SPIRA_PATH = DATA_PATH / 'preparation' / 'spira'
SPIRA_PATH.mkdir(parents=True, exist_ok=True)

SPIRA_FLOW_PATH = SPIRA_PATH / 'flow'
SPIRA_FLOW_PATH.mkdir(parents=True, exist_ok=True)

SPIRA_ACCURACY_PATH = SPIRA_PATH / 'accuracy'
SPIRA_ACCURACY_PATH.mkdir(parents=True, exist_ok=True)


def convert_spira(project, spira_di, name):
    # read and rewrite to normalize and export as data
    spira_csv = spira_di.as_df()
    spira_csv = spira_csv.rename(columns={'codice_spira': 'spira_code', 'chiave':'spira_unique_id', 'angolo': 'angle'})
    
    spira_melted = spira_csv.melt(id_vars=['data', 'spira_unique_id', 'spira_code', 'angle', 'longitudine', 'latitudine'],
                       value_vars=[col for col in spira_csv.columns if col.startswith(('00_', '01_', '02_', '03_', '04_', '05_', '06_',
                                                                               '07_', '08_', '09_', '10_', '11_', '12_', '13_',
                                                                               '14_', '15_', '16_', '17_', '18_', '19_', '20_',
                                                                               '21_', '22_', '23_'))],
                       var_name='hour',
                       value_name='count')
    
    spira_melted['DateTime'] = pd.to_datetime(spira_melted['data'].astype(str) + ' ' + spira_melted['hour'].str.replace('_', ':').str[:5] + ':00')
    spira_melted['DateTime'] = spira_melted['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')

          
    spira_melted.drop(columns=['data', 'hour'], inplace=True)

    spira_melted.to_parquet(SPIRA_FLOW_PATH / (name + '.parquet') )
    # project.log_dataitem(name=name, kind="table", data=spira_melted)

def convert_spira_accur(project, spira_accur_di, name):
    accuracy_csv = spira_accur_di.as_df()
    accuracy_csv=accuracy_csv.rename(columns={'codice_spira_2': 'spira_code', 'data_2': 'data'})

    acc_melted = accuracy_csv.melt(id_vars=['data', 'spira_code'],
                   value_vars=[col for col in accuracy_csv.columns if col.startswith(('00_', '01_', '02_', '03_', '04_', '05_', '06_',
                                                                           '07_', '08_', '09_', '10_', '11_', '12_', '13_',
                                                                           '14_', '15_', '16_', '17_', '18_', '19_', '20_',
                                                                           '21_', '22_', '23_'))],
                   var_name='hour',
                   value_name='count')

    acc_melted['DateTime'] = pd.to_datetime(acc_melted['data'] + ' ' + acc_melted['hour'].str.replace('_', ':').str[:5] + ':00')
    acc_melted.drop(columns=['data', 'hour'], inplace=True)
    acc_melted['count'] = acc_melted['count'].str.replace('%', '').astype(int)
    
    acc_melted['DateTime'] = acc_melted['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    acc_melted.to_parquet(SPIRA_ACCURACY_PATH / (name + '.parquet') )
    # project.log_dataitem(name=name, kind="table", data=acc_melted)


def main():
    project = dh.get_or_create_project(PROJECT_NAME)

    for year in ["2023", "2024", "2025", "2026"]:
        
        di = project.get_dataitem(f"spire_flow1h_{year}_source")
        convert_spira(project=project, spira_di=di, name=f"spira_flow_data_{year}")

        di_acc = project.get_dataitem(f"spire_accur_flow1h_{year}_source")
        convert_spira_accur(project=project, spira_accur_di=di_acc, name=f"spira_accur_data_{year}")

if __name__ == '__main__':
    main()

# uv run python -m bdt_mobility.modules.transport.preparation.spira.step_01_convert_spira
