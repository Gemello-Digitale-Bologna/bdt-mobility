import pandas as pd

from bdt_mobility.utils.config import DATA_PATH
from bdt_mobility.extraction.platform.outputs import PlatformOutput
from pathlib import Path
from .s3_download import s3_download_file



EXPORT_FILES = [
    'Export_ZOI_20250701-20260101.xls',
    'Export_11_02_2025, 20_04_24 - 20_11_24.xlsx',
    'Export_11_02_2025, 20_11_24 - 11_02_25.xlsx',
    'Export_ZOI_20250101-20250701.xls',
    'Export_30_09_2024, 15_49_11.xlsx',
    'Export_11_02_2025, 20_11_23 - 20_05_24.xlsx',
    'Export ZOI from 09_10_2023 to 09_10_2024.xlsx',
    'Export_ZOI_2024.xls'
    ]

GATE_PATH = DATA_PATH / 'preparation' / 'gate'
GATE_PATH.mkdir(parents=True, exist_ok=True)

EXPORT_EXCEL_FOLDER = GATE_PATH / 'gate_excel'
EXPORT_EXCEL_FOLDER.mkdir(parents=True, exist_ok=True)

EXPORT_PARQUET_FOLDER = GATE_PATH / 'gate_parquet'
EXPORT_PARQUET_FOLDER.mkdir(parents=True, exist_ok=True)

ZOI_GATE_FILE = 'ZOI.xlsx'
ZOI_GATE_FILE_FOLDER = GATE_PATH / 'gate_data'
ZOI_GATE_FILE_FOLDER.mkdir(parents=True, exist_ok=True)


def _convert(export):
    converted = []

    ir = export.iterrows()
    
    # Parse general header
    i,r = next(ir)
    assert r["Da"] == "Da"
    
    i,r = next(ir)
    while True:
        # Parse data row
        assert r["Da"] == r["Da"] 
        data = r["Da"]
        # Parse varco header
        i,r = next(ir)
        assert r["Da"] != r["Da"]
        assert r["A"] == "Varco" 
        # Parse varco section
        i,r = next(ir)
        while True:
            # Parse varco row
            if r["Da"] == r["Da"]: break
            assert r["A"] == r["A"]
            varco = r["A"]
            # Parse veicolo header
            i,r = next(ir)
            assert r["Da"] != r["Da"]
            assert r["A"] != r["A"]
            if (r["Dettaglio totali"] == "Motivo spostamento"):
                # Parse motivo spostamento
                i,r = next(ir)
                assert r["Dettaglio totali"] == "Transito Incompleto"
                i,r = next(ir)
            assert r["Dettaglio totali"] == "Classe veicoli"
            while True:
                # Parse veicolo row
                i,r = next(ir)
                if (r["Da"] == r["Da"]) or (r["A"] == r["A"]) or (r["Dettaglio totali"] != r["Dettaglio totali"]):
                    break
                assert r["Dettaglio totali"] != "Classe veicoli"
                veicolo = r["Dettaglio totali"]
                cont = r["Dettaglio totali.1"]
                converted.append({"Data": data, "Varco": varco, "Veicolo": veicolo, "Conteggio": cont})
            if r["Dettaglio totali"] != r["Dettaglio totali"]: break
        # Parse data trailer
        while True:
            try:
                i,r = next(ir)
                if (r["Da"] == r["Da"]) or (r["A"] == r["A"]): break
            except StopIteration:
                return pd.DataFrame(converted)

def read_gates(fname):
    gates = pd.read_excel(fname)
    gates = gates.iloc[:gates["Varco"].isnull().idxmax()]
    gates[["ID","Varco"]] = gates["Varco"].str.split('\\) ', n=1, expand=True)
    # Move ID column to the front
    gates = gates[gates.columns.tolist()[-1:] + gates.columns.tolist()[:-1]]
    gates.loc[gates["Varco"] == "Corelli Nord", "Varco"] = "Corelli_1 Nord"
    gates.loc[gates["Varco"] == "Corelli Sud", "Varco"] = "Corelli_2 Sud"
    gates.loc[gates["Varco"] == "Bonvicini", "Varco"] = "Porrettana/Bonvicini"
    gates["ID"] = gates["ID"].astype("string")
    gates["Varco"] = gates["Varco"].astype("string")
    gates["Indirizzo"] = gates["Indirizzo"].astype("string")
    gates["Settore"] = gates["Settore"].astype("string")
    gates["Link google maps"] = gates["Link google maps"].astype("string")
    gates = gates.rename(columns={'Varco':'gate'})
    return gates

def check(df):
    unique = df.groupby(['Data', 'gate', 'vehicle','count']).nunique().reset_index()
    multiple = unique.loc[unique.duplicated(keep=False, subset=['Data', 'gate', 'vehicle'])]
    if multiple.size:
        raise Exception(f"Multiple rows with different values ({multiple.shape[0]} rows)")
    if df.size != unique.size:
        raise Exception(f"Multiple rows with same values ({df.shape[0] - unique.shape[0]} rows)")


def download_gates():
    for file_name in EXPORT_FILES:
        s3_download_file(
            bucket_name='protected',
            file_path=f'/bdt_mobility/{file_name}',
            output_path=EXPORT_EXCEL_FOLDER / file_name
        )

    s3_download_file(
        bucket_name='protected',
        file_path=f'/bdt_mobility/{ZOI_GATE_FILE}',
        output_path=ZOI_GATE_FILE_FOLDER / ZOI_GATE_FILE
    )


def create_gate_data():

    gates = read_gates(ZOI_GATE_FILE_FOLDER / 'ZOI.xlsx')

    merged = None
    for file in Path(EXPORT_EXCEL_FOLDER).iterdir():
        if str(file.name) == 'Export_30_09_2024, 15_49_11.xlsx':
            continue
        export = pd.read_excel(file)
        converted = _convert(export) 
        converted.loc[converted["Veicolo"] == "Ciclomotore ", "Veicolo"] = "Ciclomotore"
        converted["Data"] = pd.to_datetime(converted["Data"], format="%d/%m/%Y %H:%M").dt.strftime('%Y-%m-%d %H:%M:%S')
        converted["Varco"] = converted["Varco"].astype("string")
        converted["Veicolo"] = converted["Veicolo"].astype("string")
        converted["Conteggio"] = converted["Conteggio"].astype("int")
        converted = converted.rename(columns={'Varco':'gate', 'Veicolo':'vehicle', 'Conteggio': 'count'})

        converted.to_parquet(EXPORT_PARQUET_FOLDER / f'{file.name.split('.')[0]}.parquet')

def main():
    download_gates()
    create_gate_data()

if __name__ == '__main__':
    main()

# uv run python -m bdt_mobility.modules.transport.preparation.gates.step_01_convert_gates