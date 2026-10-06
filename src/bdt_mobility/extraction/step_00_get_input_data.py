from pathlib import Path
import json
import digitalhub as dh

from bdt_mobility.extraction.config import PROJECT_NAME
from bdt_mobility.utils.paths import GEO_DATA_PATH, OD_DATA_PATH, SENSOR_DATA_PATH, PT_DATA_PATH, VEHICLE_DATA_PATH, EMISSION_DATA_PATH, FRAGILITIES_DATA_PATH, OTP_DATA_PATH, MUNICIPALITY_DATA_PATH, GTFS_DATA_PATH
from bdt_mobility.utils.paths import DATA_PATH, ensure_dirs
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed

import digitalhub as dh  # type: ignore[import-untyped]
from digitalhub.utils.exceptions import BackendError  # type: ignore[import-untyped]


from bdt_mobility.preparation.config import YEAR

MAIN_OUTPUTS = []

dataitems = {
    GEO_DATA_PATH: [
        "zoi_manual_border",
        f"geo_nodes_{YEAR}", 
        f"geo_edges_{YEAR}", 
        f"ltz_boundary_{YEAR}",
        "comuni_01012023",
        f"poi_bologna_{YEAR}",
        f"buildings_bologna_{YEAR}",
        ],
    OD_DATA_PATH: [
        "pums_od"
    ],
    SENSOR_DATA_PATH: [
        "spira_flow_data_2024",
        "spire_flow5m_2024",
        "spira_accur_data_2024",
        "gates",
        "gate_data"
    ],
    PT_DATA_PATH: [
        "tper_vigente",
        "tper_fermate_autobus",
        "zona500_geocoded_localities_costs",
    ],
    GTFS_DATA_PATH: [
        "gommagtfsbo"
    ],
    MUNICIPALITY_DATA_PATH: [
        "municipality_taxonomy_map"
    ],
    VEHICLE_DATA_PATH: [
        "car_segment_mass",
        "autovetture_circolanti_per_alimentazione",
        "autovetture_e_motocicli_per_normativa_ecologica_euro",
        "parco_circolante_EmiliaRomagna",
        "parco_circolante_EmiliaRomagna_actual_metadata"
    ],
    EMISSION_DATA_PATH: [
        "emission_factors_2024",
        "fuel_combustion"
    ],
    FRAGILITIES_DATA_PATH: [
        "indici_di_fragilita_dal_2021",
        "sportelli_sociali_utenti_annuali",
        "aree_statistiche",
        "popolazione_residente",
        "famiglie_residenti_con_almeno_un_componente_minore_di_14_anni",
        "famiglie_residenti_con_almeno_un_componente_minore_di_18_anni",
        "famiglie_residenti",
        "famiglie-residenti-con-stranieri-per-cittadinanza-tipologia-numero-di-componenti",
        "redditi_mediani_aree"
    ],
    OTP_DATA_PATH: [
        "od-coords-extended",
        "otp_results_extended_allBologna_CAR_PARK_TRANSIT",
        "otp_results_extended_restrictedZoi_CAR_PARK_TRANSIT",
        "od-coords-zoi",
        "otp_results_zoi_allBologna_CAR_PARK_TRANSIT",
        "otp_results_zoi_allBologna_TRANSIT",
    ],
}


def download_dataitem(name, path):
    dataitem = dh.get_dataitem(name, PROJECT_NAME)
    files = dataitem.files
    if isinstance(files, dict):
        files = files["content"]
    dataitem_path = dataitem.spec.path
    if dataitem_path.endswith("/"):  # Download multiple files in a directory with the name as `name`
        new_filename = dataitem.download(path, overwrite=True)
    else: # Download file and rename it as `name`
        filename = dataitem.download(path, overwrite=True)
        filename = Path(filename)
        if filename.stem != name:
            new_filename = filename.with_name(name).with_suffix(filename.suffix)
            filename.rename(new_filename)
    return {'key': dataitem.key, 'spec': dataitem_path}


def main():
    # Logging setting
    logging.basicConfig(level=logging_level)
    with timed():
        # Create directories
        DATA_PATH.mkdir(parents=True, exist_ok=True)
        ensure_dirs()
        # Create data list dictionary
        data_list = {}
        # Download base dataitems
        missing = []
        for path in dataitems:
            for dataitem in dataitems[path]:
                logging.info(f"Downloading dataitem {dataitem}")
                try:
                    data_list[dataitem] = download_dataitem(dataitem, path)
                except BackendError as e:
                    logging.warning(f"Skipping {dataitem!r}: {e}")
                    missing.append(dataitem)
        if missing:
            logging.warning(f"The following dataitems were not found on the server: {missing}")

        logging.info(f"Writing the data list")
        with open(DATA_PATH / "input_data_list.txt", "w") as f:
            f.write(json.dumps(data_list, indent=4))


if __name__ == "__main__":
    main()
