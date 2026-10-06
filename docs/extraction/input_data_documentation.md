# Input Data to the Extraction Pipeline

This section describes the datasets used as input to the extraction pipeline. Data is organized by **thematic category**, which also corresponds to the directory in which the data is downloaded and stored (`*_DATA_PATH`). For each dataset, the resource name and a brief description of its content and intended use in the pipeline are provided.

> **Note on naming convention**: some datasets include the `_{YEAR}` suffix (e.g. `geo_nodes_{YEAR}`), indicating that the resource is versioned by year and is downloaded/updated periodically. The reference year is set via the `YEAR` variable in the pipeline configuration.

## 1. Geographic data — `GEO_DATA_PATH`

Datasets related to the geographic and infrastructural representation of the territory.

| Dataset name | Description |
|---|---|
| `zoi_manual_border` | Geographic boundary of the ZOI (manually defined perimeter) |
| `geo_nodes_{YEAR}` | Nodes of the OSM road network for the reference year |
| `geo_edges_{YEAR}` | Edges of the OSM road network for the reference year |
| `ltz_boundary_{YEAR}` | Boundaries of the Limited Traffic Zone (LTZ) in the city center of Bologna for the reference year |
| `comuni_01012023` | Administrative municipal boundaries in the Bologna province, updated as of January 1, 2023 |
| `poi_bologna_{YEAR}` | OSM points of interest (POI) in Bologna  |
| `buildings_bologna_{YEAR}` | OSM building footprints dataset in Bologna |

## 2. Origin-destination matrices — `OD_DATA_PATH`

Datasets related to the Origin-Destination (OD) matrix of people flows in the province of Bologna. It is present in the datalake as a unique item named `pums_od`.

| Dataset name | Description |
|---|---|
| `PROGETTO-OD.xslx` | OD matrix derived from the Sustainable Urban Mobility Plan (PUMS) |
| `Shape_zone.SHP` | Shapefile of the PUMS zones, represented as polygons with projected CRS 23032. It does not include external zones |
| `Shape_zone_centroid.SHP` | Shapefile of the PUMS zones, represented as centroids with projected CRS 23032 |

## 3. Traffic sensor data — `SENSOR_DATA_PATH`

Data collected from traffic monitoring sensors (induction loops, gates).

| Dataset name | Description |
|---|---|
| `spira_flow_data_2024` | Vehicle flow data recorded by induction loop sensors, 2024 |
| `spire_flow5m_2024` | Vehicle flow data aggregated at 5-minute intervals, 2024 |
| `spira_accur_data_2024` | Accuracy/quality data for induction loop sensor measurements, 2024 |
| `gates` | Registry of traffic detection gates, with ids and locations |
| `gate_data` | Transit data recorded at the gates |

## 4. Public transport — `PT_DATA_PATH`

Datasets related to the public transport (PT) service in the city and in the province of Bologna.

> **Note:** only the public road trasport is included.

| Dataset name | Description |
|---|---|
| `tper_vigente` | Current PT network and services from the public transport operator (PO) of Bologna, TPER |
| `tper_fermate_autobus` | Registry of bus stops from TPER |
| `zona500_geocoded_localities_costs` | Geocoded localities in the Bologna province with associated costs/fares to reach the city center with PT |
| `gommagtfsbo` | GTFS data for road-based public transport in the Bologna area |

## 5. Vehicle fleet data — `VEHICLE_DATA_PATH`

| Dataset name | Description |
|---|---|
| `car_segment_mass` | Average vehicle mass by segment/category. |
| `autovetture_circolanti_per_alimentazione` | Number of registered cars classified by fuel type |
| `autovetture_e_motocicli_per_normativa_ecologica_euro` | Registered cars and motorcycles classified by Euro emission standard |
| `parco_circolante_EmiliaRomagna` | Vehicle fleet in circulation in Emilia-Romagna |
| `parco_circolante_EmiliaRomagna_actual_metadata` | Updated metadata for the Emilia-Romagna vehicle fleet dataset |

## 6. Emission factors — `EMISSION_DATA_PATH`

| Dataset name | Description |
|---|---|
| `emission_factors_2024` | Emission factors by vehicle/fuel type, 2024 |
| `fuel_combustion` | Data related to fuel combustion |

## 7. Social vulnerability indicators — `FRAGILITIES_DATA_PATH`

Socio-demographic datasets used to assess social vulnerability of the population. Each one is
read by a single ethics extraction step (see `pipeline_description.md`).

| Dataset name | Description | Read by |
|---|---|---|
| `indici_di_fragilita_dal_2021` | Social vulnerability indices, time series from 2021 onward | `step_03` |
| `sportelli_sociali_utenti_annuali` | Annual number of users of social service desks in Bologna | `step_05` |
| `aree_statistiche` | Statistical areas of Bologna | `step_02`, `step_05`, `step_06`, `step_08` |
| `popolazione_residente` | Resident population data. | `step_04`, `step_05` |
| `famiglie_residenti_con_almeno_un_componente_minore_di_14_anni` | Resident households with at least one member under 14 years old in Bologna | `step_06` |
| `famiglie_residenti_con_almeno_un_componente_minore_di_18_anni` | Resident households with at least one member under 18 years old in Bologna | `step_05`, `step_06`, `step_08` |
| `famiglie_residenti` | Total number of resident households in Bologna | `step_06` |
| `famiglie-residenti-con-stranieri-per-cittadinanza-tipologia-numero-di-componenti` | Resident households with at least one foreign member, by citizenship of the head of household, household type and number of members | `step_08` |
| `redditi_mediani_aree` | Median and mean declared IRPEF income per statistical area, by citizenship, sex and age band, with contributor counts. **Derived**, not a raw open-data source — see below | `step_07` |

### 7.1 Citizenship-aware sources

The last two datasets carry a citizenship dimension. No indicator reads them yet.

- `famiglie-residenti-con-stranieri-...` distinguishes the citizenship of the head of
  household and of the minors. It only covers households with at least one foreign member;
  households with none are reconstructed in `step_08` from
  `famiglie_residenti_con_almeno_un_componente_minore_di_18_anni`, where the citizenship of
  the head of household is unknown.
- `redditi_mediani_aree` distinguishes `Italiani` / `Stranieri` / `__TOTALE__`.

### 7.2 `redditi_mediani_aree` — a preparation-derived dataset

`redditi_mediani_aree` is produced by the preparation pipeline, logged as a dataitem and
downloaded back by `step_00` like any other input. `modules/ethics/preparation/step_01_redditi_mediani`
builds it from three dataitems extracted from the Tableau workbook published by
[inumeridibolognametropolitana.it](https://inumeridibolognametropolitana.it/studi-e-ricerche/i-redditi-2023-dichiarati-dalle-cittadine-e-dai-cittadini-di-bologna):
`redditi_area`, `redditi_zona` and `redditi_quartiere`, each aggregated at its own
territorial level and carrying `COD_AREAST` / `CODZONA` / `NUMQUART`, which join directly
to `aree_statistiche`.

Each holds median and mean declared IRPEF income plus the contributor count, by citizenship,
sex and age band, for 2002-2023. Cells below `MIN_SAMPLE` (150 contributors) are dropped: the
publisher withholds them, and a median over fewer contributors is neither disclosable nor
meaningful.

The step builds the full `area x year x (citizenship, sex, age band)` grid and fills missing
cells by widening the geography first (area -> zona -> quartiere), then loosening the
demographic category. Every row records where its value came from in `fonte_reddito_mediano`
and `livello_fallback_categoria`, and carries `contribuenti` so a consumer can weight or
exclude substituted values.

Median income is an individual quantity; `rmpe_fam` in `indici_di_fragilita_dal_2021` is a
family-equivalent income. The economic MPI uses `rmpe_fam`.

## 8. OTP routing results — `OTP_DATA_PATH`

Input/output data related to multimodal routing computed via OpenTripPlanner (OTP).

| Dataset name | Description |
|---|---|
| `od-coords-extended` | Origin-destination coordinates for the "extended" scenario, i.e.: origins and destinations in the centroids of PUMS zones |
| `otp_results_extended_allBologna_CAR_PARK_TRANSIT` | OTP results for the "extended" scenario, full Bologna area, allowing multimodal trips with private vehicles (car) and public transport (transit) |
| `otp_results_extended_restrictedZoi_CAR_PARK_TRANSIT` | OTP results for the "extended" scenario, simulating a restricted ZOI, allowing multimodal trips with private vehicles (car) and public transport (transit) |
| `od-coords-zoi` | Origin-destination coordinates for the "zoi" scenario, i.e.: origins and destinations in the centroids of PUMS zones or of the ZOI |
| `otp_results_zoi_allBologna_CAR_PARK_TRANSIT` | OTP results for the "zoi" scenario, full Bologna area, allowing multimodal trips with private vehicles (car) and public transport (transit) |
| `otp_results_zoi_allBologna_TRANSIT` | OTP results for the "zoi" scenario, full Bologna area, allowing trips with only public transport (transit) |
