
import polars as pl
import polars.selectors as cs
import json
import numpy as np

from bdt_mobility.utils import data_cleaner
from bdt_mobility.utils import data_reader

from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.extraction.platform.outputs import PlatformOutput

MAIN_OUTPUTS = [
    PlatformOutput("cars_joint_by_comune", PathRegistry.CARS_JOINT_BY_COMUNE, "table"),
    PlatformOutput("poll_ef", PathRegistry.POLL_EF_PARQUET, "table"),
    PlatformOutput("co2_ef", PathRegistry.CO2_EF_PARQUET, "table"),
    PlatformOutput("euro_class_split", PathRegistry.EURO_CLASS_SPLIT_PARQUET, "table"),
]


def _get_expressions(type: str):
    # Implementa le formule matematiche del ef in funzione della velocità e del co2 in funzione del ec
    if type=="ef":
        return (
        (
            pl.col("alpha") * pl.col("speed") ** 2
            + pl.col("beta") * pl.col("speed")
            + pl.col("gamma")
            + pl.col("delta") / pl.col("speed")
        )
        / (
            pl.col("epsilon") * pl.col("speed") ** 2
            + pl.col("zita") * pl.col("speed")
            + pl.col("hta")
        )
        * (1 - pl.col("reduction factor [%]"))
        )
    elif type=="co2":
        return pl.col("ec") / pl.col("fuel_calorific") * pl.col("fuel_co2_combustion")


def get_car_fleet_prov():
    '''
    Returns a pl dataframe with columns:
    - euro_level: int for the euro level type [0--6]
    - fuel: string with the fuel type [b/gpl, g/wank, gasol, ...]
    - segment: string with length of the vehicles categorized in segment levels [small, large, mini, medium]
    - len: counting column, giving how many vehicles have the carachteristics
    '''
    # Load the file for segment classification: 
    # I veicoli sono classificati per segment usando intervalli di massa
    car_segment_mass = (
        pl.read_csv(PathRegistry.CAR_SEGMENT_MASS)
        .with_columns(
            (pl.col("avg_mass") - pl.col("avg_mass") * pl.col("cov_mass")).alias("min_mass")
        )
        .with_columns(pl.col("min_mass").shift(-1).alias("max_mass"))
        .sort("max_mass", nulls_last=True)
    )
    car_segment_mass[0, 'min_mass'] = 0 #smallest class includes all small length down to 0
    car_segment_mass[-1, 'max_mass'] = np.inf #largest class includes all big length up to inf

    # Dati provincia (EmiliaRomagna/Circolante_Emilia.csv): 
    # Contiene la distribuzione congiunta dettagliata [euro_level, fuel, segment] 
    # ma solo aggregata a livello provinciale, non per singolo comune.
    cars_emiro_meta = json.loads(
        (PathRegistry.PARCO_CIRCOLANTE_META).read_text()
    )
    schema = {
        c["name"]: pl.Float64 if c["datatype"] in ("numeric", "number") else pl.Utf8
        for c in cars_emiro_meta["columns"]
    }
    cars_distrib_prov = (
        pl.scan_csv(
            PathRegistry.VEHICLE_DATA / "parco_circolante_EmiliaRomagna.csv",
            schema=schema,
            has_header=False,
        )
        .drop("progressivo", "marca", "data immatricolazione")
        .filter(pl.col("provincia residenza") == "BOLOGNA") # filter for bologna province
        .drop("provincia residenza")
        .filter(pl.col("tipo veicolo") == "A") # filter for car vehicles
        .rename({"massa complessiva": "mass"})
        .join_where(
            car_segment_mass.lazy(),
            pl.col("mass") < pl.col("max_mass"),
            pl.col("mass") >= pl.col("min_mass"),
        ) # add the segment column
        .select(
            pl.col("destinazione").alias("purpose"),
            pl.col("uso").alias("use"),
            pl.col("alimentazione").alias("fuel"),
            pl.col("classe euro").alias("euro_level"),
            pl.col("segment"),
        )
        .group_by(pl.all())
        .agg(pl.len())
        .with_columns(
            pl.when(pl.col("euro_level") == "N")
            .then(None)
            .otherwise(pl.col("euro_level"))
            .cast(int)
            .alias("euro_level")
        )
        .filter(pl.col("purpose") == "AUTOVETTURA PER TRASPORTO DI PERSONE") # filter for private use
        .drop("purpose")
        .with_columns(data_cleaner.str_norm_expr(cs.string()))
        .group_by("euro_level", "fuel", "segment")
        .agg(pl.col("len").sum())
        .collect()
    )

    return cars_distrib_prov


def get_car_fleet_comuni():
    year = 2019
    # Dati comuni (autovetture-e-motocicli-*.parquet): 
    # Contengono distribuzioni marginali [euro_level] e [fuel] disaggregate per ogni comune, 
    # ma non la loro distribuzione congiunta né il segment.
    cars_euro_com = (
        pl.read_parquet(
            PathRegistry.CARS_EURO_PARQUET
        )
        .filter(pl.col("anno").dt.year() == year)
        .filter(pl.col("tipologia") == "AUTOVEICOLI")
        .drop("provincia", "anno", "tipologia")
        .with_columns(data_cleaner.municipality_norm_expr(cs.string()))
        .sort("comune")
    )
    cars_fuel_com = (
        pl.read_parquet(PathRegistry.CARS_FUEL_PARQUET)
        .filter(pl.col("anno").dt.year() == year)
        .drop("provincia", "anno")
        .with_columns(data_cleaner.municipality_norm_expr(cs.string()))
        .sort('comune')
    )

    ## Estimate the distribution of [fuels] at the province level
    # Calcola la distribuzione proporzionale dei fuel a livello provinciale 
    # sommando tutti i comuni e normalizzando. 
    prov_cars_fuel_from_com = cars_fuel_com.select(cs.numeric()).sum().select(
        pl.all() / pl.sum_horizontal(pl.col("*"))
    )

    # Filtering the data
    cars_euro_com = cars_euro_com.drop('nd') # filter: discard some fuel type (remark: nc=electric)
    cars_fuel_com = cars_fuel_com[
        [s.null_count() < cars_fuel_com.height for s in cars_fuel_com]
    ].fill_null(0) # filter: Keep only Comuni with at least 1 non-null value, then fill nas with 0s

    relevant_fuels = np.asarray(prov_cars_fuel_from_com.columns)[(prov_cars_fuel_from_com > 0.0001).to_numpy()[0]]
    cars_fuel_com = cars_fuel_com.select(~cs.numeric(), *relevant_fuels) # filter: remove fuel with extreme low frequency

    return cars_euro_com, cars_fuel_com

def get_efs():
    # Emission factors:
    # Coefficienti che quantificano quanti grammi di inquinante vengono emessi 
    # per ogni chilometro percorso da un veicolo. Dipendono da caratteristiche del veicolo 
    # (fuel, euro level, segment) e dalla velocità
    ef = (
        pl.read_excel(
            PathRegistry.EMISSION_FACTORS_XLSX,
            sheet_name="HOT_EMISSIONS_PARAMETERS",
            drop_empty_cols=True,
            columns=list(range(19)),
            read_options={"dtypes": {7: "float", 8: "float"}}
        )
        # load and slope only present for trucks and buses, so exclude them here.
        .select(pl.all().exclude("Load", "Road Slope").name.to_lowercase())
        .filter(pl.col("category") == "Passenger Cars")
        .drop("category")
        .rename({"euro standard": "euro_level"})
    )

    # EF expression: 
    ef_expr = _get_expressions(type="ef")
    ef_params = [n for n in ef_expr.meta.root_names() if n != 'speed']

    # Filtering and processing of EF data:
    # 1) map conventional to euro 0: euro 0 from BO data is pre 1992: https://www.ayvens.com/it-it/blog/conducenti/classificazione-euro-auto/
    #    ECE is for European norms before 1992, most recent being ECE 15/04 (1985-1992), only applies to petrol though; 
    #    improved conventional and open loop are from some national norms, doesn't apply to IT
    # 2) keep only "euro" columns and cast to int
    # 3) keep only one instance per "fuel", "segment", "euro_level", "pollutant" and parameters
    ef = (
        ef.with_columns(pl.col("euro_level").replace("Conventional", "Euro 0"))
        .filter(pl.col("euro_level").str.starts_with("Euro"))
        .with_columns(
            pl.col("euro_level").str.extract_groups(
                "^Euro (?<euro_level>[0-9]) ?(?<euro_sublevel>.*)$"
            )
        )
        .unnest("euro_level")
        .with_columns(pl.col("euro_level").cast(int))
        # sort and keep first to keep empty string for technology or mode in duped rows.
        .sort("fuel", "segment", "euro_level", 'euro_sublevel', "pollutant", 'mode', 'technology')
        .unique(["fuel", "segment", "euro_level", "pollutant", *ef_params], keep='first', maintain_order=True)
        .with_columns(data_cleaner.str_norm_expr(cs.string())) #fix string formats (underscore for spaces, everything to lowercase)
    )

    ef = ef.filter(pl.col('euro_sublevel').is_in(["", "a/b/c"])).drop('euro_sublevel') #filter: discard d/e/d-temp (see remark below)
    # Remark: Le normative Euro 6 hanno diverse sotto-categorie (a, b, c, d, e) con requisiti progressivamente più stringenti, specialmente per le particelle (PM) e con procedure di test più realistiche da c in poi.
    # Idealmente si dovrebbero usare i fattori d/e (più realistici). Ma: i dati EEA per d/e sono incompleti (solo diesel disponibile)
    # Soluzione:Si tengono solo i veicoli Euro 6 a/b/c, che hanno dati completi per tutti i fuel types, anche se sono meno rappresentativi della realtà (test meno stringenti).

    ef = ef.filter(pl.col('technology').is_in(["", "gdi", "dpf"])) #filter: scarta PFI, LNT, SCR 
    # Remark: I veicoli hanno diverse tecnologie motoristiche che influenzano le emissioni: PFI, GDI, DPF, LNT/SCR.
    # Dopo i filtri precedenti, rimangono duplicati solo per veicoli a benzina (petrol, petrol_hybrid), che possono avere PFI, GDI o GDI+GPF.
    # Soluzione: Si tiene solo GDI perché dati completi, più veicoli nella categoria,; per diesel, si tiene DPF perchè standard.

    return ef


def align_fuel_data_sources(cars_distrib_prov, cars_fuel_com):
    ## Fuel names
    # Dictionaries to link fuel types manes among different data sources
    prov_to_bo_fuels = {
        "benz": "benzina",
        "gasol": "gasolio",
        "b/gpl": "benzina_e_gas_liquido",
        "b/met": "benzina_e_metano",
        "ibrido_benzina/elettrico": "ibrido_benzina",
        "ibrido_gasolio/elettrico": "ibrido_gasolio",
        "elettr": "elettricita",
    }
    com_to_eu_fuels = {
        "benzina": "petrol",
        "gasolio": "diesel",
        "benzina_e_gas_liquido": "lpg_bifuel",
        "benzina_e_metano": "cng_bifuel",
        "metano": "cng_bifuel_~_cng",
        "ibrido_benzina": "petrol_hybrid",
        "ibrido_gasolio": "diesel_phev", # see remark below
        "elettricita": "battery_electric",
    }
    prov_to_eu_fuels = {k: com_to_eu_fuels[v] for k, v in prov_to_bo_fuels.items()}
    # Remark: I veicoli diesel ibridi sono molto pochi in circolazione. La maggior parte degli ibridi diesel sono PHEV (plug-in hybrid electric vehicles).
    # L'EEA probabilmente ha deciso di usare i coefficienti di "diesel_phev" anche per gli ibridi diesel non plug-in, dato che la differenza è minima e il campione sarebbe troppo piccolo

    # Renaming fuels with eu standard
    cars_distrib_prov = cars_distrib_prov.with_columns(
        pl.col("fuel").replace_strict(prov_to_eu_fuels, default=None)
    ).drop_nulls(["fuel", "euro_level", "segment"])

    cars_fuel_com.columns = [com_to_eu_fuels.get(c, c) for c in cars_fuel_com.columns]

    # In the comuni dataset, set to pivot table
    cars_fuel_com = (
        cars_fuel_com.unpivot(
            on=cs.numeric(), index="comune", variable_name="fuel", value_name="count"
        )
        .fill_null(0)
    )
    
    return cars_distrib_prov, cars_fuel_com, 

def  align_euro_level_data_sources(cars_distrib_prov, cars_euro_com):
    # In prov data source, set to -1 the euro_level of electric vehicles.
    cars_distrib_prov = cars_distrib_prov.with_columns(
        pl.when(pl.col("fuel") != "battery_electric")
        .then(pl.col("euro_level"))
        .otherwise(pl.lit(-1))
    )

    # In the comuni datasets, set to -1 the euro_level of electric vehicles,
    # create the euro_level column with integers, and set to pivot table
    cars_euro_com = (
        cars_euro_com.unpivot(
            on=cs.numeric(), index="comune", variable_name="euro_level", value_name="count"
        )
        .with_columns(
            pl.col("euro_level")
            .replace({"nc": "euro_-1"})
            .str.extract("euro_(.+)")
            .cast(int)
        )
        .fill_null(0)
    )

    return cars_distrib_prov, cars_euro_com

def align_ef_data_sources(ef):
    # EF: set to -1 the euro_level of electric vehicles.
    ef_expr = _get_expressions(type="ef")
    ef_params = [n for n in ef_expr.meta.root_names() if n != 'speed']
    ef = ef.with_columns(
        pl.when(pl.col("fuel") != "battery_electric")
        .then(pl.col("euro_level"))
        .otherwise(pl.lit(-1))
    ).unique(["fuel", "segment", "euro_level", "pollutant", *ef_params], keep="first", maintain_order=True)

    # Fix the names of the fuel to manage the bifuels
    ef = ef.with_columns(
        pl.when(pl.col("fuel").str.contains("~"))
        .then(pl.col("fuel").str.extract("(.+)_~_.+"))
        .otherwise(pl.col("fuel"))
        .alias("fuel_type"),
        pl.when(pl.col("fuel").str.contains("~"))
        .then(pl.col("fuel").str.extract(".+_~_(.+)"))
        .otherwise(pl.col("fuel"))
        .alias("fuel_actual")
    )
    # Si aggiungono due colonne fuel_type e fuel_actual per gestire i bifuel 
    # (es. lpg_bifuel~lpg → fuel_type="lpg_bifuel", fuel_actual="lpg"), 
    # distinguendo il tipo di motore dal carburante effettivamente usato; 
    # per i non-bifuel entrambe coincidono col fuel originale.

    return ef

def get_joint_from_marginals(cars_distrib_prov, cars_euro_com, cars_fuel_com):
    # Si stima la distribuzione congiunta [comune, euro_level, fuel, segment] combinando distribuzioni marginali da fonti diverse

    # # Dai dati di provincia, proporzione di ogni segment condizionato al livello euro
    # prov_segment_in_euro = (
    #     cars_distrib_prov.group_by("euro_level", "segment")
    #     .agg(pl.col("len").sum())
    #     .sort("euro_level", "segment")
    #     .with_columns(
    #         prop_segment_in_euro = pl.col("len") / pl.col("len").sum().over("euro_level")
    #     )
    # )

    # Dai dati di provincia, proporzione di ogni fuel condizionato al livello euro
    prov_fuel_in_euro = ( 
        cars_distrib_prov.group_by('euro_level', 'fuel') # assumption made: see remark below
        .agg(pl.col('len').sum())
        .with_columns(
            prop = pl.col('len') / pl.col('len').sum().over('euro_level')
        )
        .sort('euro_level', 'prop')
    )
    # Remark: the probability to have fuel knowing the euro level: intuitively, since euro level distrib is
    # more flat, knowing the euro level reduces entropy more than knowing the fuel. So
    # assuming this distribution to be uniform in space makes us lose more info. euro level
    # should depend more on space than fuel, since economic disparities impact it a lot, but
    # not so much the fuel choice (except for electric, which is solved because it's a
    # separate level) -> assume distrib of fuel given euro level to be equal to its province average 
    # Solution: actually use uniform fuel distribution knowing the euro level

    # Dalle proporzioni di fuel e segment condizionate a classe euro, distribuzione congiunta P(euro_level, fuel, segment | comune) per ogni comune
    # Apply fuel vs euro distribution from province and apply it uniformly in every comune. 
    # But: also know the distribution of fuels in every comune, so use it.
    com_prop_expr = pl.col("count") / pl.col("count").sum().over("comune")
    cars_joint = (
        cars_euro_com.with_columns(prop=com_prop_expr)
        .join(cars_fuel_com.with_columns(prop=com_prop_expr), on="comune", suffix="_fuel")
        .join(
            prov_fuel_in_euro.select("euro_level", "fuel", "prop"),
            on=["euro_level", "fuel"],
            suffix="_fuel_in_euro_prov",
        )
        .with_columns(
            prop_euro_and_fuel=pl.col("prop") * pl.col("prop_fuel_in_euro_prov")
        )
        .with_columns(
            (pl.col('prop_euro_and_fuel') * pl.col('prop_fuel') / pl.col('prop_euro_and_fuel').sum().over('comune', 'fuel')).fill_nan(0),
        )
        .select('comune', 'euro_level', 'fuel', 'prop_euro_and_fuel')
        .sort(pl.all())
    )

    prov_segment_in_joint = cars_distrib_prov.with_columns(
        prop=pl.col("len") / pl.col("len").sum().over("euro_level", "fuel")
    ).drop("len")

    cars_joint = (
        cars_joint.join(prov_segment_in_joint, on=["euro_level", "fuel"])
        .with_columns(prop_all=pl.col("prop") * pl.col("prop_euro_and_fuel"))
        .drop("prop", "prop_euro_and_fuel")
    )

    # Geo matching comune / OD statistical areas
    comune_gdf = data_reader.comune_shapes(PathRegistry.COMUNE_SHAPES)
    cars_joint = cars_joint.join(
        pl.DataFrame(comune_gdf[["com_id", "comune"]]).with_columns(
            data_cleaner.municipality_norm_expr("comune")
        ),
        on="comune",
        how="left",
    )
    assert (
        cars_joint.filter(pl.col("com_id").is_null()).select("comune").unique().item()
        == "non_definito_bo"
    )
    cars_joint = cars_joint.filter(pl.col("com_id").is_not_null()).drop("comune")

    return cars_joint

def get_emissions():

    logging.info("Reading and processing the car fleet data in Bologna Province")
    # CAR FLEET: Bologna province [joint distribution segment, fuel, euro_level]
    cars_distrib_prov = get_car_fleet_prov()

    logging.info("Reading and processing the car fleet data in each Comune of the Bologna Province")
    # CAR FLEET: Comuni [marginal distribution segment, fuel, euro_level]
    cars_euro_com, cars_fuel_com = get_car_fleet_comuni()

    logging.info("Reading and processing emission factors from EEA")
    # EEA data: emission factors
    ef = get_efs()

    logging.info("Aligning data sources")
    # Alignment for the different data sources.
    cars_distrib_prov, cars_fuel_com = align_fuel_data_sources(cars_distrib_prov, cars_fuel_com)
    cars_distrib_prov, cars_euro_com = align_euro_level_data_sources(cars_distrib_prov, cars_euro_com)
    ef = align_ef_data_sources(ef)
    
    logging.info("Computing joint distribution per Comune of segment, fuel, euro_level")
    # Estimate CAR FLEET: Comuni [joint distribution segment, fuel, euro_level]
    cars_joint = get_joint_from_marginals(cars_distrib_prov, cars_euro_com, cars_fuel_com)

    logging.info("Prepare emission factor tables of CO2 and pollutants")
    # EMISSIONS
    # calorific in MJ/kg and co2 in kg/kg of fuel
    fuel_combustion = (
        pl.read_csv(PathRegistry.FUEL_COMBUSTION_CSV)
        .with_columns(data_cleaner.str_norm_expr(pl.col('fuel')))
    )

    poll_ef = ef.filter(pl.col('pollutant') != 'ec')

    co2_ef = ef.filter(pl.col('pollutant') == 'ec').join(
        fuel_combustion.select(
            pl.col("fuel").alias("fuel_actual"), cs.numeric().name.prefix("fuel_")
        ),
        on="fuel_actual",
    )
    
    return cars_joint, poll_ef, co2_ef

def get_euro_class_split():
    logging.info("Reading and processing the car fleet data in Bologna Province")
    # CAR FLEET: Bologna province [joint distribution segment, fuel, euro_level]
    cars_distrib_prov = get_car_fleet_prov()

    logging.info("Aggregaring the car fleet per euro level")
    # exclude euro_level == -1 (electric) before aggregating
    euro_counts = (
        cars_distrib_prov.filter(pl.col("euro_level") != -1)
        .group_by("euro_level")
        .agg(pl.col("len").sum().alias("count"))
        .sort("euro_level")
    )
    total = cars_distrib_prov.select(pl.col("len").sum()).item()
    return (
        euro_counts
        .with_columns((pl.col("count") / total).alias("prop"))
    )    

def get_euro_class_emissions(poll_ef, ef_expr):
    logging.info("Computing the average emission factor per euro class")
    nox = (
        poll_ef
        .filter(pl.col("pollutant") == "nox")
        .with_columns(speed=70)
        .with_columns(emission=ef_expr)
    )
    return nox.group_by("euro_level") \
    .agg(pl.col("emission").mean()) \
    .sort("euro_level")

def test_emissions_co2():
    _, _, co2_ef = get_emissions()
    co2_expr = _get_expressions(type="co2")
    ef_expr = _get_expressions(type="ef")
    # Example computation of CO2/km:
    a = co2_ef.with_columns(speed=70).with_columns(ec=ef_expr).with_columns(co2_kg=co2_expr)
    print(a)

def test_emissions_poll():
    _, poll_ef, _ = get_emissions()
    ef_expr = _get_expressions(type="ef")
    # Example computation of pollutant mass or number/km:
    b = poll_ef.with_columns(speed=70).with_columns(emission=ef_expr)
    print(b)

def main():
    logging.basicConfig(level=logging_level)
    with timed():
        euro_class_split = get_euro_class_split()
        cars_joint, poll_ef, co2_ef = get_emissions()
        euro_class_emissions = get_euro_class_emissions(poll_ef, _get_expressions(type="ef"))
        
        logging.info("Saving the results")
        PathRegistry.ANALYSIS_OUTPUTS.mkdir(parents=True, exist_ok=True)
        cars_joint.write_parquet(PathRegistry.CARS_JOINT_BY_COMUNE)
        poll_ef.write_parquet(PathRegistry.POLL_EF_PARQUET)
        co2_ef.write_parquet(PathRegistry.CO2_EF_PARQUET)
        euro_class_split.write_parquet(PathRegistry.EURO_CLASS_SPLIT_PARQUET)
        euro_class_emissions.write_parquet(PathRegistry.EURO_CLASS_EMISSIONS_PARQUET)

if __name__ == "__main__":
    main()
