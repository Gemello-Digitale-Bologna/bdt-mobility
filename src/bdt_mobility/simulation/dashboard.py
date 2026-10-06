if __name__ == "__main__":
    import streamlit.runtime
    if not streamlit.runtime.exists():
        import subprocess
        import sys
        sys.exit(subprocess.call(["streamlit", "run", __file__, *sys.argv[1:]]))


import numpy as np
from contextlib import contextmanager
from datetime import timedelta
from functools import partial
from civic_digital_twins.dt_model import DistributionIndex
import streamlit as st
import streamlit.components.v1 as components

from bdt_mobility.model import MobilityModel, compute_kpis, evaluate_model
from bdt_mobility.model.axis import ZoneInTimeseriesIndex
from bdt_mobility.simulation.utils import to_number, to_time
from bdt_mobility.simulation.view import plot_flow, plot_traffic, plot_emissions, set_map
from bdt_mobility.simulation.i18n import t, get_lang
from bdt_mobility.model.config_vars import behavioral_params, policy_params, config_params, ENSEMBLE_SIZE, ETHICAL_YEAR

from bdt_mobility.utils import data_reader, constants, paths


import pandas as pd
from scipy import stats


def get_zones():
    # Spatial data
    gdf_zoi = data_reader.ZOI_shape(paths.PathRegistry.ZOI_BORDER)

    aree_gdf_inside_raw, aree_gdf_outside_raw = data_reader.AOI_shapes(
        paths.PathRegistry.OD_SHAPE_ZONE,
        aoi_type="od", df_around=gdf_zoi, return_type="both"
    )

    aree_gdf_inside_raw["id_zone"] = aree_gdf_inside_raw["id_zone"].astype(int)
    aree_gdf_inside_raw["type_zoi"] = "inside"
    aree_gdf_outside_raw["id_zone"] = aree_gdf_outside_raw["id_zone"].astype(int)
    aree_gdf_outside_raw["type_zoi"] = "outside"
    aree_gdf_inside_raw = aree_gdf_inside_raw.to_crs(constants.CRS_LATLONG)
    aree_gdf_outside_raw = aree_gdf_outside_raw.to_crs(constants.CRS_LATLONG)
    aree_gdf = pd.concat([aree_gdf_inside_raw, aree_gdf_outside_raw], axis=0)
    aree_gdf = aree_gdf.reset_index(drop=True).reset_index().sort_values(by="index")
    aree_gdf_inside = aree_gdf[aree_gdf["type_zoi"] == "inside"]

    in_zones = np.array(aree_gdf.loc[aree_gdf["type_zoi"] == "inside", "index"].unique())

    zones_in_to_names = {id: aree_gdf[aree_gdf["index"] == id]["name"].iloc[0] for id in in_zones}  # type: ignore[union-attr]
    names_in_to_zones = {aree_gdf[aree_gdf["index"] == id]["name"].iloc[0]: id for id in in_zones}  # type: ignore[union-attr]

    return aree_gdf_inside, zones_in_to_names, names_in_to_zones


def get_ui_values():
    return {
        p["id"].name: st.session_state[p["id"].name]
        for p in all_params
    }


def build_overrides(ui_values):
    # TODO: the code works only with indices and distribution indices of type uniform!!
    overrides = {}
    for p in all_params:
        name = p["id"].name
        id = p['id']
        value = ui_values[name]

        if "type" in p and p["type"] == "Time":
            if isinstance(p["id"], DistributionIndex):
                loc, scale = value
                overrides[id] = stats.uniform(loc=to_number(loc), scale=to_number(scale))
            else:
                overrides[id] = to_number(value)

        else:
            if isinstance(p["id"], DistributionIndex):
                loc, scale = value
                overrides[id] = stats.uniform(loc=loc, scale=scale)
            else:
                overrides[id] = value

    return overrides


if '__model__' not in st.session_state:
    ## Initialize
    default_config = MobilityModel.default_config(
        config_params=config_params
    )
    default_inputs = MobilityModel.default_inputs(
        config=default_config, 
        policy_params=policy_params, 
        behavioral_params=behavioral_params
    )
    st.session_state['__model__'] = MobilityModel(
        config=default_config,
        inputs=default_inputs,
    )
    ## Evaluate
    st.session_state['__subs__'] = evaluate_model(
        model=st.session_state['__model__'],
        overrides=None,
        size=ENSEMBLE_SIZE,
        rng=89,
    )
    st.session_state['__saved_models__'] = {}
    st.session_state['__selected__'] = "Simulazione"


m = st.session_state['__model__']
aree_gdf_inside, zones_to_names, names_in_to_zones = get_zones()
in_zones = aree_gdf_inside["index"].to_numpy()
set_map(aree_gdf_inside)

def _build_params():
    """Build parameter list with current language translations."""
    return [
        {"name": t("params_policy"),
         "params": [
             {'id': m.I_P_start_time, 'label': t("params_start_time"), 'value': to_time(m.I_P_start_time.concrete_default),
              'min': to_time(0), 'max': to_time(3600 * 24 - 900), 'step': timedelta(seconds=900), 'type': 'Time'},
             {'id': m.I_P_end_time, 'label': t("params_end_time"), 'value': to_time(m.I_P_end_time.concrete_default),
              'min': to_time(0), 'max': to_time(3600 * 24 - 900), 'step': timedelta(seconds=900), 'type': 'Time'},
             {'id': m.I_P_fraction_exempted, 'label': t("params_exempted"), 'value': m.I_P_fraction_exempted.concrete_default,
              'min': 0.0, 'max': 1.0, 'step': 0.05},
         ]},
        {"name": t("params_ticket"),
         "params": [
             {'id': m.I_P_cost[e], 'label': t("params_ticket_cost").format(e), 'value': m.I_P_cost[e].concrete_default,
              'min': 0.0, 'max': 10.0, 'step': 0.25} for e in range(7)
         ]},
        {"name": t("params_behavior"),
         "params": [
             {'id': m.I_B_p50_cost, 'label': t("params_acceptable_cost"),
              'value': (m.I_B_p50_cost.params["loc"], m.I_B_p50_cost.params["scale"]),
              'min': 0.0, 'max': 10.0, 'step': 0.25},
             {'id': m.I_B_p50_anticipating, 'label': t("params_propensity_forward"),
              'value': m.I_B_p50_anticipating.concrete_default,
              'min': 0.0, 'max': 6.0, 'step': 0.10},
             {'id': m.I_B_p50_postponing, 'label': t("params_propensity_postpone"),
              'value': m.I_B_p50_postponing.concrete_default,
              'min': 0.0, 'max': 6.0, 'step': 0.10}]
              +
              ([] if config_params["time_shift_strategy"] == "fixed" else [
             {'id': m.I_B_p50_anticipation, 'label': t("params_median_forward"),
                'value': m.I_B_p50_anticipation.concrete_default,
                'min': 0.10, 'max': 6.0, 'step': 0.10},
             {'id': m.I_B_p50_postponement, 'label': t("params_median_postpone"),
                'value': m.I_B_p50_postponement.concrete_default,
                'min': 0.10, 'max': 6.0, 'step': 0.10} ]),
        },
        {"name": t("params_tpm"),
         "params": [
              {'id': m.I_P_pt_frequency_modification, 'label': t("params_pt_frequency"), 'value': m.I_P_pt_frequency_modification.concrete_default,
              'min': -1.0, 'max': 1.0, 'step': 0.25},
              {'id': m.I_P_pt_capillarity_modification, 'label': t("params_pt_capillarity"), 'value': m.I_P_pt_capillarity_modification.concrete_default,
              'min': -1.0, 'max': 1.0, 'step': 0.25},
              {'id': m.I_P_pt_cost_modification, 'label': t("params_pt_cost"), 'value': m.I_P_pt_cost_modification.concrete_default,
              'min': -1.0, 'max': 1.0, 'step': 0.25},
              {'id': m.I_P_pt_time_modification, 'label': t("params_pt_time"), 'value': m.I_P_pt_time_modification.concrete_default,
              'min': -1.0, 'max': 1.0, 'step': 0.25},
         ]},
        {"name": t("params_tpm_behavior"),
         "params":
              ([] if config_params["modal_shift_option"] == "no" or config_params["modal_shift_option"] == "active" else [
             {'id': m.I_B_pt_capillarity, 'label': t("params_behavior_capillarity"),
              'value': m.I_B_pt_capillarity.concrete_default,
              'min': 0.0, 'max': 10.0, 'step': 0.5},
             {'id': m.I_B_pt_frequency, 'label': t("params_behavior_frequency"),
              'value': m.I_B_pt_frequency.concrete_default,
              'min': -3.0, 'max': 0.0, 'step': 0.05},
             {'id': m.I_B_pt_cost, 'label': t("params_behavior_cost"),
              'value': m.I_B_pt_cost.concrete_default,
              'min': -1.0, 'max': 0.0, 'step': 0.05},
             {'id': m.I_B_pt_time, 'label': t("params_behavior_time"),
              'value': m.I_B_pt_time.concrete_default,
              'min': -0.1, 'max': 0.0, 'step': 0.01} ])
         },
    ]

params = _build_params()

all_params = [p for g in params for p in g["params"]]

for p in all_params:
    if p['id'].name not in st.session_state:
        st.session_state[p['id'].name + "_default"] = p['value']
        st.session_state[p['id'].name] = p['value']

############################## DOWNLOAD ##############################

time_index = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 5)]

def prepare_dataframe(subs, index_ref, zone_index_ref, index_mod, zone_index_mod,
                      include_ref, include_mod, include_zones):
    def zone_series(index, zone):
        if isinstance(index, ZoneInTimeseriesIndex):
            return np.asarray(subs.expected_value(index))[zone]
        return np.asarray(subs[index[zone]])

    df = pd.DataFrame(index=pd.Index(time_index, name=t("plot_time")))
    if include_ref:
        df["TOTALE (rif)"] = subs[index_ref].mean(axis=0)
        if include_zones:
            for z in in_zones:
                zone_name = zones_to_names[z]
                df[f"{zone_name} (rif)"] = zone_series(zone_index_ref, z)
    if include_mod:
        df["TOTALE (mod)"] = subs[index_mod].mean(axis=0)
        if include_zones:
            for z in in_zones:
                zone_name = zones_to_names[z]
                df[f"{zone_name} (mod)"] = zone_series(zone_index_mod, z)
    return df

@st.dialog(t("dialog_download_title"))
def download(name, file_name, df_function):
    def download_js(df):
        return """
            <button type="button" id="picker">Download</button>
            <script>
                async function run() {
                    const handle = await showSaveFilePicker({
                        suggestedName: '%s',
                        types: [{
                            description: 'CSV Data',
                            accept: {'text/plain': ['.csv']},
                        }],
                    });
                    const blob = new Blob([`%s`]);
                    const writableStream = await handle.createWritable();
                    await writableStream.write(blob);
                    await writableStream.close();
                    parent.document.querySelector('[aria-label="Close"]').click()
                }
                document.getElementById("picker").onclick = run
            </script>
            """ % (file_name, df.to_csv())

    st.write(t("dialog_download_progress").format(name))
    include_reference = st.checkbox(t("dialog_download_include_ref"), value=True)
    include_modified = st.checkbox(t("dialog_download_include_mod"), value=True)
    include_zones = st.checkbox(t("dialog_download_include_zones"), value=False)
    df = df_function(include_reference, include_modified, include_zones)
    components.html(download_js(df), height=30)

################################# SAVE AND GET MODELS

def save_model(name, subs, params):
    st.session_state['__saved_models__'][name] = ( subs, params )

def get_saved_model(name):
    return st.session_state['__saved_models__'][name]

def get_saved_models_list():
    return list(st.session_state['__saved_models__'].keys())

def get_params():
    return {p['id'].name: st.session_state[p['id'].name] for p in all_params}


def get_model_name():
    if st.session_state.get("__model_name_key__"):
        return t(st.session_state["__model_name_key__"])
    return st.session_state["__model_name__"]

def set_params(pvals):
    for p in all_params:
        p['value'] = pvals[p['id'].name]
        st.session_state[p['id'].name] = p['value']

@st.dialog(t("dialog_save_title"))
def save_model_dialog():
    name = st.text_input(t("dialog_save_name"))
    existing = name in get_saved_models_list()
    if st.button(t("btn_confirm"), key="confirm-save-model"):
        if name == "":
            st.toast(t("dialog_save_error_empty"))
        elif name == t("scenario_current"):
            st.toast(t("dialog_save_error_reserved"))
        elif existing:
            st.toast(t("dialog_save_error_exists"))
        else:
            save_model(name,
                       st.session_state['__subs__'],
                       get_params(),
                       )
            st.session_state['__model_name__'] = name
            st.session_state.pop('__model_name_key__', None)
            st.session_state['__model_changed__'] = False
            st.rerun()

@st.dialog(t("dialog_load_title"))
def get_saved_model_dialog():
    st.markdown(t("dialog_load_warning"))
    name = st.selectbox(t("dialog_load_name"),
                        get_saved_models_list(),
                        placeholder=t("dialog_load_placeholder"),
                        )
    if st.button(t("btn_confirm"), key="confirm-get-saves-model"):
        if name is not None:
            st.session_state['__subs__'], pvals = get_saved_model(name)
            set_params(pvals)
            st.session_state['__model_name__'] = name
            st.session_state.pop('__model_name_key__', None)
            st.session_state['__model_changed__'] = False
        st.rerun()

if "__model_name__" not in st.session_state.keys():
    st.session_state["__model_name_key__"] = "scenario_example"
    st.session_state["__model_name__"] = t("scenario_example")
    st.session_state["__model_changed__"] = False
    save_model(st.session_state["__model_name__"],
               st.session_state['__subs__'],
               get_params())


################################## KPIS

def _build_kpi_viz():
    """Build KPI visualization list with current language translations."""
    return [
        {
            "category": t("kpi_flow_category"),
            "kpis": {
                'Base inflow [veh/day]': t("kpi_flow_base"),
                'Modified inflow [veh/day]': t("kpi_flow_modified"),
                'Inflow difference [veh/day]': t("kpi_flow_difference"),
                'Time-shifted inflow [veh/day]': t("kpi_flow_shifted"),
            },
        },
        {
            "category": t("kpi_traffic_category"),
            "kpis": {
                'Base traffic [max veh]': t("kpi_traffic_base"),
                'Modified traffic [max veh]': t("kpi_traffic_modified"),
                'Traffic difference [max veh]': t("kpi_traffic_difference"),
            }
        },
        {
            "category": t("kpi_modal_category"),
            "kpis": {
                'Mode-shifted inflow [veh/day]': t("kpi_modal_shifted"),
            }
        },
        {
            "category": t("kpi_emissions_category"),
            "kpis": {
                'Base emissions [NOx gr/day]': t("kpi_emissions_base"),
                'Emissions [NOx gr/day]': t("kpi_emissions_modified"),
                'Emissions difference [NOx gr/day]': t("kpi_emissions_difference"),
            }
        },
        {
            "category": t("kpi_revenue_category"),
            "kpis": {
                'Paying inflow [veh/day]': t("kpi_revenue_paying"),
                'Average cost [€/day]': t("kpi_revenue_avg_cost"),
                'Collected fees [€/day]': t("kpi_revenue_collected"),
            }
        },
    ]

kpi_viz = _build_kpi_viz()

def display_kpis(subs):
    kpis = compute_kpis(m, subs)
    for c in kpi_viz:
        st.markdown(f'**{c["category"]}**')
        for p, l in c['kpis'].items():
            if p in kpis.keys():
                st.write(f'- {l}: {kpis[p]:_}'.replace('_', '$\\,$'))


#################################################################################
#####################          STREAMLIT APP              #######################
#################################################################################
st.set_page_config(
    page_title=t("dashboard_title"),
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown(
    """
    <style>
        div.block-container{padding-top:1.5rem;}
        [data-testid="stSidebar"]{
            visibility: hidden;
        }
        div.st-key-custom_viz_setup_expander [data-testid=stExpander] details summary {
            background-color: rgb(255, 75, 75);
            color: white;
        }
        div.st-key-custom_viz_setup_expander [data-testid=stExpander] details summary p{
            font-size: 1rem;
        }
        div.st-key-custom_kpis_expander [data-testid=stExpander] details summary p{
            font-size: 28px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True
)

HORIZONTAL_STYLE = """
<style class="hide-element">
    /* Hides the style container and removes the extra spacing */
    .element-container:has(.hide-element) {
        display: none;
    }
    /*
        The selector for >.element-container is necessary to avoid selecting the whole
        body of the streamlit app, which is also a stVerticalBlock.
    */
    div[data-testid="stVerticalBlock"]:has(> .element-container .horizontal-marker) {
        display: flex;
        flex-direction: row !important;
        flex-wrap: wrap;
        gap: 0.5rem;
        align-items: baseline;
    }
    /* Buttons and their parent container all have a width of 704px, which we need to override */
    div[data-testid="stVerticalBlock"]:has(> .element-container .horizontal-marker) div {
        width: max-content !important;
    }
    /* Just an example of how you would style buttons, if desired */
    /*
    div[data-testid="stVerticalBlock"]:has(> .element-container .horizontal-marker) button {
        border-color: red;
    }
    */
</style>
"""

@contextmanager
def st_horizontal():
    st.markdown(HORIZONTAL_STYLE, unsafe_allow_html=True)
    with st.container():
        st.markdown('<span class="hide-element horizontal-marker"></span>', unsafe_allow_html=True)
        yield
################################# TITLE

st.title(t("dashboard_title"))

# Language selector - placed below title to avoid overlap with Streamlit header
col_title, col_lang = st.columns([6, 1])
with col_lang:
    lang_options = {"Italiano": "it", "English": "en"}
    current_lang = get_lang()
    current_label = [k for k, v in lang_options.items() if v == current_lang][0]
    selected_label = st.selectbox("Language", list(lang_options.keys()),
                                  index=list(lang_options.keys()).index(current_label))
    selected_lang = lang_options[selected_label]
    if selected_lang != current_lang:
        st.session_state["__lang__"] = selected_lang
        st.rerun()

################################# VISUALIZATION PARAMETERS

ZONE = 0
ZONE_NAME = ""
TIME = (to_time(0), to_time(3600 * 24))

c_buttons, c_viz = st.columns([3, 0.75])
with c_viz:
    with st.container(key="custom_viz_setup_expander"):
        with st.expander(t("viz_settings"), False):
            viewtype = st.selectbox(t("viz_type"), [t("viz_full_area"), t("viz_zone_selection"), t("viz_map")])
            if viewtype == t("viz_zone_selection"):
                multi_zone = st.checkbox(t("viz_multi_zone"))
                if multi_zone:
                    zone = st.multiselect(t("viz_zones"), sorted(names_in_to_zones), placeholder=t("viz_zones_placeholder"))
                else:
                    zone = st.selectbox(t("viz_zone"), sorted(names_in_to_zones))
            if viewtype == t("viz_map"):
                time_interval = st.checkbox(t("viz_time_interval"))
                if time_interval:
                    time = st.slider(t("viz_time"), value=TIME,
                                     min_value=to_time(0), max_value=to_time(3600 * 24), step=timedelta(seconds=900),
                                     format="HH:mm")
                else:
                    time = st.slider(t("viz_time"), value=to_time(3600 * 6),
                                     min_value=to_time(0), max_value=to_time(3600 * 24), step=timedelta(seconds=900),
                                     format="HH:mm")
            match viewtype:
                case view if view == t("viz_full_area"):
                    ZONE = 0
                case view if view == t("viz_zone_selection"):
                    if not zone:
                        ZONE = in_zones
                        ZONE_NAME = sorted(names_in_to_zones)
                    elif type(zone) is list:
                        ZONE = [names_in_to_zones[z] for z in zone]
                        ZONE_NAME = zone
                    else:
                        ZONE = [names_in_to_zones[zone]]
                        ZONE_NAME = [zone]
                case view if view == t("viz_map"):
                    ZONE = -1
                    TIME = time
                case _:
                    ZONE = 0

with c_buttons:
    if st.session_state['__selected__'] == "Comparazione":
        if st.button(t("btn_back"), type="primary"):
            st.session_state['__selected__'] = "Simulazione"
            st.rerun()
    else:  # "Simulazione"
        with st_horizontal():
            if st.button(t("btn_save"), type="primary"):
                save_model_dialog()
            if st.button(t("btn_load"), type="primary"):
                get_saved_model_dialog()
            if st.button(t("btn_compare"), type="primary"):
                st.session_state['__selected__'] = "Comparazione"
                st.rerun()

if st.session_state['__selected__'] == "Comparazione":
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(t("scenario_a"))
        s1 = st.selectbox(t("scenario_a"),
                          [t("scenario_current")] + get_saved_models_list(),
                          label_visibility="collapsed")
        if s1 == t("scenario_current"):
            subs1 = st.session_state['__subs__']
        else:
            subs1, _ = get_saved_model(s1)
    with c2:
        st.subheader(t("scenario_b"))
        s2 = st.selectbox(t("scenario_b"),
                          [t("scenario_current")] + get_saved_models_list(),
                          label_visibility="collapsed")
        if s2 == t("scenario_current"):
            subs2 = st.session_state['__subs__']
        else:
            subs2, _ = get_saved_model(s2)
    with st.container(key="custom_kpis_expander"):
        with st.expander(t("header_indicators")):
            c1, c2 = st.columns(2)
            with c1:
                display_kpis(subs1)
            with c2:
                display_kpis(subs2)
    st.subheader(t("header_inflow"))
    c1, c2 = st.columns(2)
    with c1:
        fig = plot_flow(subs1, m, ZONE, TIME)
        st.plotly_chart(fig, width='content', key="flow_left")
    with (c2):
        fig = plot_flow(subs2, m, ZONE, TIME)
        st.plotly_chart(fig, width='content', key="flow_right")
    st.subheader(t("header_traffic"))
    c1, c2 = st.columns(2)
    with c1:
        fig = plot_traffic(subs1, m, ZONE, TIME)
        st.plotly_chart(fig, width='content', key="traffic_left")
    with (c2):
        fig = plot_traffic(subs2, m, ZONE, TIME)
        st.plotly_chart(fig, width='content', key="traffic_right")
    st.subheader(t("header_emissions"))
    c1, c2 = st.columns(2)
    with c1:
        fig = plot_emissions(subs1, m, ZONE, TIME)
        st.plotly_chart(fig, width='content', key="emissions_left")
    with (c2):
        fig = plot_emissions(subs2, m, ZONE, TIME)
        st.plotly_chart(fig, width='content', key="emissions_right")

if st.session_state['__selected__'] == "Simulazione":

    model_name_placeholder = st.empty()
    if st.session_state['__model_changed__']:
        model_name_placeholder.header(t("scenario_header_modified").format(get_model_name()))
    else:
        model_name_placeholder.header(t("scenario_header").format(get_model_name()))

    # Create a row layout
    c_kpis, c_plot, c_form = st.columns([1, 2, 0.75])

    # Parameters
    with c_form:
        c_form.header(t("header_parameters"))
        for g in params:
            with st.expander(f'**{g["name"]}**', False):
                for p in g["params"]:
                    time_format = 'HH:mm' if 'type' in p.keys() and p['type'] == 'Time' else None
                    st.slider(p['label'], min_value=p['min'], max_value=p['max'], step=p['step'],
                              format=time_format, key=p['id'].name)


    with (c_plot):
        changed = False
        which_changed = []
        for p in all_params:
            if isinstance(p['id'], DistributionIndex):
                (l, s) = st.session_state[p['id'].name]
                (l2, s2) = p['value']
                if (l2, s2) != (l, s):
                    changed = True
                    which_changed.append(p['id'].name)
            else:
                if p['value'] != st.session_state[p['id'].name]:
                    changed = True
                    which_changed.append(p['id'].name)

        if changed:
            ui_values = get_ui_values()
            overrides_dict = build_overrides(ui_values=ui_values)
            st.session_state['__subs__'] = evaluate_model(
                model=m,
                overrides=overrides_dict,
                size=ENSEMBLE_SIZE,
                rng=89,
            )
            st.session_state['__model_changed__'] = True
            model_name_placeholder.header(t("scenario_header_modified").format(get_model_name()))

        subs = st.session_state['__subs__']

        c1, c2 = st.columns([3,1])
        with c1:
            st.subheader(t("header_inflow"))
        with c2:
            download_button = st.button(t("btn_download"), type="primary", key="download_inflow")
            if download_button:
                download(t("header_inflow"), "inflow.csv",
                         partial(prepare_dataframe,
                                 subs, m.TS_inflow, m.TS_inflow_zone,
                                 m.I_modified_inflow, m.I_modified_inflow_zone))

        fig = plot_flow(subs, m, ZONE, TIME)
        st.plotly_chart(fig, width='content')

        c1, c2 = st.columns([3,1])
        with c1:
            st.subheader(t("header_traffic"))
        with c2:
            download_button = st.button(t("btn_download"), type="primary", key="download_traffic")
            if download_button:
                download(t("header_traffic"), "traffic.csv",
                         partial(prepare_dataframe,
                                 subs, m.I_traffic, m.TS_traffic_zone,
                                 m.I_modified_traffic, m.I_modified_traffic_zone))
        fig = plot_traffic(subs, m, ZONE, TIME)
        st.plotly_chart(fig, width='content')

        c1, c2 = st.columns([3,1])
        with c1:
            st.subheader(t("header_emissions"))
        with c2:
            download_button = st.button(t("btn_download"), type="primary", key="download_emissions")
            if download_button:
                download(t("header_emissions"), "emissions.csv",
                         partial(prepare_dataframe,
                                 subs, m.I_emissions, m.I_emissions_zone,
                                 m.I_modified_emissions, m.I_modified_emissions_zone))
        fig = plot_emissions(subs, m, ZONE, TIME)
        st.plotly_chart(fig, width='content')

#################### ETHICAL VISUALISATION #############################

        if config_params["enable_ethics"]:
            from bdt_mobility.modules.ethics.model.viz import plot_mpi_map

            # selection to navigate the cases
            ETHICAL_CASE_LABELS = {"elderly": t("ethical_case_elderly"), "minors": t("ethical_case_minors"), "income": t("ethical_case_income")}
            ETHICAL_GENDER_LABELS = {"None": t("ethical_gender_aggregate"), "female": t("ethical_gender_female"), "male": t("ethical_gender_male")}
            ec1, ec2 = st.columns(2)
            with ec1:
                ethical_case = st.selectbox(
                    t("ethical_select_case"),
                    list(ETHICAL_CASE_LABELS.keys()),
                    format_func=lambda k: ETHICAL_CASE_LABELS[k],
                )
            with ec2:
                ethical_gender = st.selectbox(
                    t("ethical_select_gender"),
                    list(ETHICAL_GENDER_LABELS.keys()),
                    format_func=lambda k: ETHICAL_GENDER_LABELS[k],
                )

            bundle = m.ethical[ethical_case]
            delta_values = bundle.extract_output_maps(subs)["delta_mpi"][ethical_gender]
            slice_spec = bundle.models[ethical_gender].slice_spec

            # plot the delta map
            st.subheader(t("ethical_delta_mpi_header").format(case=ETHICAL_CASE_LABELS[ethical_case], gender=ETHICAL_GENDER_LABELS[ethical_gender], year=ETHICAL_YEAR))
            fig = plot_mpi_map(
                delta_values,
                label=t("ethical_delta_mpi_label").format(case=ETHICAL_CASE_LABELS[ethical_case]),
                column=slice_spec.delta_mpi_name,
                od_zones=aree_gdf_inside,
                year=ETHICAL_YEAR,
            )
            st.plotly_chart(fig, width='content')


    with c_kpis:
        st.subheader(t("header_indicators"))
        display_kpis(subs)
