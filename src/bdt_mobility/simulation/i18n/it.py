"""Italian translations (default language)."""

# ===========================================================================
# Dashboard
# ===========================================================================
dashboard_title = "Digital Twin Studio - Mobilità Bologna"
scenario_current = "SCENARIO ATTUALE"
scenario_example = "Politica di esempio"
scenario_header = 'Scenario: "{}"'
scenario_header_modified = 'Scenario: modifica di "{}"'

# ===========================================================================
# Buttons
# ===========================================================================
btn_save = "Salva scenario"
btn_load = "Carica scenario"
btn_compare = ">> Confronto"
btn_back = "<< Simulazione"
btn_download = "Scarica dati"
btn_confirm = "Conferma"

# ===========================================================================
# Parameter groups
# ===========================================================================
params_policy = "Parametri generali regolamento"
params_ticket = "Costo ticket"
params_behavior = "Parametri comportamentali"
params_tpm = "Interventi su TPM"
params_tpm_behavior = "Parametri comportamentali / TPM"

# ===========================================================================
# Parameter labels
# ===========================================================================
params_start_time = "Ora inizio della policy (in hh:mm)"
params_end_time = "Ora termine della policy (in hh:mm)"
params_exempted = "Veicoli esonerati dalla policy (in % sul totale)"
params_ticket_cost = "Costo ingresso Euro {} (in \u20ac)"
params_acceptable_cost = "Soglia di costo accettabile (in \u20ac, mediano)"
params_propensity_forward = "Disponibilit\u00e0 anticipo viaggio (in ore, mediano)"
params_propensity_postpone = "Disponibilit\u00e0 posticipo viaggio (in ore, mediano)"
params_median_forward = "Tempo mediano di anticipo (in ore)"
params_median_postpone = "Tempo mediano mediano posticipo (in ore)"
params_pt_frequency = "Modifica % della frequenza del TPM"
params_pt_capillarity = "Modifica % della capillarit\u00e0 del TPM"
params_pt_cost = "Modifica % della tariffa del TPM"
params_pt_time = "Modifica % della differenza temporale TPM-auto"
params_behavior_capillarity = "Livello di importanza della capillat\u00e0 del TPM"
params_behavior_frequency = "Livello di importanza della frequenza del TPM"
params_behavior_cost = "Livello di importanza del costo del TPM"
params_behavior_time = "Livello di importanza della differenza temporale TPM-auto"

# ===========================================================================
# Section headers
# ===========================================================================
header_parameters = "Parametri"
header_inflow = "Veicoli in ingresso"
header_traffic = "Traffico"
header_emissions = "Emissioni"
header_indicators = "Indicatori"

# ===========================================================================
# Visualization settings
# ===========================================================================
viz_settings = "Impostazioni di visualizzazione"
viz_type = "Tipo di visualizzazione"
viz_full_area = "Intera Area Urbana"
viz_zone_selection = "Selezione Zone"
viz_map = "Mappa"
viz_multi_zone = "Zone multiple"
viz_zones = "Zone"
viz_zones_placeholder = "Tutte le zone"
viz_zone = "Zona"
viz_time_interval = "Intervallo"
viz_time = "Orario"

# ===========================================================================
# Comparison view
# ===========================================================================
scenario_a = "Scenario A"
scenario_b = "Scenario B"

# ===========================================================================
# Download dialog
# ===========================================================================
dialog_download_title = "Scaricamento dati"
dialog_download_progress = "Scaricamento {}..."
dialog_download_include_ref = "Includere dati di riferimento"
dialog_download_include_mod = "Includere dati modificati"
dialog_download_include_zones = "Includere dettagli zone"

# ===========================================================================
# Save dialog
# ===========================================================================
dialog_save_title = "Salvataggio scenario"
dialog_save_name = "Nome dello scenario"
dialog_save_error_empty = "**Errore:** Selezionare un nome per il salvataggio."
dialog_save_error_reserved = (
    "**Errore:** Il nome utilzzato \u00e8 riservato: scegliere un altro nome!"
)
dialog_save_error_exists = (
    "**Errore:** Il nome \u00e8 gi\u00e0 stato utilizzato: scegliere un altro nome!"
)

# ===========================================================================
# Load dialog
# ===========================================================================
dialog_load_title = "Caricamento scenaro"
dialog_load_warning = "**Attenzione!** Eventuali modifiche ai parametri andranno perse."
dialog_load_name = "Nome dello scenario"
dialog_load_placeholder = "Nessuno scenario disponible"

# ===========================================================================
# KPI categories
# ===========================================================================
kpi_flow_category = "Flusso (veicoli/giorno in ingresso nella ZOI)"
kpi_traffic_category = "Traffico (nr. massimo veicoli circolanti)"
kpi_modal_category = "Cambio modale (veicoli/giorno)"
kpi_emissions_category = "Emissioni (Nox gr/giorno)"
kpi_revenue_category = "Entrate"

# ===========================================================================
# KPI labels
# ===========================================================================
kpi_flow_base = "Flusso riferimento"
kpi_flow_modified = "Flusso modificato"
kpi_flow_difference = "Differenza flusso"
kpi_flow_shifted = "Flusso anticipato/posticipato"
kpi_traffic_base = "Traffico di riferimento"
kpi_traffic_modified = "Traffico modificato"
kpi_traffic_difference = "Differenza traffico"
kpi_modal_shifted = "Flusso trasferito su TPM"
kpi_emissions_base = "Emissioni di riferimento"
kpi_emissions_modified = "Emissioni modificate"
kpi_emissions_difference = "Differenza in emissioni"
kpi_revenue_paying = "Veicoli paganti (veicoli/giorno)"
kpi_revenue_avg_cost = "Costo medio per veicolo (\u20ac/giorno)"
kpi_revenue_collected = "Pagamenti collezionati (\u20ac/giorno)"

# ===========================================================================
# Ethical visualization
# ===========================================================================

ethical_select_case = "EthicalLens \U0001f50d sulla Popolazione:"
ethical_select_gender = "Aggregazione di Genere:"
ethical_case_elderly = "Anziani"
ethical_case_minors = "Famiglie con minori"
ethical_case_income = "Reddito"
ethical_gender_aggregate = "Aggregato"
ethical_gender_female = "Femminile"
ethical_gender_male = "Maschile"
ethical_delta_mpi_header = "\u0394MPI {case} \u2014 {gender} ({year})"
ethical_delta_mpi_label = "\u0394MPI {case}"

# ===========================================================================
# Plot labels (used in view.py)
# ===========================================================================
plot_time = "Ora"
plot_flow_label = "Flusso in ingresso"
plot_flow_unit = "veicoli/ora"
plot_flow_map_label = "Flusso"
plot_traffic_label = "Traffico"
plot_traffic_unit = "veicoli circolanti"
plot_traffic_map_unit = "max veicoli circolanti"
plot_emissions_label = "Emissioni"
plot_emissions_unit = "NOx gr/ora"
plot_emissions_map_unit = "NOx gr/giorno"

# ===========================================================================
# Plot legend (used in plot.py)
# ===========================================================================
plot_reference = "Riferimento"
plot_modified = "Modificata (mediana)"
plot_zone_code = "Codice zona"
