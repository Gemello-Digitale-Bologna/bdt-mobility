from bdt_mobility.simulation.plot import plot_field_graph, plot_multifield_graph, plot_map_graph, set_map
from bdt_mobility.simulation.i18n import t


def plot_flow(subs, m, zone=0, time=None):
    if isinstance(zone, int) and zone == 0:
        fig = plot_field_graph(subs.expected_value(m.I_modified_inflow),
                               horizontal_label=t("plot_time"),
                               vertical_label=t("plot_flow_label"),
                               vertical_unit=t("plot_flow_unit"),
                               vertical_transformer=lambda x: x * 12,
                               reference_line=subs.expected_value(m.TS_inflow))
    elif isinstance(zone, int) and zone == -1:
        fig = plot_map_graph(subs, m.I_modified_inflow_zone, time=time,
                             label=t("plot_flow_map_label"), unit=t("plot_flow_unit"))
    else:
        fig = plot_multifield_graph(subs, m.I_modified_inflow_zone, zone,
                                    horizontal_label=t("plot_time"),
                                    vertical_label=t("plot_flow_label"),
                                    vertical_unit=t("plot_flow_unit"),
                                    vertical_transformer=lambda x: x * 12,
                                    reference_index=m.TS_inflow_zone)
    return fig


def plot_traffic(subs, m, zone=0, time=None):
    if isinstance(zone, int) and zone == 0:
        fig = plot_field_graph(subs.expected_value(m.I_modified_traffic),
                               horizontal_label=t("plot_time"),
                               vertical_label=t("plot_traffic_label"),
                               vertical_unit=t("plot_traffic_unit"),
                               reference_line=subs.expected_value(m.I_traffic))
    elif isinstance(zone, int) and zone == -1:
        fig = plot_map_graph(subs, m.I_modified_traffic_zone, time=time,
                             label=t("plot_traffic_label"), unit=t("plot_traffic_map_unit"), function=max)
    else:
        fig = plot_multifield_graph(subs, m.I_modified_traffic_zone, zone,
                                    horizontal_label=t("plot_time"),
                                    vertical_label=t("plot_traffic_label"),
                                    vertical_unit=t("plot_traffic_unit"),
                                    reference_index=m.TS_traffic_zone)
    return fig


def plot_emissions(subs, m, zone=0, time=None):
    if isinstance(zone, int) and zone == 0:
        fig = plot_field_graph(subs.expected_value(m.I_modified_emissions),
                               horizontal_label=t("plot_time"),
                               vertical_label=t("plot_emissions_label"),
                               vertical_unit=t("plot_emissions_unit"),
                               vertical_transformer=lambda x: x * 12,
                               reference_line=subs.expected_value(m.I_emissions))
    elif isinstance(zone, int) and zone == -1:
        fig = plot_map_graph(subs, m.I_modified_emissions_zone, time=time,
                             label=t("plot_emissions_label"), unit=t("plot_emissions_map_unit"))
    else:
        fig = plot_multifield_graph(subs, m.I_modified_emissions_zone, zone,
                                    horizontal_label=t("plot_time"),
                                    vertical_label=t("plot_emissions_label"),
                                    vertical_unit=t("plot_emissions_unit"),
                                    vertical_transformer=lambda x: x * 12,
                                    reference_index=m.I_emissions_zone)
    return fig
