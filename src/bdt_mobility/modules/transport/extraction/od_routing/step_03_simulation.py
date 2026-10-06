# Simulation scripts for OD routing
# First, import the information on the traffic simulator to be executed from the context.
# Then, import the main function of the corresponding script, and execute it.

from bdt_mobility.modules.transport.extraction.od_routing.parameters import context
from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.extraction.platform.outputs import DynamicPlatformOutputs

MAIN_OUTPUTS = DynamicPlatformOutputs(PathRegistry.TRAFFIC_SIMULATOR, "table")

if context["simulator"] == "uxsim":
    from bdt_mobility.modules.transport.extraction.od_routing.simulation_uxsim import main
elif context["simulator"] == "fts":
    from bdt_mobility.modules.transport.extraction.od_routing.simulation_fts import main
elif context["simulator"] == "dsf":
    from bdt_mobility.modules.transport.extraction.od_routing.simulation_dsf import main
else:
    raise ValueError(f"Unknown simulator: {context["simulator"]}. Choose from uxsim, fts")

if __name__ == "__main__":
    main(ctx=context)