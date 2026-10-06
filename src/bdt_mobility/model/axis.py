from civic_digital_twins.dt_model import (
    TIME_AXIS, 
    DomainAxis,
    Index,
    SetType,
    SequenceType,
)

# Define the axis
ZONE_IN_AXIS = DomainAxis("internal zones", type=SetType())
ECL_AXIS = DomainAxis("euro class level", type=SequenceType())
ZONE_PT_AXIS = DomainAxis("public transport zones", type=SetType())

# Define the new Index with dimension 1 along the axis
class ZoneInIndex(Index):
    FIXED_AXES = (ZONE_IN_AXIS,)
    def __init__(self, name, value=None):
        super().__init__(name, value, axes=self.FIXED_AXES)

class EclIndex(Index):
    FIXED_AXES = (ECL_AXIS,)
    def __init__(self, name, value=None):
        super().__init__(name, value, axes=self.FIXED_AXES)

class ZonePtIndex(Index):
    FIXED_AXES = (ZONE_PT_AXIS,)
    def __init__(self, name, value):
        super().__init__(name, value, axes=self.FIXED_AXES)

# Define the new Index with dimension 2 along the axes
class ZoneInTimeseriesIndex(Index):
    FIXED_AXES = (ZONE_IN_AXIS, TIME_AXIS)
    def __init__(self, name, value=None):
        super().__init__(name, value, axes=self.FIXED_AXES)

class ZoneInEclIndex(Index):
    FIXED_AXES = (ZONE_IN_AXIS, ECL_AXIS)
    def __init__(self, name, value=None):
        super().__init__(name, value, axes=self.FIXED_AXES)

class TimeseriesEclIndex(Index):
    FIXED_AXES = (TIME_AXIS, ECL_AXIS)
    def __init__(self, name, value=None):
        super().__init__(name, value, axes=self.FIXED_AXES)

class ZonePtTimeseriesIndex(Index):
    FIXED_AXES = (ZONE_PT_AXIS, TIME_AXIS)
    def __init__(self, name, value=None):
        super().__init__(name, value, axes=self.FIXED_AXES)

class ZonePtEclIndex(Index):
    FIXED_AXES = (ZONE_PT_AXIS, ECL_AXIS)
    def __init__(self, name, value=None):
        super().__init__(name, value, axes=self.FIXED_AXES)
