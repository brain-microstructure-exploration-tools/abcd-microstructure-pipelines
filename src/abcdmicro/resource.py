from __future__ import annotations


class Resource:
    """Base class for resources."""


class VolumeResource(Resource):
    """Base class for resources representing a volume or volume stack.
    An n-D array where n >= 3 and where three of the dimensions are spatial
    and have associated header information describing a patient coordinate system."""


class InMemoryVolumeResource(VolumeResource):
    """A volume resource that is loaded into memory.
    An n-D array where n >= 3 and where three of the dimensions are spatial
    and have associated header information describing a patient coordinate system."""


class BvalResource(Resource):
    """Base class for resources representing a list of b-values associated with a 4D DWI
    volume stack."""


class InMemoryBvalResource(BvalResource):
    """A b-value list that is loaded into memory."""


class BvecResource(Resource):
    """Base class for resources representing a list of b-vectors associated with a 4D DWI
    volume stack."""


class InMemoryBvecResource(BvecResource):
    """A b-vector list that is loaded into memory."""
