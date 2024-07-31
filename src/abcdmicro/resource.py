from __future__ import annotations


class Resource:
    pass


class VolumeResource(Resource):
    pass


class InMemoryVolumeResource(VolumeResource):
    pass


class BvalResource(Resource):
    pass


class InMemoryBvalResource(BvalResource):
    pass


class BvecResource(Resource):
    pass


class InMemoryBvecResource(BvecResource):
    pass
