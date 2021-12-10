from zarr.creation import create
import numpy as np

from zarr import indexing

SIZE = np.iinfo(np.int32).max
ORIGIN = SIZE // 2
USE_ZARR_ATTR = (
    "store", "path", "name", "read_only", "chunk_store", "dtype", "compression", "compression_opts",
    "dimension_separator", "fill_value", "order", "synchronizer", "filters", "attrs", "itemsize", "nbytes_stored", "nchunks_initialized",
    "is_view", "info",  "write_empty_chunks", "ndim"
)


def _apply_offset(selection, ndim=None):
    if ndim is not None and (not isinstance(selection, tuple) or len(selection) != ndim):
        raise ValueError("Must subset each axis on a BoundlessArray")
    if isinstance(selection, tuple):
        return tuple([_apply_offset(x) for x in selection])
    elif isinstance(selection, slice):
        if selection.start is None or selection.stop is None:
            raise ValueError("Cannot use unbounded on a BoundlessArray")
        return slice(selection.start + ORIGIN, selection.stop + ORIGIN, selection.step)
    elif indexing.is_integer(selection):
        return selection + ORIGIN
    elif indexing.is_bool_array(selection):
        raise TypeError(f"Cannot mask a BoundlessArray")
    elif indexing.is_integer_list or indexing.is_integer_array(selection):
        return TypeError(f"Cannot multiindex a BoundlessArray")
    else:
        raise TypeError(f"Unexpected selection type '{type(selection)}'")


class BoundlessArray:
    def __init__(self, zarray):
        self.zarray = zarray

    @classmethod
    def create(cls, ndim, **kwargs):
        assert ndim > 0
        shape = (SIZE, ) * ndim
        kwargs.setdefault("write_empty_chunks", False)
        return cls(create(shape=shape, **kwargs))

    def __getattribute__(self, item):
        if item in USE_ZARR_ATTR:
            return getattr(self.zarray, item)
        return object.__getattribute__(self, item) 

    def __getitem__(self, selection):
        return self.zarray[_apply_offset(selection, ndim=self.ndim)]

    def __setitem__(self, selection, value):
        self.zarray[_apply_offset(selection, ndim=self.ndim)] = value


if __name__ == "__main__":
    boundless = BoundlessArray.create(ndim=2)
    boundless[0, -1000] = 6
    boundless[2**10, 0] = 7
    print(boundless.nchunks_initialized)
    print(boundless[0, -1000], boundless[2**10, 0])
    print(boundless[0:10, -1000])
