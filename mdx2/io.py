import importlib
import warnings

import h5py
import hdf5plugin
import numpy as np
from nexusformat.nexus import NXentry, NXfield, NXgroup, NXroot, NXvirtualfield
from nexusformat.nexus import nxload as nexus_nxload
from nexusformat.nexus.tree import NX_CONFIG

import mdx2

# FUNCTIONS FOR LOADING AND SAVING MDX2 CLASSES TO NEXUS FILES

# Large numeric arrays of mdx2 objects (hkl-table columns, peaks, Miller-index grids, correction maps, masks) are
# written with Blosc LZ4 + byte shuffle (array_field, used in the objects' to_nexus), not nexusformat's default for
# fields above its maxsize (gzip level 4 + HDF5 shuffle, ~100 KB chunks, single-threaded). Eiger 16M sweep (pod, one
# thread): the 53.7 M-row corrected table took 33 s to write and 7.5 s to read with gzip, 1.9 s and 1.4 s with Blosc
# (11% larger); the 3.2 GB Miller-index grid 63 s / 9.6 s vs 2.4 s / 1.5 s (7% larger). Byte shuffle beat bit shuffle
# on these float arrays. Reading needs the hdf5plugin filters (imported here). The image datastore has its own
# setting (mdx2.data.DATASTORE_COMPRESSION).
ARRAY_COMPRESSION = dict(
    compression=hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE), shuffle=False
)
TABLE_CHUNK_BYTES = 1 << 20  # table columns: ~1 MB chunks (h5py's guess for 1-D data is ~100 KB)


def array_field(value, name=None, column=False, **kwargs):
    """NXfield for a numeric array that may be large. Up to nexusformat's maxsize (10000 elements) a plain field (no
    compression, as nexusformat); above it ARRAY_COMPRESSION, and for column=True (1-D table columns) chunks of
    ~TABLE_CHUNK_BYTES; other arrays keep nexusformat's chunking."""
    value = np.asarray(value)
    if value.dtype.kind not in "biuf" or value.size <= NX_CONFIG["maxsize"]:
        return NXfield(value, name=name, **kwargs)
    opts = dict(ARRAY_COMPRESSION)
    if column and value.ndim == 1:
        opts["chunks"] = (max(1, min(value.size, TABLE_CHUNK_BYTES // value.dtype.itemsize)),)
    return NXfield(value, name=name, **opts, **kwargs)


def _patch_virtualfields(g):
    """Recursively patch NXvirtualfields in a Nexus group to set their shape."""
    if isinstance(g, NXgroup):
        for entry in g.entries.values():
            _patch_virtualfields(entry)
    elif isinstance(g, NXvirtualfield):
        # Patch virtual field shape - this has been thoroughly tested but may need
        # debugging if nexusformat package changes its internal handling of virtual datasets
        with g.nxfile as f:
            g._shape = f.get(g.nxpath).shape


def nxload(filename, mode="r", **kwargs):
    """Wrapper around nexusformat.nexus.nxload to check mdx2 version."""
    nxroot = nexus_nxload(filename, mode=mode, **kwargs)
    mdx2_version_file = nxroot.attrs.get("mdx2_version")
    if mdx2_version_file != mdx2.__version__:
        warnings.warn(
            f"mdx2 version mismatch: file version {mdx2_version_file}, installed version {mdx2.__version__}",
            UserWarning,
            stacklevel=2,
        )
    _patch_virtualfields(nxroot)
    return nxroot


def nxsave(nxsobj, filename, mode="w", **kwargs):
    """Wrapper around nexusformat.nexus.nxsave to add mdx2 version."""
    nxroot = NXroot(NXentry(nxsobj))
    nxroot.attrs["mdx2_version"] = mdx2.__version__
    nxroot.save(filename, mode=mode, **kwargs)
    return nxroot


def loadobj(filename, objectname):
    """Load mdx2 objects from nexus files with security validation.

    Parameters
    ----------
    filename : str
        Path to the nexus file
    objectname : str
        Name of the object within the nexus file entry

    Returns
    -------
    object
        The deserialized mdx2 object

    Raises
    ------
    ValueError
        If the module is not from the mdx2 package
    TypeError
        If the class does not have a callable from_nexus method
    """
    nxroot = nxload(filename, "r")
    nxs = nxroot["/entry/" + objectname]
    mod = nxs.attrs["mdx2_module"]
    cls = nxs.attrs["mdx2_class"]

    # Security: Only allow mdx2 modules to prevent arbitrary code execution
    if not mod.startswith("mdx2."):
        raise ValueError(f"Untrusted module '{mod}': only mdx2.* modules are allowed")

    # Import and get the class
    _tmp = importlib.__import__(mod, fromlist=[cls])
    Class = getattr(_tmp, cls)

    # Validate the class has the expected interface
    if not hasattr(Class, "from_nexus") or not callable(getattr(Class, "from_nexus")):
        raise TypeError(f"{mod}.{cls} does not have a callable from_nexus method")

    return Class.from_nexus(nxs)


def saveobj(obj, filename, name=None, append=False, mode="w"):
    # simple wrapper to save mdx2 objects as nxs files
    if not hasattr(obj, "to_nexus"):
        raise TypeError(f"{type(obj).__name__} does not support Nexus serialization (missing to_nexus method)")
    nxsobj = obj.to_nexus()
    if name is not None:
        nxsobj.rename(name)
    nxsobj.attrs["mdx2_module"] = type(obj).__module__
    nxsobj.attrs["mdx2_class"] = type(obj).__name__
    if append:
        root = nxload(filename, "r+")
        root["entry/" + nxsobj.nxname] = nxsobj
    else:
        nxsave(nxsobj, filename, mode=mode)
    return nxsobj


class NexusFileIndex:
    def __init__(self, *file_paths):
        self._object_index = {}

        def _inspect_nexus_file(filename):
            file_info = {}
            mdx2_object_info = {}
            try:
                with h5py.File(filename, "r") as f:
                    file_info = {k: v for k, v in f.attrs.items()}
                    if "mdx2_version" in file_info:  # it's an mdx2 file, so look for mdx2 objects in the entry group
                        for obj_name, obj in f["/entry"].items():
                            mdx2_object_info[obj_name] = {ak: av for ak, av in obj.attrs.items()}
                            mdx2_object_info[obj_name]["keys"] = list(obj.keys())
            except (OSError, KeyError) as e:
                warnings.warn(f"Error inspecting file {filename}: {e}", stacklevel=2)
            return file_info, mdx2_object_info

        for f in file_paths:
            file_info, mdx2_object_info = _inspect_nexus_file(f)
            if "mdx2_version" not in file_info:
                warnings.warn(
                    f"File {f} does not appear to be an mdx2 file (missing 'mdx2_version' attribute)",
                    stacklevel=2,
                )
            elif not mdx2_object_info:
                warnings.warn(
                    f"File {f} does not appear to contain any mdx2 objects in the /entry group",
                    stacklevel=2,
                )
            else:
                self._object_index[f] = mdx2_object_info

    def find_objects(self, mdx2_class: type):
        matches = []
        for file_name, object_info in self._object_index.items():
            for object_name, attrs in object_info.items():
                if attrs.get("mdx2_class") == mdx2_class.__name__ and attrs.get("mdx2_module") == mdx2_class.__module__:
                    matches.append((file_name, object_name))
        return matches
