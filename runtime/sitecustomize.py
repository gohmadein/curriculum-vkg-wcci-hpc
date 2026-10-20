"""Process-local dataset path override; no user configuration writes."""
import os
if os.environ.get("VKG_DATA_ROOT"):
    import grid2op.MakeEnv.PathUtils as _paths
    _paths.DEFAULT_PATH_DATA = os.environ["VKG_DATA_ROOT"]
