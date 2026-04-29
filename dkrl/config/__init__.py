"""
Configuration subpackage for DKRL.

Re-exports constants, hardware settings, model defaults, and config loading.
"""

from dkrl.config._constants import (  # noqa: F401
    DT_MAX,
    EPSILON,
    MIN_SEGMENT_LENGTH,
    TROPHIC_LEVEL_MAP,
    UERE,
    VARIANCE_CEILING,
    VARIANCE_FLOOR,
    Z_68,
    Z_90,
    Z_95,
)
from dkrl.config.defaults import NIGPModelConfig  # noqa: F401
from dkrl.config.hardware import (  # noqa: F401
    get_data_dir,
)
from dkrl.config.loader import (  # noqa: F401
    config_to_model_config,
    load_config,
)
