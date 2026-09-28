__version__ = "1.2.2"

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm.modules.mamba_simple import Mamba

# Language model is optional (requires transformers)
try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except ImportError:
    # transformers not available - language model features disabled
    MambaLMHeadModel = None
