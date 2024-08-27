import copy
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.generation.configuration_utils import GenerationConfig
    from transformers.generation.logits_process import LogitsProcessorList