from transformers.models.auto import AutoConfig, AutoModel

from .configuration_higgs_audio_v2_tokenizer import HiggsAudioV2TokenizerConfig
from .modeling_higgs_audio_v2_tokenizer import HiggsAudioV2TokenizerModel, HiggsAudioV2TokenizerPreTrainedModel


# Newer transformers releases ship their own "higgs_audio_v2_tokenizer" config.
# Wan2GP instantiates these classes directly, so only register them with the
# Auto* factories when transformers doesn't already own the model type.
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    _already_registered = "higgs_audio_v2_tokenizer" in CONFIG_MAPPING
except Exception:
    _already_registered = False

if not _already_registered:
    AutoConfig.register("higgs_audio_v2_tokenizer", HiggsAudioV2TokenizerConfig)
    AutoModel.register(HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel)


__all__ = [
    "HiggsAudioV2TokenizerConfig",
    "HiggsAudioV2TokenizerModel",
    "HiggsAudioV2TokenizerPreTrainedModel",
]
