# add parent directory to sys.path
import sys
sys.path.append('.')
import importlib
import logging
import torch


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


# =  =  =  =  =  =  =  =  =  =  =  Model Registry  =  =  =  =  =  =  =  =  =  =  =  =  =
# Maps a public model name (the value passed via --model_name) to the module under
# model_src/ that implements it. Each module must expose two functions following the
# naming convention:
#     <module>_model_loader(self)            -> set up self.model / processor / etc.
#     <module>_model_generation(self, input) -> return the prediction(s) for one input
#
# To add a new model: create model_src/<module>.py with those two functions and add
# a single line below. The module is imported lazily (only when its model is used),
# so heavy / optional dependencies are not loaded for other models.
MODEL_REGISTRY = {
    "cascade_whisper_large_v3_llama_3_8b_instruct":              "whisper_large_v3_with_llama_3_8b_instruct",
    "cascade_whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct": "whisper_large_v2_gemma2_9b_cpt_sea_lionv3_instruct",
    "Qwen2-Audio-7B-Instruct":                                  "qwen2_audio_7b_instruct",
    "SALMONN_7B":                                               "salmonn_7b",
    "WavLLM_fairseq":                                           "wavllm_fairseq",
    "Qwen-Audio-Chat":                                          "qwen_audio_chat",
    "MERaLiON-AudioLLM-Whisper-SEA-LION":                       "meralion_audiollm_whisper_sea_lion",
    "gemini-1.5-flash":                                         "gemini_1_5_flash",
    "gemini-2-flash":                                           "gemini_2_flash",
    "whisper_large_v3":                                         "whisper_large_v3",
    "whisper_large_v2":                                         "whisper_large_v2",
    "gpt-4o-audio":                                             "gpt_4o_audio",
    "phi_4_multimodal_instruct":                                "phi_4_multimodal_instruct",
    "seallms_audio_7b":                                         "seallms_audio_7b",
    "oruk_fourier_2b":                                          "oruk_fourier_2b",
}


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
class Model(object):

    def __init__(self, model_name_or_path):

        self.dataset_name = None
        self.model_name   = model_name_or_path
        self.device       = "cuda" if torch.cuda.is_available() else "cpu"

        self.load_model()
        logger.info("Loaded model: {}".format(self.model_name))
        logger.info("= = "*20)


    def _resolve(self, suffix):
        """Return the model_src function named ``<module>_model_<suffix>`` for the
        current model. ``suffix`` is either 'loader' or 'generation'."""
        if self.model_name not in MODEL_REGISTRY:
            raise NotImplementedError("Model {} not implemented yet".format(self.model_name))
        module_name = MODEL_REGISTRY[self.model_name]
        module = importlib.import_module("model_src.{}".format(module_name))
        return getattr(module, "{}_model_{}".format(module_name, suffix))


    def load_model(self):
        self._resolve("loader")(self)


    def generate(self, input):
        with torch.no_grad():
            return self._resolve("generation")(self, input)
