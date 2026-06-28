import importlib.util
import os
import sys

_FINAL_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "final_models"
)
if _FINAL_MODELS_DIR not in sys.path:
    sys.path.append(_FINAL_MODELS_DIR)

_spec = importlib.util.spec_from_file_location(
    "_final_models_data_prep", os.path.join(_FINAL_MODELS_DIR, "data_prep.py")
)
_final_models_data_prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_final_models_data_prep)

load_and_prepare = _final_models_data_prep.load_and_prepare
