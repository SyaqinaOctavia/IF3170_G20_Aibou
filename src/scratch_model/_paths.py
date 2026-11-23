import re
from pathlib import Path

SAVED_MODELS_DIR = Path(__file__).resolve().parents[2] / "output" / "saved_models"

def resolve_model_path(filename, name, load=False): #nama file bisa default atau tulis sendiri
    if filename is not None:
        return SAVED_MODELS_DIR / f"{filename}.json"
    pattern = re.compile(rf"^{re.escape(name)}_(\d+)\.json$")
    numbers = [int(m.group(1)) for f in SAVED_MODELS_DIR.glob(f"{name}_*.json") if (m := pattern.match(f.name))]
    if load:
        if not numbers:
            raise FileNotFoundError(f"Belum ada model '{name}' di {SAVED_MODELS_DIR}")
        return SAVED_MODELS_DIR / f"{name}_{max(numbers)}.json"
    return SAVED_MODELS_DIR / f"{name}_{max(numbers, default=0) + 1}.json"
