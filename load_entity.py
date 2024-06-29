import pickle
from pathlib import Path


ENTITY_MODEL_PATH = Path(__file__).parent / 'main_pipeline' / 'entity_model'

def _load_entity_model() -> None:
    """Load entity phrase matcher model."""
    with ENTITY_MODEL_PATH.open('rb') as f:
        return pickle.load(f)

my_model = _load_entity_model()
