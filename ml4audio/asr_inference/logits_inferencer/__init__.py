from beartype.roar import BeartypeDecorHintPep585DeprecationWarning
from warnings import filterwarnings

filterwarnings("ignore", category=BeartypeDecorHintPep585DeprecationWarning)
from misc_utils import beartyped_dataclass_patch
