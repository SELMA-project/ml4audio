from beartype.roar import BeartypeDecorHintPep585DeprecationWarning
from warnings import filterwarnings

filterwarnings("ignore", category=BeartypeDecorHintPep585DeprecationWarning)

from misc_utils.beartyped_dataclass_patch import (
    beartype_all_dataclasses_of_this_files_parent,
)

beartype_all_dataclasses_of_this_files_parent(__file__)
