"""Analysis-side path helpers.

This module used to carry its own copy of ``get_folder_path`` (identical to the one in
``robust_kernelbench/utils/utils.py``, and already drifting).  It now simply re-exports the
shared implementation so there is a single definition of the experiments layout.

Existing analysis scripts no longer import through this module; import from
:mod:`robust_kernelbench.utils.utils` directly.
"""

from robust_kernelbench.utils.utils import (  # noqa: F401
    get_file_name,
    get_folder_path,
    get_problem_id_from_file_name,
    read_file,
    write_code,
    construct_generation_dict,
)

__all__ = [
    "get_folder_path",
    "get_file_name",
    "get_problem_id_from_file_name",
    "read_file",
    "write_code",
    "construct_generation_dict",
]
