"""Filesystem layout helpers shared by the whole pipeline.

Everything here is deliberately dependency-free (stdlib only) so that the very
light-weight modules (and the tests) can import it without pulling in torch.
"""

from __future__ import annotations

import os
from pathlib import Path

VERSION = "0.0.2"


def get_folder_path(experiment_name, trial=None, postfix=None, prefix="experiments"):
    """Creates the folder and returns path"""
    if trial:
        trial_name = f"trial_{trial}"
        folder_path = os.path.join(experiment_name, trial_name)
    else:
        folder_path = os.path.join(experiment_name)

    if postfix:
        folder_path = os.path.join(folder_path, postfix)

    if prefix:
        folder_path = os.path.join(prefix, folder_path)

    Path(folder_path).mkdir(parents=True, exist_ok=True)

    return folder_path


def get_file_name(problem_id, sample_id=None, suffix=None):
    """Creates the filename"""
    if sample_id:
        file_name = f"{problem_id}__sample_{sample_id}"
    else:
        file_name = f"{problem_id}__sample_"

    if suffix and type(suffix) == str:
        file_name = file_name + "_" + suffix

    return file_name


def get_problem_id_from_file_name(file_name):
    """Returns problem_id"""
    problem_id = int(file_name.split("__")[0])
    return problem_id


def read_file(file_path) -> str:
    """Read a text file, returning "" (and logging a warning) when it is missing.

    Historically this lived in ``evaluate_single`` and was imported *from there* by the
    compile drivers, which dragged torch into modules that only needed a file read.
    """
    if not os.path.exists(file_path):
        import logging

        logging.getLogger(__name__).warning("File %s does not exist", file_path)
        return ""

    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:  # pragma: no cover - io failure path
        import logging

        logging.getLogger(__name__).warning("Error reading file %s: %s", file_path, e)
        return ""


def write_code(fullpath, code_src):
    """Write source code to ``fullpath``."""
    with open(fullpath, "w") as file:
        file.write(code_src)


def find_ending_digit_and_string(name):
    """Extracts the name when it is: name_9812"""
    out_digit = []
    out_name = ""
    idx = 0
    for idx, c in enumerate(name[::-1]):
        if c in "0123456789":
            out_digit.insert(0, c)
        else:
            break
    out_name = name[:-idx]

    return out_name, "".join(out_digit)


def construct_generation_dict(folder):
    """
    This function assumes that inside 'folder' there will be files named according to
    the KernelBench problem names, e.g. ``1_Square_matrix_multiplication_0.py``.

    It returns::

        {'1_Square_matrix_multiplication_': ['1_Square_matrix_multiplication_0.py', ...]}

    Note: this lived in ``evaluate_all`` and was imported by the compile drivers, which
    created an import cycle (compile -> evaluate_all -> evaluate_single).  It is pure
    filesystem logic, so it now lives here.
    """
    files = sorted(os.listdir(folder))

    out_dict = {}

    for file in files:
        if file.endswith(".py"):
            file_basename = file.split(".py")[0]
            file_basename_root, file_digit = find_ending_digit_and_string(file_basename)

            if out_dict.get(file_basename_root):
                out_dict[file_basename_root].append(file)
            else:
                out_dict[file_basename_root] = [file]

    return out_dict
