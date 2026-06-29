# An Example command to invoke this file with a "already compiled option."
# python /workspace/robust_kernelbench/robust_kernelbench/evaluate_single.py --new_path experiments/exp_v3/trial_1/code/66__sample_None0.py --original_path experiments/exp_v3/trial_1/kernel/66__sample__ref_kernel/66_conv_standard_3D__asymmetric_input__asymmetric_kernel.py --experiment exp_v3 --problem_id 66 --sample_id 0
# python /workspace/robust_kernelbench/robust_kernelbench/evaluate_single.py --original_path experiments/exp_v3/trial_1/kernel/88__sample__ref_kernel/88_MinGPTNewGelu.py --new_path experiments/exp_v3/trial_1/code/88__sample_None0.py --experiment exp_v3 --problem_id 88 --sample_id 0
# python /workspace/robust_kernelbench/robust_kernelbench/evaluate_single.py --original_path experiments/exp_v3/trial_1/kernel/98__sample__ref_kernel/98_KLDivLoss.py --new_path experiments/exp_v3/trial_1/code/98__sample_None0.py --experiment exp_v3 --problem_id 98 --sample_id 0 --build_dir_new experiments/exp_v3/trial_1/kernel/98__sample__solution_kernel

import os
import json
import io
import sys

from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.cpp_extension

import numpy as np

import gc

from datasets import load_dataset

from tqdm import tqdm

import pandas as pd

import shutil

from datetime import datetime

import traceback
import tempfile
from typing import Any


import multiprocessing as mp
from multiprocessing import Pool,Lock

from utils import (
    get_folder_path,
    get_file_name,
    get_problem_id_from_file_name
)

from utils_data import (
    CompileResult,
    KernelExecResult
)

from utils_evaluate_common import (
    register_and_format_exception,
    format_exception_string,
    OutputCapture
)

from utils_compile import (
    load_original_model_and_inputs,
    load_custom_model,
    # compile_and_get_output,
    # exec_and_get_output
)

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)  # Ensure spawn is used

# BASIC DEFINITIONS
level1_representative_subset_problem_ids = [1, 3, 6, 18, 23, 26, 33, 36, 40, 42, 48, 54, 57, 65, 77, 82, 86, 87]
level2_representative_subset_problem_ids = [1, 2, 8, 18, 23, 28, 33, 43]
level3_representative_subset_problem_ids = [1, 5, 8, 11, 20, 33, 38, 43]

CURRENT_DATETIME=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def read_file(file_path) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""
    
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def get_dataframe_results(results):
    """Dataframe Results"""
    # tmp = [y.__dict__ for y in results]

    out_df = pd.DataFrame.from_dict(results)

    return out_df


# This is from the "eval.py" file.
def set_seed(seed: int):
    torch.manual_seed(seed)
    # NOTE: this only sets on current cuda device
    torch.cuda.manual_seed(seed)


def time_execution_with_cuda_event(
    kernel_fn: callable,
    *args,
    num_warmup: int = 3,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
    ) -> list[float]:
    """
    Time a CUDA kernel function over multiple trials using torch.cuda.Event

    Args:
        kernel_fn: Function to time
        *args: Arguments to pass to kernel_fn
        num_trials: Number of timing trials to run
        verbose: Whether to print per-trial timing info
        device: CUDA device to use, if None, use current device

    Returns:
        List of elapsed times in milliseconds
    """
    if device is None:
        if verbose:
            print(f"Using current device: {torch.cuda.current_device()}")
        device = torch.cuda.current_device()

    # Warm ups
    for _ in range(num_warmup):
        kernel_fn(*args)
        torch.cuda.synchronize(device=device)

    print(
        f"[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, warm up {num_warmup}, trials {num_trials}"
    )
    elapsed_times = []

    # Actual trials
    for trial in range(num_trials):
        # create event marker default is not interprocess
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        kernel_fn(*args)
        end_event.record()

        # Synchronize to ensure the events have completed
        torch.cuda.synchronize(device=device)

        # Calculate the elapsed time in milliseconds
        elapsed_time_ms = start_event.elapsed_time(end_event)
        if verbose:
            print(f"Trial {trial + 1}: {elapsed_time_ms:.3g} ms")
        elapsed_times.append(elapsed_time_ms)

    return elapsed_times

def get_timing_stats(elapsed_times: list[float], device: torch.device = None) -> dict:
    """Get timing statistics from a list of elapsed times.

    Args:
        elapsed_times: List of elapsed times in milliseconds
        device: CUDA device, record device info
    Returns:
        Dict containing mean, std, min, max and num_trials
        all timing are in ms
    """

    stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }

    if device:
        stats["hardware"] = torch.cuda.get_device_name(device=device)
        stats["device"] = str(device)  # for debugging

    return stats


#Additional Functions (eval.py and others?)
def gpu_cache_clean(device: torch.device = torch.device("cuda:0")):
    """Cleans up the GPU cache."""
    # Clear CUDA cache and reset GPU state
    torch.cuda.empty_cache()
    # with torch.cuda.device(device):
    #     torch.cuda.empty_cache()

    #     # does this help?
    #     torch.cuda.reset_peak_memory_stats(device=device)

    #     torch.cuda.synchronize(
    #         device=device
    #     )  # Wait for all CUDA operations to complete

def memory_cleanup(curr_context: dict|Any=None, device: torch.device=None, custom_string: str="", verbose: bool=False):
    """
    Clean up env, gpu cache, and compiled CUDA extensions after evaluation
    """  # delete ran-specific function definitions before next eval run
    if verbose:
        print("\n======Start")
        if custom_string:
            print(custom_string)
        print("---before")
        print(f"Allocated memory: {torch.cuda.memory_allocated()}")
        print(f"Cached memory: {torch.cuda.memory_reserved()}")
    # print(curr_context)
    
    if curr_context:
        del curr_context
    gc.collect()
    torch.cuda.empty_cache()

    # gpu_cache_clean(device=device)
    if verbose:
        print("\n---after")
        print(f"Allocated memory: {torch.cuda.memory_allocated()}")
        print(f"Cached memory: {torch.cuda.memory_reserved()}")
        print("======End\n")



# TODO: edit this to include multiple new_model_instances...
# Clean compile folder...
def run_and_check_correctness(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    num_correct_trials: int,
    verbose=False,
    seed=42,
    device=None,
    ) -> KernelExecResult:
    """
    run the model and check correctness,
    assume model already loaded and compiled (loaded and compiled in the caller)
    this is all on GPU, requiring cuda device and transfer .cuda()

    num_correct_trials: run the evalutation multiple times with (ideally) different random inputs to ensure correctness
    """
    pass_count = 0
    exception_str = ""
    # Generate num_correct_trials seeds deterministically from the initial seed
    torch.manual_seed(seed)
    correctness_trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)
    ]

    with torch.no_grad():

        for trial in range(num_correct_trials):

            trial_seed = correctness_trial_seeds[trial]
            if verbose:
                print(f"[Eval] Generating Random Input with seed {trial_seed}")

            set_seed(trial_seed)
            inputs = get_inputs_fn()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]

            set_seed(trial_seed)
            model = original_model_instance.cuda(device=device)

            set_seed(trial_seed)
            model_new = new_model_instance.cuda(device=device)

            output = model(*inputs)
            torch.cuda.synchronize(device=device)
            # ensure all GPU operations are completed before checking results

            try:
                output_new = model_new(*inputs)
                torch.cuda.synchronize(device=device)
                if output.shape != output_new.shape:
                    exception_str = f"Output shape mismatch: Expected {output.shape}, got {output_new.shape}"
                    metadata, exception_str = register_and_format_exception(
                        "correctness_issue",
                        exception_str,
                        metadata,
                    )
                    if verbose:
                        print(
                            f"[FAIL] trial {trial}: Output shape mismatch: Expected {output.shape}, got {output_new.shape}"
                        )
                    kernel_exec_result = KernelExecResult(
                        compiled=True, 
                        correctness_success=True,
                        correctness=False, 
                        metadata=metadata, 
                        main_error=format_exception_string(exception_str),
                    )
                    return kernel_exec_result

                # check output value difference
                if not torch.allclose(
                    output, output_new, atol=1e-02, rtol=1e-02
                ):  # fail
                    max_diff = torch.max(torch.abs(output - output_new)).item()
                    avg_diff = torch.mean(torch.abs(output - output_new)).item()
                    metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
                    metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
                    exception_str = f"Output mismatch: max difference is: {max_diff:.6f}"
                    metadata["correctness_issue"] = "Output mismatch"
                    if verbose:
                        print(f"[FAIL] trial {trial}: Output mismatch")
                else:  # pass
                    pass_count += 1
                    if verbose:
                        print(f"[PASS] trial {trial}: New Model matches Model")

            except Exception as e:
                print("[Error] Exception happens during correctness check")
                print(f"Error in launching kernel for ModelNew: {e}")

                metadata, exception_str = register_and_format_exception(
                    "runtime_error", e, metadata, truncate=True
                )
                kernel_exec_result= KernelExecResult(
                    compiled=True,
                    correctness_success=False,
                    correctness=False, 
                    metadata=metadata, 
                    main_error=format_exception_string(exception_str),
                )
                return kernel_exec_result
                # break

    if verbose:
        print(
            f"[Eval] Pass count: {pass_count}, num_correct_trials: {num_correct_trials}"
        )

    # put all the useful info here!
    metadata["correctness_trials"] = f"({pass_count} / {num_correct_trials})"

    if pass_count == num_correct_trials:
        return KernelExecResult(
            compiled=True, 
            correctness_success=True,
            correctness=True, 
            metadata=metadata, 
            main_error=format_exception_string(exception_str),
        )
    else:
        return KernelExecResult(
            compiled=True, 
            correctness_success=True,
            correctness=False, 
            metadata=metadata, 
            main_error=format_exception_string(exception_str),
        )


# TODO: need to clean up build dirs...
def eval_kernel_against_ref(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    verbose: bool = False,
    measure_performance: bool = False,
    build_dir: os.PathLike = None,
    build_dir_new: os.PathLike = None,
    device: torch.device = None, # have to run on GPU
    ) -> KernelExecResult:
    """
    Evaluate the custom kernel against the original model

    num_correct_trials: number of trials to initialize different random inputs; correctness pass only if all trials pass
    num_perf_trials: run the evalutation many times to take the average
    device: GPU (cuda) device to run the evalutation on
    """
    # TODO: check device is busy
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"
    torch.set_printoptions(
        precision=4,  # Decimal places
        threshold=10,  # Total number of elements before truncating
        edgeitems=3,  # Number of elements at beginning and end of dimensions
        linewidth=80,  # Maximum width before wrapping
    )

    # set CUDA device
    torch.cuda.set_device(device)

    context = {}
    context_new = {}

    if verbose:
        print(f"[Eval] Start Evalulation! on device: {device}")
        print("[Eval] Loading Original Model")

    (Model, get_init_inputs, get_inputs), orig_compilation_status, (orig_std_compile, orig_std_runtime) = load_original_model_and_inputs(
        original_model_src, 
        context, 
        build_dir
    )
    set_seed(seed_num)  # set seed for reproducible input
    init_inputs = get_init_inputs()
    init_inputs = [
        x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
    ]

    with torch.no_grad():
        set_seed(seed_num)  # set seed for reproducible weights
        original_model = Model(*init_inputs)
        assert hasattr(original_model, "forward")
        if verbose:
            print("[Eval] Original Model Loaded")


    # Now we can compile new model.
    # torch.cuda.synchronize(device=device)  # This is needed as we otherwise get stuck on the next step... ?
    # print("[Eval] Sync done....")
    
    if verbose:
        print("[Eval] Loading and Compiling New Model with Custom CUDA Kernel")

    try:
        metadata = {}  # for storing result metadata
        # metadata["hardware"] = torch.cuda.get_device_name(device=device)
        metadata["device"] = str(device)  # for debugging
        metadata["execution_status"] = None
    except Exception as e:
        print("[Eval] Failed with basics....")
        print(e)

    # this is where compilation happens
    compilation_output = False
    try:
        
        # os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        custom_model_src = (
            "import os\n" f"os.environ['TORCH_USE_CUDA_DSA'] = '1'\n"
        ) + custom_model_src
        print("[Eval] running load custom model...")
        # add hash for later to distinguish between multi-turn kernels
        ModelNew, compilation_status, (std_compile, std_runtime), (comp_o, exe_o) = load_custom_model(
            custom_model_src, 
            context, 
            build_dir_new,
            device
        )
        compilation_output = True
        print("[Eval] Loading Custom Model returned something")
        metadata["compilation_status"] = compilation_status
        
        try:
            torch.cuda.synchronize(device=device)  # This will cause CUDA errors
            print("[Eval] CUDA synchronize (Model compilation) successful...")
        except torch.AcceleratorError as e:
            print("[Eval] CUDA synchronize (Model compilation) failed.")
            metadata["compilation_error"] = e
            metadata["compilation_output"] = std_compile

            tmp_traceback = traceback.format_exc() #traceback.print_exc()
            metadata["compilation_traceback"] = tmp_traceback

            return KernelExecResult(
                cuda_success=False,
                compiled=True, 
                runtime_success=True, 
                metadata=metadata, 
                main_output=std_compile,
                main_traceback=tmp_traceback if tmp_traceback else "",

            )  # skip further steps
    except Exception as e:
        print(
            f"[Eval] ERROR - Failed to compile custom CUDA kernel: Record as compilation or runtime failure."
        )
        if not compilation_output:
            print("[Eval] No compilation output. Therefore getting it from the exception.")
            try:
                _, compilation_status, (std_compile, std_runtime), (comp_o, exe_o) = e.kernel_output_added
            except:
                # hopefully this means that load_custom_model returned a useful output
                pass

        if "lock" in str(e) or "No such file or directory" in str(e):
            # this is a lock file error, likely due to concurrent compilation
            # this does not necessarily mean the compilation failed, but we should retry
            print(
                f"[Eval] Lock file error during compilation, Please retry. Error: {e}"
            )
            memory_cleanup(context, device)
            memory_cleanup(context_new, device)
            raise e

        else:
            if compilation_status=="compilation_error":
                print("---")
                metadata["compilation_error"] = e
                metadata["compilation_output"] = std_compile

                tmp_traceback = traceback.format_exc() #traceback.print_exc()
                metadata["compilation_traceback"] = tmp_traceback

                memory_cleanup(context, device)
                memory_cleanup(context_new, device)

                return KernelExecResult(
                    compiled=False, 
                    runtime_success=False, 
                    metadata=metadata, 
                    main_output=std_compile,
                    main_traceback=tmp_traceback if tmp_traceback else "",

                )  # skip further steps
            elif compilation_status=="runtime_error":
                print("---2")
                metadata["runtime_error"] = e
                metadata["runtime_output"] = std_runtime
                metadata["compilation_status"] = compilation_status

                tmp_traceback = traceback.format_exc() #traceback.print_exc()
                metadata["runtime_traceback"] = tmp_traceback


                memory_cleanup(context, device)
                memory_cleanup(context_new, device)

                return KernelExecResult(
                    compiled=True, 
                    runtime_success=False, 
                    metadata=metadata, 
                    main_output=std_runtime, 
                    main_error=format_exception_string(e),
                    main_traceback=tmp_traceback if tmp_traceback else "",
                )  # skip further steps
            else:
                memory_cleanup(context, device)
                memory_cleanup(context_new, device)

                print(f"[STRANGE] You are having success in compilation / runtime, but still an exception.\n---:\n{e}")
                raise e
                
    print("[EVAL] Compilation was fully successful.")
    # at this point we passed compilation
    exec_std_output=""
    try:
        with torch.no_grad():
            set_seed(seed_num)  # set seed for reproducible weights
            with OutputCapture() as capture:
                custom_model = ModelNew(*init_inputs)
                try:
                    exec_std_output = capture.get_output()
                except Exception as e:
                    print("[Strange] capture.get_output() failed")
                    print(e)
                    exec_std_output = ""
            assert hasattr(custom_model, "forward")
            try:
                torch.cuda.synchronize(device=device)  # This will cause CUDA errors
                print("[Eval] CUDA synchronize (Model loading) successful...")
            except torch.AcceleratorError as e:
                print("[Eval] CUDA synchronize (Model loading) failed...")
                metadata["execution_error"] = e
                metadata["execution_output"] = exec_std_output
                metadata["execution_status"] = "error_model_loading"

                tmp_traceback = traceback.format_exc() #traceback.print_exc()
                metadata["execution_traceback"] = tmp_traceback

                return KernelExecResult(
                    cuda_success=False,
                    compiled=True, 
                    runtime_success=True, 
                    loading_success=True, #setting this as true... as the code worked... 
                    metadata=metadata, 
                    main_output=std_compile,
                    main_traceback=tmp_traceback if tmp_traceback else "",

                )  # skip further steps

            print("[Eval] New Model with Custom CUDA Kernel Loaded")
    except RuntimeError as e:
        print(
            f"[Eval] Failed to load custom CUDA kernel; Compiled but not able to run, count as execution error. \nError: {e}"
        )
        # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        memory_cleanup(context, device)
        memory_cleanup(context_new, device)

        metadata["execution_error"] = e
        metadata["execution_output"] = exec_std_output
        metadata["execution_status"] = "error_model_loading"

        tmp_traceback = traceback.format_exc() #traceback.print_exc()
        metadata["execution_traceback"] = tmp_traceback

        return KernelExecResult(
            compiled=True, 
            runtime_success=True, 
            loading_success=False, 
            correctness_success=False,
            timing_success=False,
            correctness=False, 
            metadata=metadata,
            main_output=exec_std_output,
            main_traceback=tmp_traceback if tmp_traceback else "",
            main_error=format_exception_string(e),
        )  # skip further steps
    except torch.AcceleratorError as e:
        print(
            f"[Eval] Failed to load custom CUDA kernel; Compiled but not able to run, count as execution error. \nError: {e}"
        )
        # No clean-up as it further causes errors...

        metadata["execution_error"] = e
        metadata["execution_output"] = exec_std_output
        metadata["execution_status"] = "error_model_loading"

        tmp_traceback = traceback.format_exc() #traceback.print_exc()
        metadata["execution_traceback"] = tmp_traceback

        return KernelExecResult(
            cuda_success=False,
            compiled=True, 
            runtime_success=True, 
            loading_success=False, 
            correctness_success=False,
            timing_success=False,
            correctness=False, 
            metadata=metadata,
            main_output=exec_std_output,
            main_traceback=tmp_traceback if tmp_traceback else "",
            main_error=format_exception_string(e),
        )  # skip further steps
    except Exception as e:
        print("[STRANGE] - A strange error during ModelNew init().")
        print(init_inputs)
        print(e)

        metadata["execution_error"] = e
        metadata["execution_output"] = exec_std_output
        metadata["execution_status"] = "error_model_loading_2"
        tmp_traceback = traceback.format_exc() # traceback.print_exc()
        metadata["execution_traceback"] = tmp_traceback

        memory_cleanup(context, device)
        memory_cleanup(context_new, device)

        return KernelExecResult(
            compiled=True, 
            runtime_success=True, 
            loading_success=False, 
            correctness_success=False,
            timing_success=False,
            correctness=False, 
            metadata=metadata,
            main_output=exec_std_output,
            main_traceback=tmp_traceback if tmp_traceback else "",
            main_error=format_exception_string(e),
        )  # skip further steps

    kernel_exec_result = None

    # Check Correctness
    if verbose:
        print("[Eval] Checking Correctness")
    try:
        kernel_exec_result = run_and_check_correctness(
            original_model,
            custom_model,
            get_inputs,
            metadata=metadata,
            num_correct_trials=num_correct_trials,
            verbose=verbose,
            seed=seed_num,
            device=device,
        )
        kernel_exec_result.runtime_success=True
        kernel_exec_result.loading_success=True


    except torch.AcceleratorError as e:
        print(f"[Eval] Failed during run_and_check_correctness - function.... (TorchAcceleratorError){e}")
        # TODO: (maybe) add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        metadata["correctness_execution_error"] = e
        # metadata["execution_output"] = exec_std_output #TODO: this one here...
        metadata["correctness_execution_status"] = "error_correctness"
        tmp_traceback = traceback.format_exc() # traceback.print_exc()
        metadata["correctness_execution_traceback"] = tmp_traceback

        # memory_cleanup(context, device)
        # memory_cleanup(context_new, device)

        return KernelExecResult(
            cuda_success=False,
            compiled=True, 
            runtime_success=True, 
            loading_success=True, 
            correctness_success=False,
            timing_success=False,
            correctness=False, 
            metadata=metadata,
            main_output=format_exception_string(e),
            main_traceback = tmp_traceback if tmp_traceback else "",
            main_error=format_exception_string(e),
        )  # skip further steps      
    except Exception as e:
        print(f"[Eval] Failed during run_and_check_correctness - function.... (Other Exception){e}")
        # TODO: (maybe) add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        metadata["correctness_execution_error"] = e
        # metadata["execution_output"] = exec_std_output #TODO: this one here...
        metadata["correctness_execution_status"] = "error_correctness"
        tmp_traceback = traceback.format_exc() # traceback.print_exc()
        metadata["correctness_execution_traceback"] = tmp_traceback

        memory_cleanup(context, device)
        memory_cleanup(context_new, device)

        return KernelExecResult(
            compiled=True, 
            runtime_success=True, 
            loading_success=True, 
            correctness_success=False,
            timing_success=False,
            correctness=False, 
            metadata=metadata,
            main_output=format_exception_string(e),
            main_traceback = tmp_traceback if tmp_traceback else "",
            main_error=format_exception_string(e),
        )  # skip further steps
    
    custom_timing_success=False
    # Measure Performance [Optional] | conditioned on compilation + correctness + no exception so far
    if measure_performance:
        try:
            if kernel_exec_result and kernel_exec_result.correctness:
                if verbose:
                    print("[Eval] Measuring Performance as Sample is Correct")

                torch.cuda.synchronize(device=device)
                set_seed(seed_num)
                inputs = get_inputs()
                inputs = [
                    x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                ]
                model_new = custom_model.cuda(device=device)
                torch.cuda.synchronize(device=device)

                elapsed_times = time_execution_with_cuda_event(
                    model_new,
                    *inputs,
                    num_trials=num_perf_trials,
                    verbose=verbose,
                    device=device,
                )
                runtime_stats = get_timing_stats(elapsed_times, device=device)

                if verbose:
                    print(f"[Eval] Performance Stats: {runtime_stats}")
                kernel_exec_result.runtime = runtime_stats["mean"]
                kernel_exec_result.runtime_stats = runtime_stats

                kernel_exec_result.timing_success = True
                custom_timing_success=True
        except torch.AcceleratorError as e:
            if verbose:
                print(f"[Eval] Error in Measuring Performance (TorchAccelerator): {e}")
            kernel_exec_result.metadata["error_during_performance"] = e
            kernel_exec_result.metadata["execution_status"] = "error_timing"
            kernel_exec_result.main_output=format_exception_string(e)
            tmp_traceback = traceback.format_exc() #traceback.print_exc()
            kernel_exec_result.metadata["execution_traceback"] = tmp_traceback
            kernel_exec_result.main_traceback = tmp_traceback if tmp_traceback else ""
            kernel_exec_result.main_error=format_exception_string(e)
            kernel_exec_result.cuda_success=False #this is the line we add...

            custom_timing_success = False

        except Exception as e:
            if verbose:
                print(f"[Eval] Error in Measuring Performance: {e}")
            kernel_exec_result.metadata["error_during_performance"] = e
            kernel_exec_result.metadata["execution_status"] = "error_timing"
            kernel_exec_result.main_output=format_exception_string(e)
            tmp_traceback = traceback.format_exc() #traceback.print_exc()
            kernel_exec_result.metadata["execution_traceback"] = tmp_traceback
            kernel_exec_result.main_traceback = tmp_traceback if tmp_traceback else ""
            kernel_exec_result.main_error=format_exception_string(e)

            custom_timing_success = False

    # Measure Performance [Optional] | conditioned on compilation + correctness + no exception so far
    if measure_performance:
        try:
            if kernel_exec_result and kernel_exec_result.correctness and custom_timing_success:
                if verbose:
                    print("[Eval] Measuring Performance of Original.")

                torch.cuda.synchronize(device=device)
                set_seed(seed_num)
                inputs = get_inputs()
                inputs = [
                    x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                ]
                model_old = original_model.cuda(device=device)
                torch.cuda.synchronize(device=device)

                elapsed_times_original = time_execution_with_cuda_event(
                    model_old,
                    *inputs,
                    num_trials=num_perf_trials,
                    verbose=verbose,
                    device=device,
                )
                runtime_original_stats = get_timing_stats(elapsed_times_original, device=device)

                if verbose:
                    print(f"[Eval] Performance Stats: {runtime_original_stats}")
                kernel_exec_result.runtime_original = runtime_original_stats["mean"]
                kernel_exec_result.runtime_original_stats = runtime_original_stats
        except Exception as e:
            if verbose:
                print(f"[Eval] Error in Measuring Performance: {e}")
            kernel_exec_result.metadata["error_during_performance_of_original"] = e
            kernel_exec_result.metadata["execution_status"] = "error_timing_original"

    if custom_timing_success:
        if  kernel_exec_result.metadata.get("execution_status"):
            if not kernel_exec_result.metadata.get("execution_status") == "error_timing_original":
                kernel_exec_result.metadata["execution_status"] = "success"

    memory_cleanup(context, device)
    memory_cleanup(context_new, device)

    return kernel_exec_result


# Create a wrapper function for multiprocessing
def eval_wrapper(reference_src, custom_src, build_dir, build_dir_new, device):
    return eval_kernel_against_ref(
        original_model_src = reference_src,
        custom_model_src = custom_src,
        seed_num = 42,
        num_correct_trials = 3,
        num_perf_trials = 3,
        verbose  = True,
        measure_performance = True,
        build_dir = build_dir,
        build_dir_new = build_dir_new,
        device = device, # have to run on GPU
    )

def evaluate_single(
    original_path, 
    new_path, 
    build_dir=None, 
    build_dir_new=None, 
    # other
    seed_num = 42,
    num_correct_trials = 3,
    num_perf_trials = 3,
    verbose  = True,
    measure_performance = True,
    device = torch.cuda.current_device() if torch.cuda.is_available() else None, # have to run on GPU
    # Admin:
    experiment_folder = None,
    level=1,
    trial=1,
    problem_id=None,
    sample_id=0,
    ):
    print(f"[Eval Main Func]\nbuild_dir_new:\t{build_dir_new}\nnew_path:\t{new_path}.")


    if build_dir:
        Path(build_dir).mkdir(parents=True, exist_ok=True)

    if build_dir_new:
        Path(build_dir_new).mkdir(parents=True, exist_ok=True)

    reference_src = read_file(original_path)
    custom_src = read_file(new_path)
    try:

        print(f"\n---:\n[Eval Main Func] Running Actual Evaluation now for l:{level}, t: {trial}, p:{problem_id}, s:{sample_id}")
        # print(f"[Eval Main Func] This is the cache location: {torch.utils.cpp_extension._get_build_directory(new_path, verbose=True)}")

        start = datetime.now()
        # results = eval_wrapper(reference_src, custom_src, build_dir, build_dir_new)

        # Why does it get stuck on this....?
        # DEBUG
        # exec(custom_src, {})

        # Run on this device
        results = eval_wrapper(reference_src, custom_src, build_dir, build_dir_new, device,)

        # # Run in a separate process
        # with Pool(processes=1) as pool:
        #     results = pool.apply(eval_wrapper, args=(reference_src, custom_src, build_dir, build_dir_new, device,))
        
        end = datetime.now()
        print(f"[Eval Main Func] Got Results. Took time: {end-start}")
        
        try:
            output = results.__dict__
            output["problem_id"] = problem_id
            output["sample_id"] = sample_id
            out_results= [output]       

        except Exception as e:
            print(f"[Eval Main Func] Error PROBLEM EVAL t:{trial} p:{problem_id} s:{sample_id}]We couldn't get a results\n----\n")
            print(e)
            output = KernelExecResult().__dict__
            output["problem_id"] = problem_id
            output["sample_id"] = sample_id
            out_results= [output]    

    except Exception as e:
        print(f"[Eval Main Func] Error PROBLEM EVAL t:{trial} p:{problem_id} s:{sample_id}]FULL FAIL\n====\n")
        print(e)
        output = KernelExecResult().__dict__
        output["problem_id"] = problem_id
        output["sample_id"] = sample_id
        out_results= [output]  

    try:
        folder_path = get_folder_path(experiment_folder, trial=trial)
        eval_filename = os.path.join(folder_path,'evaluations.csv')

        results_df = get_dataframe_results(out_results)

        if os.path.exists(eval_filename):
            print(f"[Eval Main Func] Writing Results file {eval_filename}. Appending.")
            results_df.to_csv(os.path.join(eval_filename), index=False, header=False, mode="a")

        else:
            print(f"[Eval Main Func] Writing Results file {eval_filename}. New.")
            results_df.to_csv(os.path.join(eval_filename), index=False)
        print("[Eval Main Func] Writing to file done, successfully. ")

    except Exception as e:
        print("[Eval Main Func] Error with writing to file.")
        print(f"[Eval Main Func] {eval_filename}")
        print(f"Exception:\n{e}")


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1, help="Num of Parallel responses")
    parser.add_argument("--prompt_type", type=str, default="normal", help="Which prompt to use...")
    parser.add_argument("--experiment", type=str, default="exp_test_run", help="experiment names")
    parser.add_argument("--level", type=int, default=1, help="Which level to run...")

    parser.add_argument("--original_path", type=str, required=True, help="Path to original model source")
    parser.add_argument("--new_path", type=str, required=True, help="Path to new model source")
    parser.add_argument("--build_dir", type=str, default=None, help="Build directory for original model")
    parser.add_argument("--build_dir_new", type=str, default=None, help="Build directory for new model")
    parser.add_argument("--seed_num", type=int, default=42, help="Random seed")
    parser.add_argument("--num_correct_trials", type=int, default=3, help="Number of correctness trials")
    parser.add_argument("--num_perf_trials", type=int, default=3, help="Number of performance trials")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    # parser.add_argument("--measure_performance", action="store_true", default=True, help="Measure performance")
    parser.add_argument("--problem_id", type=int, default=None, help="Problem ID")
    parser.add_argument("--sample_id", type=int, default=0, help="Sample ID")
    parser.add_argument("--trial", type=int, default=1, help="Which trial you are currently running...")

    # TODO: add lora support

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()
    print("[Eval Main] - Entered Evaluate single...")
    evaluate_single(
        original_path=args.original_path,
        new_path=args.new_path,
        build_dir=args.build_dir,
        build_dir_new=args.build_dir_new,
        seed_num=args.seed_num,
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        verbose=args.verbose,
        measure_performance=True,
        experiment_folder=args.experiment,
        level=args.level,
        trial=args.trial,
        problem_id=args.problem_id,
        sample_id=args.sample_id,
    )
    print("[Eval Main] - Exiting...")
  

# Example Bash command:
# orginal_path = experiments/exp_local_v3/trial_1/kernel/9__sample__ref_kernel/9_Tall_skinny_matrix_multiplication_.py
# new_path = experiments/exp_local_v3/trial_1/code/9__sample_None0.py

# python3 robust_kernelbench/evaluate_single.py \
# --original_path "experiments/exp_local_v3/trial_1/kernel/98__sample__ref_kernel/98_KLDivLoss.py" \
# --new_path experiments/exp_local_v3/trial_1/code/98__sample_None0.py \
# --experiment exp_local_v3

"""
# EXAMPLE USAGE:

python3 robust_kernelbench/evaluate_single.py \
--original_path "experiments/exp_slurm_v4_6_tts_Qwen2_5_Coder_7B_Instruct_20260107_150105/trial_2/kernel/39__sample__ref_kernel/39_L2Norm_.py" \
--new_path experiments/exp_local_v3/trial_1/code/39__sample_0.py \
--experiment exp_slurm_v4_6_tts_Qwen2_5_Coder_7B_Instruct_20260107_150105
"""