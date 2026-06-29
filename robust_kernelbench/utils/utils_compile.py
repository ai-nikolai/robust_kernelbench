import traceback
import tempfile
from typing import Any

import torch
import torch.nn as nn

from .utils_data import (
    CompileResult,
    KernelExecResult
)

from .utils_evaluate_common import (
    exec_and_get_output,
    compile_and_get_output,
    format_exception_string
)

def load_original_model_and_inputs(
    model_original_src: str, context: dict, build_dir: str,
    ): #-> tuple[nn.Module, callable, callable]:
    """
    Load class from original NN.module pytorch code
    this is pytorch reference and we feed that to model to see if there will be any improvement
    """
    if build_dir:
        context["BUILD_DIRECTORY"] = build_dir
        # Add import at the start of the source code
        model_original_src = (
            "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_dir}'\n"
        ) + model_original_src

    try:
        comp_out, output, potential_exception = compile_and_get_output(model_original_src, "<string>", "exec")
        if potential_exception:
            raise(potential_exception)
    except SyntaxError as e:
        print(f"Syntax Error in original code {e}")
        return None, "compilation_error", (output, None)

    try:
        exec_out, output2, potential_exception = exec_and_get_output(model_original_src, context)  # expose to current namespace
        if potential_exception:
            raise(potential_exception)
    except Exception as e:
        print(f"Error in executing original code {e}")
        return None, "runtime_error", (output, output2)

    # these should be defined in the original model code and present in the context
    get_init_inputs_fn = context.get("get_init_inputs")
    get_inputs_fn = context.get("get_inputs")
    Model = context.get("Model")
    return (Model, get_init_inputs_fn, get_inputs_fn), "success", (output, output2)

def load_custom_model(
    model_custom_src: str, context: dict, build_directory: str = None, device = None
    ):# -> nn.Module:
    """
    Load class from custom NN.module pytorch code
    this is the code output by LLM with calls to custom cuda kernels
    """
    print("[Compilation] start of load custom model...")
    comp_out = None
    exec_out = None
    if build_directory:
        context["BUILD_DIRECTORY"] = build_directory
        # Add import at the start of the source code
        model_custom_src = (
            "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
        ) + model_custom_src

    try:
        print(f"[Compilation] - Attempting Compilation")
        comp_out, output, potential_exception = compile_and_get_output(model_custom_src, "<string>", "exec")
        if potential_exception:
            potential_exception.kernel_output_added = (None, "pre_compilation_error", (output, None), (comp_out, exec_out))
            print(f"[Compilation Error] Some error occurred.")
            raise(potential_exception)
        else:
            print(f"[Compilation Success] - pot_exc: {potential_exception} &output\n---\n{output}")
        # DANGER: need to delete refernece from global namespace
    except SyntaxError as e:
        print(f"[Syntax Error] in custom generated code or Compilation Error")
        e.kernel_output_added = (None, "pre_compilation_error", (output, None), (comp_out, exec_out))
        raise e
    except Exception as e:
        print(f"[Other Error] in custom generated code or Compilation Error")
        e.kernel_output_added = (None, "pre_compilation_error", (output, None), (comp_out, exec_out))
        raise e    

        # return None, "compilation_error", (output, None), (comp_out, exec_out)
    # try:
    #     torch.cuda.synchronize(device=device)  # This is needed, as otherwise or something like that... as otherwise the process seems to be hanging...
    # except Exception as e:
    #     e.kernel_output_added =(None, "pre_compilation_error", (output, None), (comp_out, exec_out))
    
    try:
        # print(f"[Exec] - Attempting Execution: {context}\n Source:\n---\n{model_custom_src}\n---\n")
        print(f"[Exec] - Attempting Execution:\n---\n")

        exec_out, output2, potential_exception = exec_and_get_output(model_custom_src, context)
        if potential_exception:
            print(f"[Exec Error] Some error occurred. exception: {potential_exception}. Output:\n---\n{output2}")
            potential_exception.kernel_output_added = (None, "runtime_error", (output, output2), (comp_out, exec_out))
            raise(potential_exception)
        else:
            print(f"[Exec Success] exception: {potential_exception}. Output:\n---\n{output2}")
        # DANGER: need to delete refernece from global namespace
    except SyntaxError as e:
        print(f"[Syntax Error 2] in custom generated code or Compilation Error")
        e.kernel_output_added = (None, "runtime_error", (output, output2), (comp_out, exec_out))
        raise(e)
    except RuntimeError as e:
        if "Error building extension" in str(e):
            print(f"[Runtime Error 2] in custom generated code or Compilation Error")
            e.kernel_output_added = (None, "compilation_error", (output, output2), (comp_out, exec_out))
            raise(e)   
        else:
            print(f"[Runtime Error 2] in custom generated code or Compilation Error")
            e.kernel_output_added = (None, "runtime_error", (output, output2), (comp_out, exec_out))
            raise(e)

    except Exception as e:
        print(f"[Other Error 2] in custom generated code or Compilation Error")
        e.kernel_output_added = (None, "runtime_error", (output, output2), (comp_out, exec_out))
        raise(e)
        # return None, "runtime_error", (output, output2), (comp_out, exec_out)
    
    try:
        print("[Eval] Trying to get ModelNew")
        ModelNew = context.get("ModelNew")
    except Exception as e:
        print("[Exec Error 2] Getting ModelNew failed.")
        e.kernel_output_added = (None, "model_new_not_found_error", (output, output2), (comp_out, exec_out))
        raise e

    return ModelNew, "success", (output, output2), (comp_out, exec_out)


def run_compilation(device, custom_model_src, context, build_dir_new,):
    """Function to just run compilation"""
    metadata = {}  # for storing result metadata
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)  # for debugging
    metadata["execution_status"] = None

    # this is where compilation happens
    compilation_output = False

    if not custom_model_src:
        error_message_formatting = "[Formatting Error] Code was empty. Make sure to produce valid code, ideally wrap it inside '```python' and '```'."
        return CompileResult(
            format_passed=False,
            metadata=metadata, 
            main_output=error_message_formatting,
            main_output_runtime=error_message_formatting,
            main_traceback=error_message_formatting,
        )  # skip further steps
    try:
        
        # os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        custom_model_src = (
            "import os\n" f"os.environ['TORCH_USE_CUDA_DSA'] = '1'\n"
        ) + custom_model_src
        # add hash for later to distinguish between multi-turn kernels
        ModelNew, compilation_status, (std_compile, std_runtime), (comp_o, exe_o)  = load_custom_model(
            custom_model_src, 
            context, 
            build_dir_new,
            device
        )
        compilation_output = True
        print("[COMPILE - utils_compile.run_compilation] Loading Custom Model returned something")
        metadata["compilation_status"] = compilation_status
        # Interesting this can cause an error during compile (I guess because some code executes the model already...)
        # torch.cuda.synchronize(device=device)  # This is needed, as otherwise or something like that... as otherwise the process seems to be hanging...
    except Exception as e:
        print(
            f"[COMPILE - utils_compile.run_compilation] ERROR - Failed to compile custom CUDA kernel: Record as compilation or runtime failure."
        )
        if not compilation_output:
            print("[COMPILE - utils_compile.run_compilation] No compilation output. Therefore getting it from the exception.")
            try:
                _, compilation_status, (std_compile, std_runtime), (comp_o, exe_o) = e.kernel_output_added
                print(f"[COMPILE - utils_compile.run_compilation] Got the following values: comp_status: {compilation_status}, std_compile:{std_compile}, std_runtime:{std_runtime}, comp_o:{comp_o}, exe_o:{exe_o}")
            except:
                print("[COMPILE - utils_compile.run_compilation] ERROR with getting .kernel_output_added")
                pass

        if "lock" in str(e) or "No such file or directory" in str(e):
            # this is a lock file error, likely due to concurrent compilation
            # this does not necessarily mean the compilation failed, but we should retry
            print(
                f"[COMPILE - utils_compile.run_compilation] Lock file error during compilation, Please retry. Error: {e}"
            )
            # memory_cleanup(context, device)
            raise e

        else:
            if compilation_status=="pre_compilation_error":
                print("[COMPILE - utils_compile.run_compilation] --- pre_compilation_error")
                metadata["compilation_error"] = e
                metadata["compilation_output"] = std_compile

                tmp_traceback = traceback.format_exc() #traceback.print_exc()
                metadata["compilation_traceback"] = tmp_traceback

                # memory_cleanup(context, device)
                return CompileResult(
                    pre_compiled=False,
                    compiled=False, 
                    runtime_success=False, 
                    metadata=metadata, 
                    main_output=std_compile if std_compile else "",
                    main_output_runtime=std_runtime if std_runtime else "",
                    main_traceback=tmp_traceback if tmp_traceback else "",
                )  # skip further steps
            elif compilation_status=="compilation_error":
                print("[COMPILE - utils_compile.run_compilation] --- compilation_error")
                metadata["compilation_error"] = e
                metadata["compilation_output"] = std_compile

                tmp_traceback = traceback.format_exc() #traceback.print_exc()
                metadata["compilation_traceback"] = tmp_traceback

                # memory_cleanup(context, device)
                return CompileResult(
                    pre_compiled=True,
                    compiled=False, 
                    runtime_success=False, 
                    metadata=metadata, 
                    main_output=std_compile,
                    main_output_runtime=std_runtime,
                    main_traceback=tmp_traceback if tmp_traceback else "",
                )  # skip further steps
            elif compilation_status=="runtime_error":
                print("[COMPILE - utils_compile.run_compilation] --- runtime_error")
                metadata["runtime_error"] = e
                metadata["runtime_output"] = std_compile
                metadata["compilation_status"] = compilation_status
                tmp_traceback = traceback.format_exc() #traceback.print_exc()
                metadata["runtime_traceback"] = tmp_traceback


                # memory_cleanup(context, device)
                return CompileResult(
                    pre_compiled=True,
                    compiled=True, 
                    runtime_success=False, 
                    metadata=metadata, 
                    main_output=std_compile,
                    main_output_runtime=std_runtime,
                    main_error=format_exception_string(e),
                    main_traceback=tmp_traceback if tmp_traceback else "",
                )  # skip further steps
            elif compilation_status=="model_new_not_found_error":
                print("[COMPILE - utils_compile.run_compilation] --- model_new_not_found_error")
                metadata["runtime_error"] = e
                metadata["runtime_output"] = std_runtime
                metadata["compilation_status"] = compilation_status

                tmp_traceback = traceback.format_exc() #traceback.print_exc()
                metadata["runtime_traceback"] = tmp_traceback
                # memory_cleanup(context, device)
                return CompileResult(
                    pre_compiled=True,
                    compiled=True, 
                    runtime_success=True, 
                    metadata=metadata, 
                    main_output=std_compile,
                    main_output_runtime=std_runtime,
                    main_error=format_exception_string(e),
                    main_traceback=tmp_traceback if tmp_traceback else "",
                )  # skip further steps   
            else:
                # memory_cleanup(context, device)
                print(f"[COMPILE STRANGE] You are having success in compilation / runtime, but still an exception.\n---:\n{e}")
                raise e
                
    print("[COMPILE - utils_compile.run_compilation] Compilation was fully successful.")

    return CompileResult(
        pre_compiled=True,
        compiled=True, 
        runtime_success=True,
        model_new_available=True,
        metadata=metadata, 
        main_output=std_compile,
        main_output_runtime=std_runtime,
) 