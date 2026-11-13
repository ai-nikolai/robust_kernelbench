import ast
import os
from collections import defaultdict

import pandas as pd

from datasets import Dataset, load_dataset

from robust_kernelbench.logging_utils import get_logger

from .utils import get_folder_path

logger = get_logger(__name__)


def extract_code(generated_response, model_name=None):
    """Function to extract code from the responses...."""
    print("[Extracting Code] Start")
    if not generated_response:
        print("[Extracting Code] Generated Response Empty - exiting the function.")
        return ""

    try:
        try:
            # Handle gpt-oss thinking mode format
            is_oss_model = model_name and ("oss" in model_name.lower())
        except Exception as e:
            print("[Extracting Code] - Error, oss model flag")
            print(f"[Extracting Code] --exception: {e}")
            is_oss_model = False

        if is_oss_model:
            print("[Extracting Code] - entering part 1")
            # Remove analysis section if present
            if "analysis" in generated_response:
                parts = generated_response.split("analysis")
                if len(parts) > 1:
                    after_analysis = parts[1]
                    if "```python" not in after_analysis:
                        print("NOTE: No python code block found after analysis, using the full response.")
                        pass
                    elif "assistant" in after_analysis:
                        generated_response = after_analysis.split("assistant")[-1]
                    else:
                        generated_response = after_analysis

        if "Optimized with CUDA operators:" in generated_response:
            print("[Extracting Code] - entering part 2")
            resp = generated_response.split("Optimized with CUDA operators:")[-1]
            if resp:
                return resp

        print("[Extracting Code] - entering part 3")
        num_code = generated_response.count("```python")
        num_cpp_code = generated_response.count("```cpp")
        num_cuda_code = generated_response.count("```cuda")
        num_code_code = generated_response.count("```code")
        special_blocks = generated_response.count("```")
        print("[Extracting Code] - entering part 4")

        if not num_code:
            print("[Extracting Code] - entering part 5")

            print("[Extracting Code] - NOTE, there were no python tags found, attempting split by [```cpp,```cuda, ```]")
            if num_cpp_code:
                answer = generated_response.split('```cpp')[-1]
                answer = answer.split("```")[0]
                if answer:
                    return answer

            if num_cuda_code:
                answer = generated_response.split('```cuda')[-1]
                answer = answer.split("```")[0]
                if answer:
                    return answer

            if num_code_code:
                answer = generated_response.split('```code')[-1]
                answer = answer.split("```")[0]
                if answer:
                    return answer

            if special_blocks:
                potential_answer = generated_response.split('```')
                answer = ""
                if potential_answer[-1]:
                    answer = potential_answer[-1]
                elif potential_answer[-2]:
                    answer = potential_answer[-2]
                if answer:            
                    return answer

            print("[Extracting Code] WARNING, no matching code blocks found / it's empty, returning the full response.")
            return generated_response
        
        if num_code > 1:
            print("[Extracting Code] Note there is more than one code block!")

        # For gpt-oss models: collect all blocks and choose the best one
        if is_oss_model and num_code > 1:
            code_blocks = []
            parts = generated_response.split("```python")
            for part in parts[1:]:
                if "```" in part:
                    code = part.split("```")[0].strip()
                    if code:
                        code_blocks.append(code)

            # Look for block with cuda_src
            for code in code_blocks:
                if "cuda_src" in code or 'cuda_src = r"""' in code:
                    print("[Extracting Code] [gpt-oss] Found block with cuda_src, using it")
                    return code
            # If no cuda_src found, look for CUDA kernel patterns
            for code in code_blocks:
                if "__global__" in code or "<<<" in code:
                    print("[Extracting Code] [gpt-oss] Found block with CUDA kernel, using it")
                    return code
            # Fall back to first block
            print("[Extracting Code] [gpt-oss] No cuda_src/CUDA patterns found, using first block")
            if code_blocks:
                return code_blocks[0]

        if special_blocks % 2 == 1:
            print("[Extracting Code] WARNING: Code blocks are not matching, it could be buggy.")

        # DEFAULT: return last code block...
        answer = generated_response.split('```python')[-1]
        answer = answer.split("```")[0]
        if answer:
            return answer
        else:
            if generated_response:
                return generated_response
            else:
                return ""
    except Exception as e:
        print("[Extracting Code] Main Error - utils_inferece.extract_code")
        print(e)
        print("[Extracting Code] ---exception end")
        print(generated_response)
        print("[Extracting Code] ---extraction end")
        if generated_response:
            return generated_response
        else:
            return ""


def _parse_generations(value):
    """Decode the list stored in a generations.csv cell without ``eval``."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return [value]
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    return [value]


def _compilation_updates(row) -> dict:
    def text(key):
        value = row.get(key, "")
        return "" if isinstance(value, float) else value

    return {
        "model_new_available": row.get("model_new_available", False),
        "compilation_output": text("main_output"),
        "compilation_output_runtime": text("main_output_runtime"),
        "compilation_output_error": text("main_error"),
        "main_output": text("main_traceback"),
    }


def _evaluation_updates(row) -> dict:
    def text(key):
        value = row.get(key, "")
        return "" if isinstance(value, float) else value

    return {
        "second_compile_success": row.get("compiled", False),
        "correctness_success": row.get("correctness_success", False),
        "correctness": row.get("correctness", False),
        "runtime": row.get("runtime"),
        "runtime_original": row.get("runtime_original"),
        "correctness_output": text("main_output"),
        "correctness_output_traceback": text("main_traceback"),
        "correctness_output_error": text("main_error"),
        "main_output": text("main_error"),
    }


def create_joined_dataset(
    dataset,
    previous_generations=None,
    previous_compilations=None,
    previous_evaluations=None,
):
    """Attach the previous round's artefacts to every dataset row.

    The result has the same columns and sample expansion as before: each
    ``(problem_id, sample_id)`` present in the previous round becomes its own
    row, and problems without previous data keep ``sample_id = 0``.  The
    implementation is a single pass over the inputs instead of repeatedly
    scanning all keys for every dataset example.
    """
    if previous_generations is None and previous_compilations is None and previous_evaluations is None:
        return dataset

    logger.info(
        "joining previous round: generations=%s compilations=%s evaluations=%s",
        len(previous_generations) if previous_generations is not None else "none",
        len(previous_compilations) if previous_compilations is not None else "none",
        len(previous_evaluations) if previous_evaluations is not None else "none",
    )

    previous: dict[tuple[int, int], dict] = {}
    sample_ids: dict[int, set[int]] = defaultdict(set)

    def ensure(problem_id, sample_id):
        key = (int(problem_id), int(sample_id))
        previous.setdefault(key, {})
        sample_ids[key[0]].add(key[1])
        return key

    if previous_generations is not None:
        for _, row in previous_generations.iterrows():
            problem_id = int(row["problem_id"])
            for sample_id, generation in enumerate(_parse_generations(row["generations"])):
                key = ensure(problem_id, sample_id)
                previous[key]["generation"] = generation

    if previous_compilations is not None:
        for _, row in previous_compilations.iterrows():
            key = ensure(row["problem_id"], row["sample_id"])
            previous[key].update(_compilation_updates(row))

    if previous_evaluations is not None:
        for _, row in previous_evaluations.iterrows():
            key = ensure(row["problem_id"], row["sample_id"])
            previous[key].update(_evaluation_updates(row))

    expanded = []
    for example in dataset:
        problem_id = int(example["problem_id"])
        for sample_id in sorted(sample_ids.get(problem_id, {0})):
            joined = example.copy()
            joined["sample_id"] = sample_id
            joined.update(previous.get((problem_id, sample_id), {}))
            expanded.append(joined)

    logger.info("joined dataset has %s rows", len(expanded))
    return Dataset.from_list(expanded)


if __name__ == "__main__":
    EXPERIMENT_NAME = "exp_local_L1_V8_2_glm_5__API1"

    data_repo = "ai-nikolai/KernelBench"

    dataset = load_dataset(data_repo)
    level = 1
    split = f"level_{level}"
    dataset = dataset[split]

    previous_folder_path = get_folder_path(EXPERIMENT_NAME, 1)
    generations_path = os.path.join(previous_folder_path,"generations.csv")
    previous_generations = pd.read_csv(generations_path)

    evaluations_path = os.path.join(previous_folder_path,"evaluations.csv")
    previous_evaluations = pd.read_csv(evaluations_path)

    compilations_path = os.path.join(previous_folder_path,"compilations.csv")
    previous_compilations = pd.read_csv(compilations_path)

    dataset = create_joined_dataset(
        dataset,
        previous_generations = previous_generations,
        previous_compilations = previous_compilations,
        previous_evaluations = previous_evaluations,
    )

    print(dataset)

    # Tests that are needed for the prompts_v2 file: