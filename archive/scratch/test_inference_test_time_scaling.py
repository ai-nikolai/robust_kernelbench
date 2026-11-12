"""
DESCRIPTION:
    A simple tester script where we can modify things and test them out.

    It's not actually running the pipeline..

USAGE:

python3 robust_kernelbench/run_inference_tts_test.py --experiment exp_V1_gpt_test --trial 4 --previous_trial 1 --online_service_url XYZ
python3 robust_kernelbench/run_inference_tts_test.py --experiment exp_V1_gpt_test --trial 4 --previous_trial 92 --online_service_url XYZ #to test when evalautions.csv is missing...

python3 robust_kernelbench/run_inference_tts_test.py --experiment exp_local_L1_V8_3_glm_5__API1 --trial 4 --previous_trial 1 --online_service_url XYZ --prompt_type kb_multi_stage #to test when evalautions.csv is missing...

python3 robust_kernelbench/run_inference_tts_test.py --experiment exp_local_L1_V8_3_deepseek_r1_0528__API1 --trial 307 --previous_trial 207 --online_service_url XYZ --prompt_type kb_multi_stage


"""

import os
import json
from pathlib import Path
import time
from datetime import datetime

import pandas as pd
from tqdm import tqdm

import statistics

# AI Related..
from datasets import load_from_disk, load_dataset


# COMMENTED OUT FOR SPEED..
# from vllm import LLM, SamplingParams
# from vllm.lora.request import LoRARequest

# import torch

# from transformers import AutoTokenizer

# import gc

# LOCAL IMPORTS
from utils.utils import (
    get_folder_path,
    get_file_name
)

from utils.utils_inference import (
    create_joined_dataset,
    extract_code
)

from prompts_v2 import (
    choose_prompt
)

# import importlib
# importlib.reload(utils)



def write_metadata_json(experiment_name, trial=1,  metadata={}):
    """Writes metadata to a json file in experiment folder"""
    folder_path = get_folder_path(experiment_name, trial)
    metadata_file = os.path.join(folder_path, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

def load_model_and_tokenizer(
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct", 
    tokenizer_name=None,
    gpu_memory_util=0.9, 
    max_lora_rank = 64,
    max_model_len=32000,
    dtype="auto", 
    enable_lora=True,
    tensor_parallel_size=1,
    ):
    """Loads the model and tokenizer..."""
    if not tokenizer_name:
        tokenizer_name = model_name

    # Load the tokenizer (same as the model)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True
    )

    print(f"\n\n===============\nSetting up VLLM \n\n\n")
    # Initialize the LLM engine
    engine = LLM(
        model=model_name,
        tokenizer=tokenizer_name,
        trust_remote_code=True,
        dtype=dtype,  # #auto, half, float16, bfloat16
        gpu_memory_utilization=gpu_memory_util,
        max_model_len=max_model_len,
        enable_lora=enable_lora,
        max_lora_rank=max_lora_rank,
        tensor_parallel_size=tensor_parallel_size,
    )
    print(f"\n\n\n---------------\nVLLM Setup finished.\n\n\n")

    return engine, tokenizer


def sample_file_exists(experiment_name, trial, problem_id, sample_id=None):
    """Check if any code file already exists for a given problem_id/sample_id"""
    folder_path_code = get_folder_path(experiment_name, trial, postfix="code")
    file_name = get_file_name(problem_id, sample_id)
    file_path_base = os.path.join(folder_path_code, file_name)

    # Check if any file with this base name exists (index 0, 1, 2, etc.)
    idx = 0
    file_path = file_path_base + str(idx) + ".py"
    while os.path.exists(file_path):
        return True  # At least one file exists
        idx += 1
        file_path = file_path_base + str(idx) + ".py"
    return False  # No files found

def find_file_idx(file_path_base, ending=".py", starting_idx = 0):
    """Finds the first file with idx that does not exist yet..."""
    # file_path = file_path_base + ending
    idx = starting_idx
    file_path = file_path_base + str(idx) + ending

    while os.path.exists(file_path):
        idx += 1
        file_path = file_path_base + str(idx) + ending

    return file_path, idx


def write_file_with_idx(
    problem_name,
    problem_id,
    generations,
    experiment_name=None,
    extract_code_flag=True,
    trial=None,
    model_name=None,
    suffix=None,
    ending=".py",
    ):
    """Writes a code file"""
    file_name=get_file_name(problem_id, suffix=suffix)

    if not experiment_name:
        experiment_name = "temp_solutions"

    folder_path_code = get_folder_path(experiment_name, trial, postfix="code")

    ending = ending #".py"
    file_path_base = os.path.join(folder_path_code,file_name)

    idx = 0
    # pbar = tqdm(generations)
    print(f"[write file] Total generations: {len(generations)}")
    for generation in generations:
        print(f"[write file] Saving file to: `{file_path_base}`")
        file_path, idx = find_file_idx(file_path_base, ending, starting_idx = idx)
        print(f"[write file] File with idx: {idx}")

        if extract_code_flag:
            try:
                print(f"[write file] Attempting to extracting code.")
                extracted_generation = extract_code(generation, model_name=model_name)
                print(f"[write file] Code extracted.")
                if extracted_generation:
                    generation = extracted_generation
                    print(f"[write file] Code extracted is non-empty.")

            except Exception as e:
                print(f"[write file] Extraction failed:\n--\n{e}")


        if generation:
            with open(file_path, "w") as file:
                print(f"[write file] Writing file: {idx}")
                file.write(generation)
                print(f"[write file] Writing finished")
        else:
            print(f"[write file] generation is empty skipping writting.")

    print("[write file] Done, exiting.")
    return idx


def count_output_tokens(tokenizer, generations):
    """Counts tokens based on generations"""
    out_tokens = {
        "num_tokens" : [],
        "mean" : None,
        "median" : None,
        "stdev" : None,
        "max" : None,
        "min" : None,
    }
    for generated_text in generations:
        out_token_ids = tokenizer(generated_text)
        out_num_tokens = len(out_token_ids["input_ids"])
        out_tokens["num_tokens"].append(out_num_tokens)
    
    out_tokens["mean"] = statistics.mean(out_tokens["num_tokens"])
    out_tokens["median"] = statistics.median(out_tokens["num_tokens"])
    try:
        out_tokens["stdev"] = statistics.stdev(out_tokens["num_tokens"])
    except statistics.StatisticsError as e:
        pass

    out_tokens["max"] = max(out_tokens["num_tokens"])
    out_tokens["min"] = min(out_tokens["num_tokens"])

    return out_tokens


def run_pipeline(
    dataset,
    tokenizer,
    engine,
    run_clean_trial=False,
    num_items=None,
    lora_name=None,
    temperature = 0.6,
    max_tokens = 1000,
    num_samples=1,
    experiment_name=None,
    prompt_type="kernelbench",
    trial=None,
    online_service_url=None,
    online_api_key="<NO_KEY>",
    use_tts_service=False,
    tts_args=None,
    model_name=None,
    **kwargs,
    ):
    """This function is the main function..."""
    print(f"[RUN PIPELINE] Entered the main inference pipeline. \n\tL{level}, \n\tT{trial}, \n\tP-{prompt_type},\n\tDS-{len(dataset)}, \n\tExp-{experiment_name}, \n\tSamples: {num_samples}")
    if len(dataset)<=0:
        return None
    # We assume dataset length of 1.

    if not experiment_name:
        experiment_name = "temp_solutions"

    # Initialize TTS service client if URL provided
    tts_client = None
    if online_service_url:
        from openai import OpenAI
        tts_client = OpenAI(base_url=online_service_url, api_key=online_api_key, timeout=3600)
        tts_args = tts_args or {}

    # Initialize results DataFrame
    results = {
        'problem_name' : [],
        'problem_id': [],
        'in_tokens' : [],
        'out_tokens' : [],
        'temperature' : [],
        'time_taken' : [],
        'generations' : [],
        'file_index' : [],
        'prompt_category' : [],
    }

    dataset = dataset.select(range(num_items)) if num_items else dataset

    # Only set up local vLLM params when not using TTS service
    lora_request = None
    sampling_params = None
    if tts_client is None:
        if lora_name:
            try:
                no_impact_lora_name = "vllm_lora_request"
                lora_request = LoRARequest(no_impact_lora_name, 1, lora_name)
            except Exception as e:
                print("NO LORA ADAPTER FOUND")
                print(e)
                lora_request=None
        else:
            print("We don't have Lora...")

        sampling_params = SamplingParams(
            n=num_samples,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    pbar = tqdm(dataset)
    for sample in pbar:
        try:
            print("\n====s")
            # These three keys are essential...
            problem_name = sample["name"]
            if not problem_name.endswith("_"):
                problem_name = problem_name+"_"
            problem_id = sample["problem_id"]
            sample_id = sample.get("sample_id", None)

            # Skip if files already exist for this sample
            if sample_file_exists(experiment_name, trial, problem_id, sample_id):
                pbar.set_description(f"[RUN PIPELINE] SKIPPING (exists): {problem_name}")
                print(f"[RUN PIPELINE] Skipping {problem_name} (problem_id={problem_id}, sample_id={sample_id}) - files already exist")
                continue

            original_source = sample["code"]

            # Here we need to add additional stuff like: what ever is needed for the other prompts...

            pbar.set_description(f"[RUN PIPELINE] Running Inference: {problem_name}")

            # TODO potentially add another section here for reflection...
            
            prompt_func, prompt_category = choose_prompt(sample, trial, prompt_type, run_clean_trial)
            print(f"[RUN PIPELINE] Chose Prompt: {prompt_category} for p={problem_id}, s={sample_id}")
            try:
                prompt = prompt_func(sample, **kwargs)
            except Exception as e:
                print(f"\n\n[RUN PIPELINE] FAILED TO GET PROMPT!")
                print(f"[RUN PIPELINE] {sample}")
                raise e
            if prompt:
                print(f"[RUN PIPELINE] Got the prompt.")
            else:
                print(f"[RUN PIPELINE] Got empty prompt: {type(prompt)}")

            file_idx = write_file_with_idx(
                problem_name,
                problem_id,
                [prompt],
                experiment_name=experiment_name,
                extract_code_flag=False,
                trial=trial,
                model_name=model_name,
                suffix="prompt",
                ending=".txt",
            )        

            messages = [{
                "content": prompt,
                "role" : "user",
            }]

            start = time.time()
            
            # COMMENT OUT START 1
            # if tts_client is not None:
            #     # Online API Path: call remote LLM service via OpenAI SDK

            #     if use_tts_service:
            #         # Use the LLM Booster Path (citation later)
            #         extra_body = {
            #             "tts_strategy": tts_args.get("tts_strategy", "offline_bon"),
            #             "tts_num_trajectories": tts_args.get("tts_num_trajectories", 8),
            #             "tts_scorer": tts_args.get("tts_scorer", "entropy"),
            #             "tts_candidates_per_step": tts_args.get("tts_candidates_per_step", 4),
            #             "tts_beam_size": tts_args.get("tts_beam_size", 4),
            #             "tts_max_steps": tts_args.get("tts_max_steps", 100),
            #             "tts_score_aggregation": tts_args.get("tts_score_aggregation", "min"),
            #             "tts_window_size": tts_args.get("tts_window_size", None),
            #         }
            #     else:
            #         extra_body = {}

            #     response = tts_client.chat.completions.create(
            #         model=model_name,
            #         messages=messages,
            #         temperature=temperature,
            #         max_tokens=max_tokens,
            #         extra_body=extra_body,
            #     )

            #     generations = [response.choices[0].message.content]
            #     in_num_tokens = response.usage.prompt_tokens if response.usage else 0
            #     out_num_tokens_raw = response.usage.completion_tokens if response.usage else 0
            # else:
            #     # Original local vLLM path: 
            #     tokens = tokenizer.apply_chat_template(messages,
            #         tokenize = False,
            #         add_generation_prompt = True,
            #         enable_thinking=False,
            #     )

            #     in_token_ids = tokenizer.apply_chat_template(messages, add_generation_prompt = True, enable_thinking=False,)
            #     in_num_tokens = len(in_token_ids)

            #     outputs = engine.generate([tokens], sampling_params=sampling_params, lora_request=lora_request, use_tqdm=False)

            #     generations = []
            #     for output in outputs:
            #         single_gen = output.outputs[0].text.strip()
            #         generations.append(single_gen)
            #     out_num_tokens_raw = None
            # COMMENT OUT END 1
            end = time.time()
            print(f"[RUN PIPELINE] Took: {end-start}")
            
            # COMMENT OUT START 2
            # print(f"[RUN PIPELINE] Num Generations: {len(generations)}.")
            # file_idx = write_file_with_idx(
            #     problem_name,
            #     problem_id,
            #     generations,
            #     experiment_name=experiment_name,
            #     extract_code_flag=True,
            #     trial=trial,
            #     model_name=model_name
            # )
            # 
            # if tts_client is not None:
            #     out_num_tokens = {"num_tokens": [out_num_tokens_raw], "mean": out_num_tokens_raw, "median": out_num_tokens_raw, "stdev": None, "max": out_num_tokens_raw, "min": out_num_tokens_raw}
            # else:
            #     out_num_tokens = count_output_tokens(tokenizer, generations)
            # 
            # results["problem_name"].append(problem_name)
            # results["problem_id"].append(problem_id)
            # results["generations"].append(generations)
            # results["in_tokens"].append(in_num_tokens)
            # results["out_tokens"].append(out_num_tokens)
            # results["temperature"].append(temperature)
            # results["time_taken"].append(end-start)
            # results["file_index"].append(file_idx)
            # results["prompt_category"].append(prompt_category)
            # COMMENT OUT END 2

        except Exception as e:
            try:
                print(f"[RUN PIPELINE] - FAILED for sample pid={problem_id} sid={sample_id}")
            except:
                print(f"[RUN PIPELINE] - FAILED for sample:\n---\n{sample}\n---\n")
            print("===exception start")
            print(f"Exception:{e}")
            print(f"Exception Type:{type(e)}")
            print(f"Exception str: {str(e)}")
            print("---exception end")
        print("----+end of sample\n\n")

    results_df = pd.DataFrame.from_dict(results)
    return results_df

def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1, help="Num of Parallel responses")
    parser.add_argument("--prompt_type", type=str, default="improve", help="Which prompt to use...")
    parser.add_argument("--trial", type=int, help="Which trial you are currently running...")
    parser.add_argument("--previous_trial", type=int, help="Which previous trial to use...")
    # Whether to ignore previous trial
    parser.add_argument("--run_clean_trial", action="store_true", help="Whether to run clean trial (i.e. ignoring it is trial > 1)")


    parser.add_argument("--experiment_name", type=str, default="exp_test_run", help="experiment names")
    parser.add_argument("--temperature", type=float, default=0.6, help="temp for LLM")
    parser.add_argument("--max_mem_util", type=float, default=0.8, help="temp for LLM")
    parser.add_argument("--num_items", type=int, help="How many data points to run...")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Model Name...")
    parser.add_argument("--level", type=int, default=1, help="Which level to run...")
    parser.add_argument("--max_model_len", type=int, default=32000, help="Max input and output tokens...")
    parser.add_argument("--max_tokens", type=int, default=8000, help="max tokens to generate")
    parser.add_argument("--tokenizer_name", type=str, help="tokenizer name if different from model name.")
    parser.add_argument("--filter", action="store_true", help="Whether to run only on the filtered examples...")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="TP size for multi-gpu, single node...")
    # TODO: add lora support

    # TTS service args
    parser.add_argument("--online_service_url", type=str, default=None, help="URL of API LLM. eg.: http://localhost:8001/v1 OR https://openrouter.ai/api/v1")
    parser.add_argument("--api_key_env_var", type=str, default="LLM_API_KEY", help="What should the env variable be called for the LLM API key, default=LLM_API_KEY")
    # Whether to actually use the LLM BOOSTER service, (only applicable if online_service_url is set to true). If use_tts_service is false, but URL is set, just the API LLM will be used.
    parser.add_argument("--use_tts_service", action="store_true", help="Whether to run tts service or just online service...")

    # BOOSTER LLM IMPLEMENTATION (Citation Later)
    parser.add_argument("--tts_strategy", type=str, default="offline_bon", help="TTS strategy: offline_bon, online_bon, beam_search")
    parser.add_argument("--tts_num_trajectories", type=int, default=8, help="Number of trajectories for offline BoN")
    parser.add_argument("--tts_scorer", type=str, default="entropy", help="Scorer: entropy, perplexity, sequence_prob")
    parser.add_argument("--tts_candidates_per_step", type=int, default=4, help="Candidates per step for online BoN / beam search")
    parser.add_argument("--tts_beam_size", type=int, default=4, help="Beam size for beam search")
    parser.add_argument("--tts_max_steps", type=int, default=100, help="Max reasoning steps")
    parser.add_argument("--tts_score_aggregation", type=str, default="min", help="Score aggregation: min, mean, max, product, last")
    parser.add_argument("--tts_window_size", type=int, default=5, help="Window size for scoring (1-5 steps)")

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()

    level1_representative_subset_problem_ids = [1, 3, 6, 18, 23, 26, 33, 36, 40, 42, 48, 54, 57, 65, 77, 82, 86, 87]
    level2_representative_subset_problem_ids = [1, 2, 8, 18, 23, 28, 33, 43]
    level3_representative_subset_problem_ids = [1, 5, 8, 11, 20, 33, 38, 43]

    if args.online_service_url:
        # Model runs on the TTS service side, no local loading needed
        engine, tokenizer = None, None
    else:
        engine, tokenizer = load_model_and_tokenizer(
            model_name=args.model_name,
            max_model_len=args.max_model_len,
            tokenizer_name=args.tokenizer_name,
            gpu_memory_util=args.max_mem_util,
            tensor_parallel_size=args.tensor_parallel_size,
        )

    data_repo = "ai-nikolai/KernelBench"

    dataset = load_dataset(data_repo)
    level = args.level
    split = f"level_{level}"
    dataset = dataset[split]

    if args.filter:
        print("Filtering Data...")
        dataset = dataset.filter(lambda x: x["problem_id"] in level1_representative_subset_problem_ids) #level1_representative_subset_problem_ids)
    
    print(dataset)

    NUM_ITEMS = args.num_items #this will run the entire dataset...

    # LORA_NAME ="my_lora"

    if args.trial>1 and not args.run_clean_trial:
        if not args.previous_trial:
            previous_trial = args.trial - 1
        else:
            previous_trial = args.previous_trial
        previous_folder_path = get_folder_path(args.experiment_name, previous_trial)
        generations_path = os.path.join(previous_folder_path,"generations.csv")
        previous_generations = pd.read_csv(generations_path)
    
        evaluations_path = os.path.join(previous_folder_path,"evaluations.csv")
        try:
            previous_evaluations = pd.read_csv(evaluations_path)
        except:
            previous_evaluations = None

        compilations_path = os.path.join(previous_folder_path,"compilations.csv")
        try:
            previous_compilations = pd.read_csv(compilations_path)
        except:
            previous_compilations = None

        dataset = create_joined_dataset(
            dataset,
            previous_generations = previous_generations,
            previous_compilations = previous_compilations,
            previous_evaluations = previous_evaluations,
        ) #this dataset now has sample_id in it as well (trial>1)

    # else:
    #     previous_generations = None
    #     previous_evaluations = None
    #     previous_compilations = None

    tts_args_dict = None
    if args.online_service_url:

        online_api_key = os.environ.get(args.api_key_env_var, "<NO_KEY>")
        
        if args.use_tts_service:
            tts_args_dict = {
                "model_name": args.model_name,
                "tts_strategy": args.tts_strategy,
                "tts_num_trajectories": args.tts_num_trajectories,
                "tts_scorer": args.tts_scorer,
                "tts_candidates_per_step": args.tts_candidates_per_step,
                "tts_beam_size": args.tts_beam_size,
                "tts_max_steps": args.tts_max_steps,
                "tts_score_aggregation": args.tts_score_aggregation,
                "tts_window_size": args.tts_window_size,
            }


    print("[MAIN] All init finished. Now running the pipeline...")
    results_df = run_pipeline(
        dataset,
        tokenizer,
        num_items=NUM_ITEMS,
        engine=engine,
        run_clean_trial=args.run_clean_trial,
        # lora_name=LORA_NAME,
        temperature = args.temperature,
        max_tokens = args.max_tokens,
        num_samples=args.num_samples,
        experiment_name=args.experiment_name,
        prompt_type=args.prompt_type,
        trial=args.trial,
        online_service_url=args.online_service_url,
        online_api_key=online_api_key,
        tts_args=tts_args_dict,
        use_tts_service = args.use_tts_service,
        model_name=args.model_name,
    )

    # Show results summary
    print("\nResults DataFrame Head:")
    # print(results_df.head())
    # Path("out").mkdir(parents=True, exist_ok=True)
    # results_df.to_csv(f'out/generations_{args.experiment_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)
    
    # folder_path = get_folder_path(args.experiment_name, args.trial)
    # results_df.to_csv(os.path.join(folder_path,'generations.csv'), index=False)

    # Metadata datastuff
    metadata = {}
    metadata["model_name"] = args.model_name
    metadata["max_model_len"] = args.max_model_len
    metadata["max_mem_util"] = args.max_mem_util
    metadata["temperature"] = args.temperature
    metadata["max_tokens"] = args.max_tokens
    metadata["num_samples"] = args.num_samples
    metadata["prompt_type"] = args.prompt_type
    metadata["trial"] = args.trial
    metadata["num_items"] = NUM_ITEMS
    metadata["level"] = args.level
    if args.online_service_url:
        metadata["online_service_url"] = args.online_service_url
        metadata["tts_strategy"] = args.tts_strategy
        metadata["tts_num_trajectories"] = args.tts_num_trajectories
        metadata["tts_scorer"] = args.tts_scorer
        metadata["use_tts_service"] = args.use_tts_service


    # write_metadata_json(args.experiment_name, args.trial, metadata)

    print("FINISHED INFERENCE SCRIPT. TRYING TO CLEAN_UP")
    # if engine is not None:
    #     del engine
    # gc.collect()
    # torch.cuda.empty_cache()

    exit()