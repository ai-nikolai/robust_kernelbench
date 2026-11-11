import os

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def load_model_and_tokenizer(
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct", 
    tokenizer_name=None,
    gpu_memory_util=0.9, 
    max_lora_rank = 64,
    max_model_len=32000,
    dtype="auto", 
    enable_lora=True
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
    )
    print(f"\n\n\n---------------\nVLLM Setup finished.\n\n\n")

    return engine, tokenizer

def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model name to use...")

    args = parser.parse_args()

    return args 


if __name__=="__main__":
    print("[Main] Entering...")
    print(f"[Main] HF_HOME = {os.environ['HF_HOME']}")
    # print(f"[Main] TORCHINDUCTOR_CACHE_DIR = {os.environ['TORCHINDUCTOR_CACHE_DIR']}")
    # print(f"[Main] TMPDIR = {os.environ['TMPDIR']}")

    args = get_args()

    if args.model_name:
        model_name = args.model_name
    else:
        model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    engine, tokenizer = load_model_and_tokenizer(model_name=model_name)
    
    sampling_params = SamplingParams(
        n=1,
        temperature=0.9,
        max_tokens=100,
        # stop=["</s>"], #need to investigate this further... (and make it better?)
        # include_stop_str_in_output=False #default is False
    )

    messages = [{
        "content": "Please tell me about life in the style of Yoda.",
        "role" : "user",
    }]

    tokens = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    
    outputs = engine.generate([tokens], sampling_params=sampling_params, lora_request=lora_request, use_tqdm=False)


    for idx, output in enumerate(outputs):
        single_gen = output.outputs[0].text.strip()
        print(f"\n---------\n[OUPUT] idx:{idx}")
        print(single_gen)