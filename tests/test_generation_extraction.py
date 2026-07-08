# robust_kernelbench/experiments/exp_local_L1_v9_qwen_coder_next_kernelbench_API1/trial_1/generations.csv
# Header: problem_name,problem_id,in_tokens,out_tokens,temperature,time_taken,generations,file_index,prompt_category

import pandas
import re
import sys

# Load the CSV file
file_path = "experiments/exp_local_L1_v9_qwen_coder_next_kernelbench_API1/trial_1/generations.csv"
df = pandas.read_csv(file_path)

file_path = "experiments/exp_local_L1_v9_qwen_coder_next_kernelbench_API1/trial_1/compilations.csv"
d2 = pandas.read_csv(file_path)

# Example: print the first few rows to verify loading
print("Loaded generations.csv successfully. First few rows:")
print(df.head())

# Complete the LLM function with a sample implementation
def extract_code_from_llm_response(response_text: str) -> str:
    """
    Extract code from an LLM response.
    Uses simple regex to find code blocks.
    """
    # Try to find code blocks enclosed in triple backticks
    pattern = r"```(?:python)?\s*([\s\S]*?)```"
    match = re.search(pattern, response_text)
    if match:
        return match.group(1).strip()
    # Fallback: return entire text
    return response_text.strip()

# Example usage of the function on a generation column
if 'generations' in df.columns:
    df['extracted_code'] = df['generations'].apply(extract_code_from_llm_response)
    print("Extracted code from generations column. Sample:")
    print(df.columns)
    input()
    # df[['problem_name', 'generations','extracted_code']].to_csv("test.csv", index=False)
    # Iterate through rows and print generations and extracted code
    print("\nIterating through rows:")
    for idx, row in df.iterrows():
        print(f"\n--- Row {idx} ---")
        print(f"Extracted code: {row['extracted_code'].encode().decode('unicode_escape')}")
        x=input(f"1:show [{row['problem_name']}]")
        if x=="1":
            print(f"Generations: {row['generations'].encode().decode('unicode_escape')}")

