import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, GPT2Model

from utils.data_utils import read_single_row_data

import pandas as pd
import torch
from torch.utils.benchmark import Timer


def print_gpu_memory_usage():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} | Name: {torch.cuda.get_device_name(i)}")
        print(f"     | Allocated: {torch.cuda.memory_allocated(i) / (1024 ** 3):.3f} GB")
        print(f"     | Cached: {torch.cuda.memory_reserved(i) / (1024 ** 3):.3f} GB")


def prepare_prompts(model_name):
    prompts = {
        'jellyfish': """You are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n### Instruction:\n\nYou are tasked with determining whether two records listed below are the same based on the information provided.\nCarefully compare the attributes of each record before making decision.\nNote: Missing values (N/A or \"nan\") should not be used as a basis for your decision.\nRecord A: [{}]\nRecord B: [{}]\nAre record A and record B the same entity? Choose your answer from: [Yes, No].\n\n### Response:\n\n""",
        'mixtral': """[INST]Do the two entity descriptions refer to the same real-world entity? Answer with 'Yes' if they do and 'No' if they do not.\nEntity 1: {}\nEntity 2: {}[/INST]""",
        'solar': """### User: Do the two entity descriptions refer to the same real-world entity? Answer with 'Yes' if they do and 'No' if they do not.\nEntity 1: {}\nEntity 2: {}\n\n### Assistant:\n""",
        'beluga': """### System:\nYou are Stable Beluga, an AI that follows instructions extremely well.\n\n### User: Do the two entity descriptions refer to the same real-world entity? Answer with 'Yes' if they do and 'No' if they do not.\nEntity 1: {}\nEntity 2: {}\n\n### Assistant:\n"""
    }

    if model_name == 'gpt2':
        data = read_single_row_data('data/prepared/dbgo', mode='mode1', print_info=False)[0]
        return data['text'].tolist()
    elif model_name in ['jellyfish', 'mixtral', 'solar', 'beluga']:
        data_df = pd.read_csv('data/prepared/dbgo/train.csv')
        l_columns = [col for col in data_df.columns if col.endswith('_l')]
        r_columns = [col for col in data_df.columns if col.endswith('_r')]
        data_df['textA'] = data_df.apply(lambda x: '\t'.join([str(x[col]) for col in l_columns]), axis=1)
        data_df['textB'] = data_df.apply(lambda x: '\t'.join([str(x[col]) for col in r_columns]), axis=1)
        prompts = [prompts[model_name].format(data_df.iloc[i]['textA'], data_df.iloc[i]['textB']) for i in range(len(data_df))]
        return prompts
    else:
        raise ValueError(f"Model {model_name} not supported.")


def benchmark_inference(model, tokenizer, dataset, initial_batch_size=4, max_batch_size=16384):
    """
    Benchmarks the inference time of a model with the largest possible batch size that fits in memory,
    and counts the number of tokens processed.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Helper function to test if a batch size fits in memory
    def can_fit_in_memory(batch_size):
        try:
            inputs = tokenizer(dataset[:batch_size], return_tensors="pt", truncation=True, max_length=350, padding=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            with torch.no_grad():
                model(**inputs)
            return True
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                return False
            else:
                raise e

    # Determine the largest batch size that fits in memory
    batch_size = initial_batch_size
    while batch_size <= max_batch_size:
        if can_fit_in_memory(batch_size):
            largest_batch_size = batch_size
            batch_size *= 2
        else:
            break

    # Prepare the inputs with the determined largest batch size
    inputs = tokenizer(dataset[:largest_batch_size], return_tensors="pt", truncation=True, max_length=350, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Count the total number of tokens processed
    total_tokens = sum(len(token_ids) for token_ids in inputs['input_ids'])

    def inference():
        with torch.no_grad():
            model(**inputs)

    # Benchmark the inference time
    t = Timer(
        stmt='inference()', globals=locals()
    )

    result = t.timeit(100)

    # Return the results
    return {
        "batch_size": largest_batch_size,
        "total_time": result.mean * 100,  # Total time for 100 runs
        "avg_time_per_inference": result.mean,  # Average time per inference
        "tokens_processed": total_tokens,
        "throughput_of_records": largest_batch_size / result.mean,  # Throughput inferences per second
        "throughput_of_tokens": total_tokens / result.mean,  # Throughput tokens per second
    }


parser = argparse.ArgumentParser(description='The inference experiment.')
parser.add_argument('--model_name', type=str)
args = parser.parse_args()

model_name = args.model_name
access_token = 'replace with your own access token for HuggingFace API'
cache_dir = 'replace with your own cache directory for storing very large models'
if model_name == 'gpt2':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('saved_models/loo_amgo_gpt2', device_map='auto')
elif model_name == 'jellyfish':
    model_id = "NECOUDBFM/Jellyfish-13B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map='auto'
    )
elif model_name == 'mixtral':
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=access_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        use_auth_token=access_token
    )
elif model_name == 'solar':
    model_id = "upstage/Llama-2-70b-instruct-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, use_auth_token=access_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        # load_in_8bit=True,
        use_auth_token=access_token,
        rope_scaling={"type": "dynamic", "factor": 2}  # allows handling of longer inputs
    )
elif model_name == 'beluga':
    model_id = "stabilityai/StableBeluga2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
else:
    raise ValueError(f"Model {model_name} not supported.")

print('Inference benchmarking on {} model...'.format(model_name))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print_gpu_memory_usage()
dataset = prepare_prompts(model_name)
benchmark_results = benchmark_inference(model, tokenizer, dataset)
print(benchmark_results)
print('Benchmarking finished.')
