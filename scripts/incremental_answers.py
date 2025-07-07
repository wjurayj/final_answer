import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import glob
import json
from tqdm import tqdm
import argparse
from typing import List, Dict, Any

class ModelHandler:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="bfloat16",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def truncate_tokens(self, input_text: str, length: int) -> str:
        tokens = self.tokenizer(input_text, return_tensors="pt")['input_ids']
        truncated_tokens = tokens[:, :length]
        output = self.tokenizer.decode(truncated_tokens[0], skip_special_tokens=True)
        return output

    def get_truncated_completion(self, collated_input: str, n_tokens: int, answer_prefix: str) -> Dict[str, Any]:
        input_tokens = self.truncate_tokens(collated_input, n_tokens) + answer_prefix

        inputs = self.tokenizer(input_tokens, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=3,
                num_return_sequences=1,
                temperature=0.0,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
            )
        
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        
        generated_tokens = outputs.sequences[0, inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return {
            'choices': [{
                'text': generated_text,
                'logprobs': {
                    'tokens': self.tokenizer.convert_ids_to_tokens(generated_tokens),
                    'token_logprobs': transition_scores[0].tolist()
                }
            }]
        }

class DataAnalyzer:
    def __init__(self, model_handler: ModelHandler, cache_file: str):
        self.model_handler = model_handler
        self.cache_file = cache_file

    @staticmethod
    def is_digit(token: str) -> bool:
        return token.strip() in [str(i) for i in range(10)]

    @staticmethod
    def is_multiple_choice(token: str) -> bool:
        return token.strip().lower() in {"a", "b", "c", "d"}

    @staticmethod
    def collate_input(example: Dict[str, Any]) -> str:
        gen_input = example['arguments']['gen_args_0']['arg_0']
        gen_output = example['filtered_resps'][0]
        return gen_input + gen_output

    def analyze_token_posteriors(self, data: List[Dict[str, Any]], increment: int, 
                               max_length: int, answer_prefix: str, multiple_choice: bool=False, n_tokens: int=3) -> List[Dict[str, Any]]:
        output = []
        for example in data:
            collated = self.collate_input(example)
            results = {"correct_answer": example['target']}
            
            for inc_count in tqdm(range(1, (max_length // increment) + 1)):
                n_tokens = increment * inc_count
                
                if n_tokens > max_length:
                    continue
                
                input_tokens = len(self.model_handler.tokenizer(
                    example['arguments']['gen_args_0']['arg_0'])['input_ids']
                )

                completion = self.model_handler.get_truncated_completion(
                    collated, n_tokens + input_tokens, answer_prefix
                )
                
                tokens = completion['choices'][0]['logprobs']['tokens']
                logprobs = completion['choices'][0]['logprobs']['token_logprobs']
                
                answer = ""
                digit_prob = 0
                counted_logprobs = []
                for token, logprob in zip(tokens, logprobs):
                    if multiple_choice:
                        if self.is_multiple_choice(token):
                            answer += token
                            digit_prob += logprob
                            counted_logprobs.append(logprob)
                    else:
                        if self.is_digit(token):
                            answer += token
                            digit_prob += logprob
                            counted_logprobs.append(logprob)

                results[n_tokens] = {
                    "answer": answer,
                    "logprob": digit_prob if answer else -5,
                    "mean_logprob": digit_prob / len(answer) if answer else -5,
                    "min_logprob": min(counted_logprobs) if answer else -5,
                }

            
            output.append(results)
            with open(self.cache_file, 'a+') as f:
                json.dump({"example_id": example.get('doc_id', ''), "results": results}, f)
                f.write('\n')

        return output

def load_data(base_dir: str, folder_name: str, model_name: str, file_pattern: str) -> List[Dict[str, Any]]:
    folder_path = os.path.join(base_dir, folder_name, model_name)
    filenames = []
    
    if os.path.exists(folder_path):
        files = glob.glob(os.path.join(folder_path, file_pattern))

        if files:
            most_recent_file = max(files)
            filenames.append(most_recent_file)
        else:
            print(f'No files found in {folder_path}')
    else:
        print(f'Folder {folder_path} does not exist')
        return []

    data = []
    for file in filenames:
        with open(file, 'r') as f:
            data.extend([json.loads(line) for line in f])
    
    return data

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze token posteriors for LLM responses')
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name of the model to use (e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Base directory containing the data')
    parser.add_argument('--folder_name', type=str, default='forcing8k',
                      help='Name of the folder containing the data')
    parser.add_argument('--file_pattern', type=str, default='samples_*',
                      help='Pattern for the file in the folder to grab the most recent version of')
    parser.add_argument('--token_limit', type=int, default=8000,
                      help='Maximum number of tokens to analyze')
    parser.add_argument('--increment', type=int, default=100,
                      help='Token increment size')
    parser.add_argument('--think_delimiter', type=str, default='</think>',
                      help='Delimiter for thinking section')
    parser.add_argument('--cache_file', type=str, required=True,
                      help='Path to cache file')
    parser.add_argument('--output_file', type=str, required=True,
                      help='Path to output file')
    parser.add_argument('--multiple_choice', action='store_true',
                        help='Whether to analyze multiple choice answers (A,B,C,D)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize model handler
    model_handler = ModelHandler(args.model_name)

    # Load data
    data = load_data(args.data_dir, args.folder_name, args.model_name.replace('/', '__'), args.file_pattern)
    
    if not data:
        print("No data found. Exiting.")
        return

    # Set up analyzer
    analyzer = DataAnalyzer(model_handler, args.cache_file)
    
    # Determine answer prefix based on think delimiter
    answer_prefix = '\n</think>\nFinal Answer: \\boxed{' if args.think_delimiter == '</think>' else '<|im_start|>answer\nFinal Answer: \\boxed{'
    
    # Run analysis
    print('Starting analysis...')
    results = analyzer.analyze_token_posteriors(
        data, 
        args.increment,
        args.token_limit,
        answer_prefix,
        multiple_choice=args.multiple_choice,
        n_tokens=1 if args.multiple_choice else 3,
    )

    # Save results
    with open(args.output_file, 'w+') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
