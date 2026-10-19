"""Contains classes for querying large language models."""
from math import ceil
import os
import time
from tqdm import tqdm
from abc import ABC, abstractmethod

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import BitsAndBytesConfig
import gc

from automatic_prompt_engineer import evaluate
from automatic_prompt_engineer.evaluation import likelihood

gpt_costs_per_thousand = {
    'davinci': 0.0200,
    'curie': 0.0020,
    'babbage': 0.0005,
    'ada': 0.0004
}

class HFCompletion:

    @classmethod
    def create(cls, model, tokenizer, **kwargs):
        #print(f"Generating text with model {model} and config {kwargs}")
        prompts = kwargs["prompt"]

        if isinstance(prompts, str):
            prompts = [prompts]

        n = kwargs["n"]
        max_tokens = kwargs["max_tokens"]
        temperature = kwargs["temperature"]
        top_p = kwargs["top_p"]

        choices = []

        for prompt in prompts:
            
            inputs = tokenizer(
                prompt,
                return_tensors="pt"
            ).to(model.device)

            outputs = []
            with torch.no_grad():
                for _ in range(n):
                    outputs.append(model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=temperature > 0,
                        temperature=temperature,
                        top_p=top_p,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id
                    ))
            outputs = torch.cat(outputs, dim=0)
            #print outputs type and shape
            #print(f"Outputs type: {type(outputs)}")
            #print(f"Outputs shape: {outputs.shape}")
            for output in outputs:

                text = tokenizer.decode(
                    output,
                    skip_special_tokens=True
                )

                generated = text[len(prompt):]

                choices.append({
                    "text": generated
                })

        return {
            "choices": choices
        }

def model_from_config(config, disable_tqdm=True):
    """Returns a model based on the config."""
    model_type = config["name"]
    if model_type == "GPT_forward":
        return GPT_Forward(config, disable_tqdm=disable_tqdm)
    elif model_type == "GPT_insert":
        return GPT_Insert(config, disable_tqdm=disable_tqdm)
    raise ValueError(f"Unknown model type: {model_type}")


class LLM(ABC):
    """Abstract base class for large language models."""

    @abstractmethod
    def generate_text(self, prompt):
        """Generates text from the model.
        Parameters:
            prompt: The prompt to use. This can be a string or a list of strings.
        Returns:
            A list of strings.
        """
        pass

    @abstractmethod
    def log_probs(self, text, log_prob_range):
        """Returns the log probs of the text.
        Parameters:
            text: The text to get the log probs of. This can be a string or a list of strings.
            log_prob_range: The range of characters within each string to get the log_probs of. 
                This is a list of tuples of the form (start, end).
        Returns:
            A list of log probs.
        """
        pass


class GPT_Forward(LLM):
    """Wrapper for GPT-3."""

    def __init__(self, config, needs_confirmation=False, disable_tqdm=True):
        """Initializes the model."""
        torch.use_deterministic_algorithms(True)
        self.config = config
        self.needs_confirmation = needs_confirmation
        self.disable_tqdm = disable_tqdm

        #print(config)
        #print(self.config['gpt_config']['model'])

        MODEL_DIR = "/storage/fascetta/llama8"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['gpt_config']['model'],
            clean_up_tokenization_spaces=False,
            cache_dir=MODEL_DIR
        )

        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['gpt_config']['model'],
            dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb_config,
            cache_dir=MODEL_DIR
        )

        self.model.eval()

        self.best_train_perf = -50.0
        self.best_dev_perf = -50.0
        self.best_last_perf = 10
        self.best_prompt = None
        self.best_exemplars = {i: None for i in range(-50,-40)}
        self.best_exemplar_indices = {i: None for i in range(-50,-40)}
        self.best_instructions = {i: None for i in range(-50,-40)}
        self.num_call = 0
        self.best_instruction = None
        self.prompts_set = dict()

    def confirm_cost(self, texts, n, max_tokens):
        total_estimated_cost = 0
        for text in texts:
            total_estimated_cost += gpt_get_estimated_cost(
                self.config, text, max_tokens) * n
        print(f"Estimated cost: ${total_estimated_cost:.2f}")
        # Ask the user to confirm in the command line
        if os.getenv("LLM_SKIP_CONFIRM") is None:
            confirm = input("Continue? (y/n) ")
            if confirm != 'y':
                raise Exception("Aborted.")

    def auto_reduce_n(self, fn, prompt, n):
        """Reduces n by half until the function succeeds."""
        try:
            return fn(prompt, n)
        except BatchSizeException as e:
            if n == 1:
                raise e
            return self.auto_reduce_n(fn, prompt, n // 2) + self.auto_reduce_n(fn, prompt, n // 2)

    def generate_text(self, prompt, n):
        if not isinstance(prompt, list):
            prompt = [prompt]
        if self.needs_confirmation:
            self.confirm_cost(
                prompt, n, self.config['gpt_config']['max_tokens'])
        batch_size = self.config['batch_size']
        #print(f"Batch size: {batch_size}")
        #print(f"Number of prompts: {len(prompt)}")
        #print(f'prompts: {prompt}')
        prompt_batches = [prompt[i:i + batch_size]
                          for i in range(0, len(prompt), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, "
                f"split into {len(prompt_batches)} batches of size {batch_size * n}")
        text = []
        
        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            print(f"Generating text for batch of size {len(prompt_batch)}...")
            text += self.auto_reduce_n(self.__generate_text, prompt_batch, n)
        
        return text, self.tokenizer

    def complete(self, prompt, n):
        """Generates text from the model and returns the log prob data."""
        if not isinstance(prompt, list):
            prompt = [prompt]
        batch_size = self.config['batch_size']
        prompt_batches = [prompt[i:i + batch_size]
                          for i in range(0, len(prompt), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, "
                f"split into {len(prompt_batches)} batches of size {batch_size * n}")
        res = []
        
        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            res += self.__complete(prompt_batch, n)
        
        return res

    def log_probs(self, text, log_prob_range=None):
        
        return self.__log_probs(text, log_prob_range)

    def __generate_text(self, prompt, n):
        """Generates text from the model."""
        if not isinstance(prompt, list):
            text = [prompt]
        config = self.config['gpt_config'].copy()
        config['n'] = n

        # If there are any [APE] tokens in the prompts, remove them
        for i in range(len(prompt)):
            prompt[i] = prompt[i].replace('[APE]', '').strip()
        response = None
        config.pop("model", None)
        while response is None:
            try:
                response = HFCompletion.create(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    **config,
                    prompt=prompt
                )
                
            except Exception as e:
                if 'is greater than the maximum' in str(e):
                    raise BatchSizeException()
                print(e)
                print('Retrying...')
                time.sleep(5)
        
        return [response['choices'][i]['text'] for i in range(len(response['choices']))]

    def __complete(self, prompt, n):
        """Generates text from the model and returns the log prob data."""
        if not isinstance(prompt, list):
            text = [prompt]
        config = self.config['gpt_config'].copy()
        config['n'] = n
        # If there are any [APE] tokens in the prompts, remove them
        for i in range(len(prompt)):
            prompt[i] = prompt[i].replace('[APE]', '').strip()
        response = None
        while response is None:
            try:
                response = HFCompletion.create(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    **config,
                    prompt=prompt
                )
            except Exception as e:
                print(e)
                print('Retrying...')
                time.sleep(5)
        return response['choices']

    def __log_probs(self, text, log_prob_range=None):
        if not isinstance(text, list):
            text = [text]

        results_log_probs = []
        results_tokens = []

        self.tokenizer.pad_token = self.tokenizer.eos_token

        for idx, t in enumerate(text):
            
            enc = self.tokenizer(
                t,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                padding=False,
                add_special_tokens=True
            )

            offsets = enc.pop("offset_mapping")[0].tolist()

            enc = {
                k: v.to(self.model.device)
                for k, v in enc.items()
            }

            with torch.no_grad():
                outputs = self.model(**enc)
            
            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)

            input_ids = enc["input_ids"][0]

            token_log_probs = []
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

            # next-token log probabilities
            for i in range(len(input_ids) - 1):
                next_token_id = input_ids[i + 1]
                lp = log_probs[0, i, next_token_id].item()
                token_log_probs.append(lp)

            #
            # Align arrays exactly like OpenAI's token_logprobs[1:]
            #
            token_log_probs = token_log_probs
            tokens = tokens[1:]
            offsets = offsets[1:]

            if log_prob_range is not None:

                char_lower, char_upper = log_prob_range[idx]

                lower_token = 0

                for j, (start, end) in enumerate(offsets):
                    if start <= char_lower:
                        lower_token = j
                    else:
                        break

                upper_token = len(offsets)

                for j, (start, end) in enumerate(offsets):
                    if start >= char_upper:
                        upper_token = j
                        break

                token_log_probs = token_log_probs[lower_token:upper_token]
                tokens = tokens[lower_token:upper_token]

            results_log_probs.append(token_log_probs)
            results_tokens.append(tokens)

        return results_log_probs, results_tokens

    def get_token_indices(self, offsets, log_prob_range):
        """Returns the indices of the tokens in the log probs that correspond to the tokens in the log_prob_range."""
        # For the lower index, find the highest index that is less than or equal to the lower index
        lower_index = 0
        for i in range(len(offsets)):
            if offsets[i] <= log_prob_range[0]:
                lower_index = i
            else:
                break

        upper_index = len(offsets)
        for i in range(len(offsets)):
            if offsets[i] >= log_prob_range[1]:
                upper_index = i
                break

        return lower_index, upper_index
    
    def get_hidden_states(self, sentences, tokenizer=None, model=None):
        # Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            #print(f'token_embeddings: {token_embeddings.shape}')
            #print(f'attention_mask: {attention_mask.shape}')
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            #print(f'input_mask_expanded: {input_mask_expanded.shape}')
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        #print(f'len sentences: {len(sentences)}')

        gc.collect()
        torch.cuda.empty_cache()

        # Tokenize sentences
        if tokenizer is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
            tokenizer = self.tokenizer
            model = self.model
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        #if sentences[0] type is str, divide in batches of 10 and compute the embeddings for each batch, then concatenate the results
        if isinstance(sentences[0], str) and isinstance(sentences, list):
            #do batches of 10
            batch_size = 10
            #batch_size = len(sentences)
            sentences = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
  
            sentence_embeddings = []
            for i in range(len(sentences)):
                batch = sentences[i]
                #print(f'Batch length: {len(batch)}')
                #print(f'sentences length: {len(sentences)}')
                encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
                #print shapes
                #print(f'encoded_input input_ids: {encoded_input["input_ids"].shape}')
                #print(f'encoded_input attention_mask: {encoded_input["attention_mask"].shape}')
                #print(f'encoded_input token_type_ids: {encoded_input["token_type_ids"].shape}')

                with torch.no_grad():
                    model_output = model(**encoded_input.to('cuda'))
                batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                sentence_embeddings.append(batch_embeddings)
            sentence_embeddings = torch.cat(sentence_embeddings, dim=0)
        else:
            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input.to('cuda'))

            # Perform pooling
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

            # Normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings

    def eval_exemplars(self, instruction, selected_indices, n_shot_data, demos_template, eval_template, eval_data, few_shot_data, conf):
        self.num_call += 1
        
        #print('Instruction: {}'.format(instruction))
        n_shot_text = demos_template.fill(n_shot_data)
        
        if (instruction[0] + n_shot_text) in self.prompts_set.keys():
            (dev_perf, instruction_score) = self.prompts_set[instruction[0] + n_shot_text]
        else:
            #print(self.config)
            #print(f'eval data {eval_data}')
            dev_perf, instruction_score = likelihood.likelihood_evaluator(instruction, eval_template, eval_data, demos_template, few_shot_data, conf, self, demo_data=n_shot_data)
            print(dev_perf.sorted())
            dev_perf = dev_perf.sorted()[1][0]
            # print(answers)
            # import pdb; pdb.set_trace()
            self.prompts_set[instruction[0] + n_shot_text] = (dev_perf, instruction_score)

        '''
        if dev_perf >= self.best_last_perf:
            self.count += 1
        '''
        print(f'len selected_indices: {len(selected_indices)}')
        
        if dev_perf >= self.best_dev_perf:
            
            self.best_instruction = instruction

            del self.best_exemplar_indices[self.best_dev_perf]
            self.best_exemplar_indices[dev_perf] = selected_indices
            self.best_exemplar_indices = dict(sorted(self.best_exemplar_indices.items(), reverse=True))
            
            del self.best_exemplars[self.best_dev_perf]
            self.best_exemplars[dev_perf] = demos_template.fill(n_shot_data)
            self.best_exemplars = dict(sorted(self.best_exemplars.items(), reverse=True))

            del self.best_instructions[self.best_dev_perf]
            self.best_instructions[dev_perf] = instruction
            self.best_instructions = dict(sorted(self.best_instructions.items(), reverse=True))

            self.best_dev_perf = min(self.best_exemplar_indices)


        print('Dev loss: {}. Dev perf: {}. Best dev perf: {}'.format(
            round(float(dev_perf), 4),
            round(float(dev_perf), 4),
            round(float(self.best_dev_perf), 4)))
        print('********* Done *********')

        return dev_perf, instruction_score, instruction[0]

    def return_best_prompt(self):
        return self.best_instruction
    
    def return_best_exemplars(self):
        return self.best_exemplars

    def return_best_instructions(self):
        return self.best_instructions

    def return_best_exemplar_indices(self):
        return self.best_exemplar_indices

    def return_prompts_set(self):
        return self.prompts_set




class GPT_Insert(LLM):

    def __init__(self, config, needs_confirmation=False, disable_tqdm=True):
        """Initializes the model."""
        self.config = config
        self.needs_confirmation = needs_confirmation
        self.disable_tqdm = disable_tqdm

    def confirm_cost(self, texts, n, max_tokens):
        total_estimated_cost = 0
        for text in texts:
            total_estimated_cost += gpt_get_estimated_cost(
                self.config, text, max_tokens) * n
        print(f"Estimated cost: ${total_estimated_cost:.2f}")
        # Ask the user to confirm in the command line
        if os.getenv("LLM_SKIP_CONFIRM") is None:
            confirm = input("Continue? (y/n) ")
            if confirm != 'y':
                raise Exception("Aborted.")

    def auto_reduce_n(self, fn, prompt, n):
        """Reduces n by half until the function succeeds."""
        try:
            return fn(prompt, n)
        except BatchSizeException as e:
            if n == 1:
                raise e
            return self.auto_reduce_n(fn, prompt, n // 2) + self.auto_reduce_n(fn, prompt, n // 2)

    def generate_text(self, prompt, n):
        if not isinstance(prompt, list):
            prompt = [prompt]
        if self.needs_confirmation:
            self.confirm_cost(
                prompt, n, self.config['gpt_config']['max_tokens'])
        batch_size = self.config['batch_size']
        assert batch_size == 1
        prompt_batches = [prompt[i:i + batch_size]
                          for i in range(0, len(prompt), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, split into {len(prompt_batches)} batches of (maximum) size {batch_size * n}")
        text = []
        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            text += self.auto_reduce_n(self.__generate_text, prompt_batch, n)
        return text

    def log_probs(self, text, log_prob_range=None):
        raise NotImplementedError

    def __generate_text(self, prompt, n):
        """Generates text from the model."""
        config = self.config['gpt_config'].copy()
        config['n'] = n
        # Split prompts into prefixes and suffixes with the [APE] token (do not include the [APE] token in the suffix)
        prefix = prompt[0].split('[APE]')[0]
        suffix = prompt[0].split('[APE]')[1]
        response = None
        while response is None:
            try:
                response = HFCompletion.create(
                    **config, prompt=prefix, suffix=suffix)
            except Exception as e:
                print(e)
                print('Retrying...')
                time.sleep(5)

        # Remove suffix from the generated text
        texts = [response['choices'][i]['text'].replace(suffix, '') for i in range(len(response['choices']))]
        return texts


def gpt_get_estimated_cost(config, prompt, max_tokens):
    """Uses the current API costs/1000 tokens to estimate the cost of generating text from the model."""
    # Get rid of [APE] token
    prompt = prompt.replace('[APE]', '')
    # Get the number of tokens in the prompt
    n_prompt_tokens = len(prompt) // 4
    # Get the number of tokens in the generated text
    total_tokens = n_prompt_tokens + max_tokens
    engine = config['gpt_config']['model'].split('-')[1]
    costs_per_thousand = gpt_costs_per_thousand
    if engine not in costs_per_thousand:
        # Try as if it is a fine-tuned model
        engine = config['gpt_config']['model'].split(':')[0]
        costs_per_thousand = {
            'davinci': 0.1200,
            'curie': 0.0120,
            'babbage': 0.0024,
            'ada': 0.0016
        }
    price = costs_per_thousand[engine] * total_tokens / 1000
    return price


class BatchSizeException(Exception):
    pass
