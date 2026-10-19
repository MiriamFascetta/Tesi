from faulthandler import disable
from automatic_prompt_engineer import data, llm
import random

def get_query(prompt_gen_template, demos_template, subsampled_data):
    """
    Returns a query for the prompt generator. A query is the prompt that is sent to the LLM.
    Parameters:
        prompt_gen_template: The template for the prompt generator queries.
        demos_template: The template for the demonstrations.
        subsampled_data: The data to use for the demonstrations.
    Returns:
        A query for the prompt generator.
    """
    inputs, outputs = subsampled_data
    demos = demos_template.fill(subsampled_data)
    return prompt_gen_template.fill(input=inputs[0], output=outputs[0], full_demo=demos)


def generate_prompts(prompt_gen_template, demos_template, prompt_gen_data, config, model = None):
    """
    Generates prompts using the prompt generator.
    Parameters:
        prompt_gen_template: The template for the prompt generator queries.
        demos_template: The template for the demonstrations.
        prompt_gen_data: The data to use for prompt generation.
        labels_train: The labels for the training data.
        config: The configuration dictionary.
    Returns:
        A list of prompts.
    """
    queries = []

    indices = []
    for _ in range(config['num_subsamples']):
        print(f"Generating prompts for subsample {_+1}/{config['num_subsamples']}...")
        inputs, outputs, idxs = data.subsample_data(prompt_gen_data, config['num_demos'], mode='gen/eval')
        
        subsampled_data = (inputs, outputs)
        indices.extend(idxs)
        #print(subsampled_data)
        queries.append(get_query(prompt_gen_template, demos_template, subsampled_data))
    #print(queries)

    # Instantiate the LLM
    if model == None:
        model = llm.model_from_config(config['model'], disable_tqdm=False)

    prompts, tokenizer = model.generate_text(
        queries, n=config['num_prompts_per_subsample'])
    
    return prompts, model, tokenizer, indices
