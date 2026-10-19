import random
from automatic_prompt_engineer import generate, evaluate, config, template, data, llm
import time
import ot
from tqdm import tqdm
import random
import torch
from ease import LlamaForMLPRegression
import numpy as np
from transformers import AutoTokenizer, AutoModel

'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import nltk
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords
nltk.download('punkt_tab')
nltk.download('stopwords')
'''

tkwargs = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}

STOP_TRAINING_AFTER_ITER=2000

def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    return f"Set all the seeds to {seed} successfully!"

def preprocess_text(sentence):
    words = nltk.word_tokenize(sentence)
    ps = PorterStemmer()
    stemmed_cleaned_sentence = ''
    for word in words:
        cleaned_word = re.sub(r'[^a-zA-Z]', ' ', word.lower())
        cleaned_word = cleaned_word.split()
        stemmed_cleaned_word = [ps.stem(w) for w in cleaned_word if w and w not in set(stopwords.words('english'))]
        if stemmed_cleaned_word:
            stemmed_cleaned_sentence += ' ' + stemmed_cleaned_word[0]
    return stemmed_cleaned_sentence.strip()

def get_simple_prompt_gen_template(prompt_gen_template, prompt_gen_mode):
    if prompt_gen_template is None:
        if prompt_gen_mode == 'forward':
            prompt_gen_template = "I gave a friend an instruction. Based on the instruction they produced the following input-output pairs:\n\n[full_DEMO]\n\nThe instruction was to [APE]"
        elif prompt_gen_mode == 'insert':
            prompt_gen_template = None
        else:
            raise ValueError(
                'Invalid prompt_gen_mode: {}'.format(prompt_gen_mode))
    return prompt_gen_template


def simple_ape(dataset,
               eval_template='Instruction: [PROMPT]\nInput: [INPUT]\nOutput: [OUTPUT]',
               prompt_gen_template=None,
               demos_template='Input: [INPUT]\nOutput: [OUTPUT]',
               eval_model='meta-llama/Llama-3.1-8B-Instruct',
               prompt_gen_model='meta-llama/Llama-3.1-8B-Instruct',
               prompt_gen_mode='forward',
               num_prompts=50,
               eval_rounds=20,
               prompt_gen_batch_size=8,
               eval_batch_size=8):
    """
    Function that wraps the find_prompts function to make it easier to use.
    Design goals: include default values for most parameters, and automatically
    fill out the config dict for the user in a way that fits almost all use cases.

    The main shortcuts this function takes are:
    - Uses the same dataset for prompt generation, evaluation, and few shot demos
    - Uses UCB algorithm for evaluation
    - Fixes the number of prompts per round to num_prompts // 3  (so the first three rounds will
        sample every prompt once)
    - Fixes the number of samples per prompt per round to 5
    Parameters:
        dataset: The dataset to use for evaluation.
        eval_template: The template for the evaluation queries.
        prompt_gen_template: The template to use for prompt generation.
        demos_template: The template for the demos.
        eval_model: The model to use for evaluation.
        prompt_gen_model: The model to use for prompt generation.
        prompt_gen_mode: The mode to use for prompt generation.
        num_prompts: The number of prompts to generate during the search.
        eval_rounds: The number of evaluation rounds to run.
    Returns:
        An evaluation result and a function to evaluate the prompts with new inputs.
    """
    prompt_gen_template = get_simple_prompt_gen_template(
        prompt_gen_template, prompt_gen_mode)
    conf = config.simple_config(
        eval_model, prompt_gen_model, prompt_gen_mode, num_prompts, eval_rounds, prompt_gen_batch_size, eval_batch_size)
    
    return find_prompts(eval_template, demos_template, dataset, dataset, conf, prompt_gen_template=prompt_gen_template)


def simple_ape_split(train_set,
                    val_set,
                    eval_template='Instruction: [PROMPT]\nInput: [INPUT]\nOutput: [OUTPUT]',
                    num_samples = None,
                    num_demos = None,
                    prompt_gen_template=None,
                    demos_template='Input: [INPUT]\nOutput: [OUTPUT]',
                    model=None,
                    prompt_gen_mode='forward',
                    num_prompts=50,
                    eval_rounds=20,
                    prompt_gen_batch_size=1,
                    eval_batch_size=8):
    
    print(set_all_seed(0))
    
    prompt_gen_size = len(train_set[0])

    conf = config.simple_config()
    
    if num_samples is not None:
        conf['evaluation']['num_samples'] = num_samples
    
    val_inputs, val_outputs, val_idxs = data.subsample_data(val_set, conf['evaluation']['num_samples'], mode='gen/eval')
    val_set = (val_inputs, val_outputs)

    if num_demos != None:
        conf['generation']['num_demos'] = num_demos

    prompt_gen_template = get_simple_prompt_gen_template(
        prompt_gen_template, prompt_gen_mode)
    
    return find_prompts(eval_template, demos_template, train_set, val_set, conf, model, prompt_gen_template=prompt_gen_template)


def iterative_ape(eval_template,
                  eval_data,
                  train_data,
                  instructions,
                  model,
                  tokenizer,
                  few_shot_data = None,
                  demos_template='Input: [INPUT]\nOutput: [OUTPUT]',
                  resampling_template = 'Generate a variation of the following instruction while keeping the semantic meaning. Keep the words "fair" and "unfair"\nInput: [INSTRUCTION]\nOutput:'):
    
    demos_template = template.DemosTemplate(demos_template)
    eval_template = template.EvalTemplate(eval_template)
    conf = config.update_config({}, 'configs/default.yaml')
    
    queries = []
    for instruction in instructions:
        queries.append(resampling_template.replace('[INSTRUCTION]', instruction))
    
    response = None
    configs = conf['generation']['model']['gpt_config'].copy()
    configs['n'] = 1
    configs.pop("model", None)
    
    while response is None:
        try:
            response = llm.HFCompletion.create(
                model=model.model,
                tokenizer=tokenizer,
                **configs,
                prompt=queries
            )
                
        except Exception as e:
            if 'is greater than the maximum' in str(e):
                raise BatchSizeException()
            print(e)
            print('Retrying...')
            time.sleep(5)
    
    prompts = [response['choices'][i]['text'] for i in range(len(response['choices']))]

    if few_shot_data is None:
        few_shot_data = train_data

    res, _ = evaluate.evaluate_prompts(prompts, eval_template, eval_data, demos_template, few_shot_data,
                                   conf['evaluation']['method'], conf['evaluation'], model)
    
    return res


def simple_eval(train_set,
                val_set,
                prompts,
                num_samples = None,
                demo_data = None,
                eval_template='Instruction: [PROMPT]\nInput: [INPUT]\nOutput: [OUTPUT]',
                demos_template='Input: [INPUT]\nOutput: [OUTPUT]',
                eval_model='meta-llama/Llama-3.1-8B-Instruct'):
    """
    Function that wraps the evaluate_prompts function to make it easier to use.
    Parameters:
        val_set: The validation set to use for evaluation.
        labels_val: The labels for the validation set.
        prompts: The list of prompts to evaluate.
        eval_template: The template for the evaluation queries.
        demos_template: The template for the demos.
        eval_model: The model to use for evaluation.
    Returns:
        An evaluation result.
    """
    print(set_all_seed(0))

    eval_template = template.EvalTemplate(eval_template)
    demos_template = template.DemosTemplate(demos_template)
    conf = config.update_config({}, 'configs/default.yaml')
    conf['evaluation']['model']['gpt_config']['model'] = eval_model

    if num_samples is not None:
        conf['evaluation']['num_samples'] = num_samples

    res, _ = evaluate.evaluate_prompts(
        prompts, eval_template, val_set, demos_template, train_set, conf['evaluation']['method'], conf['evaluation'], model=eval_model, demo_data=demo_data)
    return res


def run_ease(prompt_gen_data,
                eval_data,
                conf,
                model_forward_api,
                lam = 0.1,
                nu = 0.01,
                NN_training_iter = 50,
                total_iter = 165,
                n_init = 40,
                n_domain = 10,
                prompts = [''],
                eval_template = '[full_DEMO]\n\nInput: [INPUT]\nOutput: [OUTPUT]',
                demos_template = 'Input: [INPUT]\nOutput: [OUTPUT]',
                sentence_embeddings_template = "Instruction: [PROMPT]\n\n[full_DEMO]\n\n",
                eval_model = 'meta-llama/Llama-3.1-8B-Instruct',
                ape = False):

    print(set_all_seed(0))

    #conf = config.update_config({}, 'configs/default.yaml')

    prompt_gen_size = len(prompt_gen_data[0])

    # Data is in the form input: single item, output: list of items
    # For prompt_gen_data, sample a single item from the output list
    #prompt_gen_data = prompt_gen_data[0], [random.sample(output, 1)[0] for output in prompt_gen_data[1]]
    
    eval_inputs, eval_outputs, eval_idxs = data.subsample_data(eval_data, conf['evaluation']['num_samples'], mode='gen/eval')
    eval_data = (eval_inputs, eval_outputs)

    d_template = template.DemosTemplate(demos_template)
    eval_template = template.EvalTemplate(eval_template)    
    #model_forward_api = llm.model_from_config(conf['generation']['model'], disable_tqdm=False)
    
    # We do not need the gradient for the weights
    for param in model_forward_api.model.parameters():
        param.requires_grad = False
    
    instruction_log = []
    r_log = []
    exemplars_log = []
    indices_log = []

    if ape:
        
        ape_idxs = [range(len(prompt_gen_data[0]))]

        # Generate prompts
        prompt_gen_template = get_simple_prompt_gen_template(None, 'forward')
        prompt_gen_template = template.GenerationTemplate(prompt_gen_template)

        print('Generating prompts...')
        generated_prompts, _, _, indices = generate.generate_prompts(
            prompt_gen_template, d_template, prompt_gen_data, conf['generation'], model_forward_api)

        ease_idxs = [i for i in range(prompt_gen_size) if (i not in indices and prompt_gen_data[1][i] == 'unfair')]

        print('Model returned {} prompts. Deduplicating...'.format(len(generated_prompts)))
        generated_prompts = list(set(generated_prompts))
        generated_prompts = [generated_prompt.strip() for generated_prompt in generated_prompts]
        print(f'len(generated_prompts): {len(generated_prompts)}')
        #print prompt type and shape
        print(f'Prompts type: {type(generated_prompts)}')
        print(f'Prompts length: {len(generated_prompts)}')
        print('Deduplicated to {} prompts.'.format(len(generated_prompts)))

    else:
        ease_idxs = [i for i in range(prompt_gen_size) if prompt_gen_data[1][i] == 'unfair']
        generated_prompts = prompts

    #tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    #model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased").to(**tkwargs)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(**tkwargs)
    
    with torch.no_grad():        
        
        # Generate initial indices for exemplars
        init_indices = []
        for _ in range(n_init):
            _, _, indices = data.subsample_data((prompt_gen_data[0], prompt_gen_data[1]), conf['evaluation']['num_few_shot'], 'few_shot', idxs = ease_idxs)
            init_indices.append(indices)
        #init_indices = np.stack(init_indices)
        init_indices = np.array(init_indices, dtype=object)

        demo_texts = []
        for i in range(len(init_indices)):
            n_shot_data = ([prompt_gen_data[0][j] for j in init_indices[i]], [prompt_gen_data[1][j] for j in init_indices[i]])
            n_shot_text = d_template.fill(n_shot_data)
            demo_texts.append(n_shot_text)
        
        if ape:
            sentences = []
            init_indices_for_prompt = np.random.choice(len(generated_prompts), n_init, replace=False if len(generated_prompts)>n_init else True)
            assert len(init_indices_for_prompt) == len(demo_texts)
            for i, prompt_index in enumerate(init_indices_for_prompt):
                sentence = sentence_embeddings_template.replace('[PROMPT]', generated_prompts[prompt_index]).replace('[full_DEMO]', demo_texts[i])
                print('i', i)
                print('prompt_index', prompt_index)
                print('sentence', sentence)
                sentences.append(sentence)
        else:
            sentences = []
            for demo_text in demo_texts:
                sentence = sentence_embeddings_template.replace('[PROMPT]', prompts[0]).replace('[full_DEMO]', demo_text)
                sentences.append(sentence)

        X_train = model_forward_api.get_hidden_states(sentences, tokenizer, model)
        #X_train = torch.cat(X_train).to(**tkwargs)

        # Evaluate the initial points
        Y = []
        pbar = tqdm(range(n_init))
        for i in pbar:
            
            n_shot_data = ([prompt_gen_data[0][j] for j in init_indices[i]], [prompt_gen_data[1][j] for j in init_indices[i]])
            n_shot_text = d_template.fill(n_shot_data)
            if ape:
                prompts = [generated_prompts[init_indices_for_prompt[i]]]

            r, _, instruction = model_forward_api.eval_exemplars(prompts, init_indices[i], n_shot_data, d_template, eval_template, eval_data, prompt_gen_data, conf['evaluation'])            
    
            indices_log.append(init_indices[i])
            exemplars_log.append(n_shot_text)
            instruction_log.append(instruction)
            r_log.append(r)
            Y.append(r)
            if r == 0:
                X_train = X_train[:len(Y)]
                break
        y_train = torch.FloatTensor(Y).unsqueeze(-1).to(**tkwargs)
        print(f"Best initial point: {y_train.max().item():.3f}")    

    best_r = y_train.max().item()
    best_init = best_r
    max_iter = total_iter - n_init

    # Start bayesian opt

    # lambda represent regularization, nu represent the trade-off, grid search
    l = LlamaForMLPRegression.NeuralTSDiag(input_dim=X_train.shape[1], lamdba=lam, nu=nu, init_x=X_train, init_y=y_train, style='ucb', diagonalize=True)
    l.train(None, None, NN_training_iter)

    # precompute the representation for all the examplar sentences
    all_examplars_sentences = [d_template.fill(([prompt_gen_data[0][i_]],[prompt_gen_data[1][i_]])) for i_ in range(len(prompt_gen_data[0]))]
    #all_examplars_representation = model_forward_api.get_hidden_states(all_examplars_sentences)
    
    # eval examplars representation
    eval_examplars_representation = model_forward_api.get_hidden_states([d_template.fill(([eval_data[0][i]],[eval_data[1][i][0]])) for i in range(len(eval_data[0]))], tokenizer, model)

    for t in range(max_iter):
        print(f"Iteration {t+1}/{max_iter}")
        if r == 0:
            break
        
        start_time = time.time()
        # Sample combinations
        #N_SAMPLE = n_domain
        N_SAMPLE = 50
        N_PROMPTS = 100
        N_SAMPLE_INSTRUCTIONS = 10
        
        batch_size = 50
        num_batches = N_SAMPLE // batch_size
        
        top_score = -50.0
        N_TIMES_DOMAIN = 6
        for _ in range(num_batches):
            print(f"Batch {_+1}/{num_batches}")
            with torch.no_grad():

                sampled_indices = []

                for _ in range(batch_size * N_TIMES_DOMAIN):
                    _, _, idxs = data.subsample_data((prompt_gen_data[0], prompt_gen_data[1]), conf['evaluation']['num_few_shot'], 'few_shot', idxs = ease_idxs)
                    sampled_indices.append(torch.tensor(idxs, device="cuda"))

                #sampled_indices = torch.stack(sampled_indices)

                # Extract embeddings and compute score
                
                ## sample the sentence based on DPP
                sentences = []
                all_ot_scores = []
                for i in range(len(sampled_indices)):
                    n_shot_data = ([prompt_gen_data[0][j.item()] for j in sampled_indices[i]], [prompt_gen_data[1][j.item()] for j in sampled_indices[i]])
                    n_shot_text = d_template.fill(n_shot_data)
                    
                    # compute the linear kernel matrix for the sampled indices
                    features_examplars = []

                    features_examplars = model_forward_api.get_hidden_states([all_examplars_sentences[j] for j in sampled_indices[i].tolist()], tokenizer, model)

                    # compute the similarity score to eval dataset using the optimal transport
                    distance_matrix = 1 - torch.mm(features_examplars, eval_examplars_representation.t()).cpu().numpy()
                    a, b = ot.unif(len(features_examplars)), ot.unif(len(eval_examplars_representation))  # uniform distribution on samples
                    ot_score = ot.emd(a, b, distance_matrix, log=True)[1]['cost']
                    all_ot_scores += [torch.tensor(ot_score)]

                    sentences.append(n_shot_text)
                # select the top determinants 
                top_indices = torch.stack(all_ot_scores).topk(batch_size, largest=False).indices
                sentences = [sentences[i] for i in top_indices]
                
                sampled_indices = [sampled_indices[i] for i in top_indices.tolist()]
                #sampled_indices = sampled_indices[top_indices]

                if ape:
                    ############## Adding Instructions here ##############
                    # Sample N_SAMPLE_INSTRUCTIONS prompts for each n_shot_data in the batch
                    indices_for_prompt = torch.ones(batch_size, len(generated_prompts)).multinomial(N_SAMPLE_INSTRUCTIONS, replacement=False).cuda()
                    # Get the sentences and compute embeddings
                    instructions_with_demos = []
                    corresponding_indices = []
                    corresponding_prompts = []
                    for i in range(len(indices_for_prompt)):
                        # For each n_shot_data (i.e., each sampled_indices)
                        sampled_prompts = [generated_prompts[j] for j in indices_for_prompt[i]]
                        for prompt in sampled_prompts:
                            sentence = sentence_embeddings_template.replace('[PROMPT]', prompt).replace('[full_DEMO]', sentences[i])
                            instructions_with_demos.append(sentence)
                            corresponding_indices.append(sampled_indices[i])
                            corresponding_prompts.append(prompt)
                    sentences = instructions_with_demos
                #print(f"Computing embeddings for {len(sentences)} sentences...")
                sentence_embeddings = model_forward_api.get_hidden_states(sentences, tokenizer, model)

            arm, best, _ = l.select(sentence_embeddings, batch_size=1)

            if best.item() > top_score:
                top_score = best.item()
                if ape:
                    best_prompt = corresponding_prompts[arm]
                    topk = corresponding_indices[arm]
                else:
                    topk = sampled_indices[arm]
            #print(f'best: {best.item()}')
        print('sampling used {} seconds'.format(time.time() - start_time))

        with torch.no_grad():
            print('topk', topk)
            
            n_shot_data = ([prompt_gen_data[0][j.item()] for j in topk], [prompt_gen_data[1][j.item()] for j in topk])
            n_shot_text = d_template.fill(n_shot_data)
            
            if ape:
                new_sentence = sentence_embeddings_template.replace('[PROMPT]', best_prompt).replace('[full_DEMO]', n_shot_text)
                print('new_sentence!!!', new_sentence)
            else:
                new_sentence = n_shot_text
            
            new_sentence_embeddings = model_forward_api.get_hidden_states(new_sentence, tokenizer, model)
            if ape:
                r, _, instruction = model_forward_api.eval_exemplars([best_prompt], topk, n_shot_data, d_template, eval_template, eval_data, prompt_gen_data, conf['evaluation'])
            else:
                r, _, instruction = model_forward_api.eval_exemplars(prompts, topk, n_shot_data, d_template, eval_template, eval_data, prompt_gen_data, conf['evaluation'])

            exemplars_log.append(n_shot_text)
            indices_log.append(','.join([str(z) for z in topk.clone().detach().cpu().tolist()]))
            
            print('Actual perfromance:', -r)

        if r > best_r:
            best_r = r
            if r >= 0:
                break

        if t < STOP_TRAINING_AFTER_ITER:
            l.train(new_sentence_embeddings, r, NN_training_iter)
            
        instruction_log.append(instruction)
        r_log.append(r)
        
        print("iter {0} --- reward: {1}".format(t, r))
        print(f"Best value found till now: {best_r}")
        
    print('Evaluate on test data...')
    prompts = model_forward_api.return_best_prompt()
    print("Best instruction is:")
    print(prompts)

    prompts_set = model_forward_api.return_prompts_set()
    print("The final instruction set is:")
    print(model_forward_api.return_prompts_set())
    
    best_exemplars = model_forward_api.return_best_exemplars()
    print("The best exemplars are:")
    print(best_exemplars)
    
    best_exemplar_indices = model_forward_api.return_best_exemplar_indices()
    print("The best exemplar indices are:")
    print(best_exemplar_indices)
    
    best_instructions = model_forward_api.return_best_instructions()
    print("The best instructions are:")
    print(best_instructions)

    try:
        if type(best_exemplars[0].item()) == int:
            best_exemplars = [d_template.fill([[prompt_gen_data[0][i]], [prompt_gen_data[1][i]]]) + '\n\n' for i in best_exemplars]
    except Exception as err:
        print(err)

    return prompts, prompts_set, best_exemplars, best_exemplar_indices, best_instructions, instruction_log, r_log, exemplars_log, indices_log, best_r, best_init


def simple_estimate_cost(dataset,
                         eval_template='Instruction: [PROMPT]\nInput: [INPUT]\nOutput: [OUTPUT]',
                         prompt_gen_template=None,
                         demos_template='Input: [INPUT]\nOutput: [OUTPUT]',
                         eval_model='meta-llama/Llama-3.1-8B-Instruct',
                         prompt_gen_model='meta-llama/Llama-3.1-8B-Instruct',
                         prompt_gen_mode='forward',
                         num_prompts=50,
                         eval_rounds=20,
                         prompt_gen_batch_size=8,
                         eval_batch_size=8):
    prompt_gen_template = get_simple_prompt_gen_template(
        prompt_gen_template, prompt_gen_mode)
    conf = config.simple_config(
        eval_model, prompt_gen_model, prompt_gen_mode, num_prompts, eval_rounds, prompt_gen_batch_size, eval_batch_size)

    return estimate_cost(eval_template, demos_template, dataset, dataset, conf, prompt_gen_template=prompt_gen_template)


def find_prompts(eval_template,
                 demos_template,
                 prompt_gen_data,
                 eval_data,
                 conf,
                 model = None,
                 base_conf='configs/default.yaml',
                 few_shot_data=None,
                 prompt_gen_template=None):
    """
    Function to generate prompts using APE.
    Parameters:
        eval_template: The template for the evaluation queries.
        demos_template: The template for the demos.
        prompt_gen_data: The data to use for prompt generation.
        eval_data: The data to use for evaluation.
        conf: The configuration dictionary.
        few_shot_data: The data to use for demonstrations during eval (not implemented yet).
        eval_method: The evaluation method to use. ('likelihood')
        prompt_gen_template: The template to use for prompt generation.
        verbosity: The verbosity level.
    Returns:
        An evaluation result. Also returns a function to evaluate the prompts with new inputs.
    """

    #conf = config.update_config(conf, base_conf)
    #print(f'base_conf: {base_conf}')
    #print(f"conf2 updated: {conf}")

    # Generate prompts
    eval_template = template.EvalTemplate(eval_template)
    demos_template = template.DemosTemplate(demos_template)
    if prompt_gen_template is None:
        prompt_gen_template = eval_template.convert_to_generation_template()
    else:
        prompt_gen_template = template.GenerationTemplate(prompt_gen_template)

    if few_shot_data is None:
        few_shot_data = prompt_gen_data
    
    print('Generating prompts...')
    prompts, model, tokenizer, _ = generate.generate_prompts(
        prompt_gen_template, demos_template, prompt_gen_data, conf['generation'], model)
    
    with open("generated_prompts.txt", "w") as f:
        f.write(str(prompts))
    
    print('Model returned {} prompts. Deduplicating...'.format(len(prompts)))
    prompts = list(set(prompts))
    #print prompt type and shape
    print(f'Prompts type: {type(prompts)}')
    print(f'Prompts length: {len(prompts)}')
    print('Deduplicated to {} prompts.'.format(len(prompts)))

    print('Evaluating prompts...')
    #print(f'conf evaluation: {conf["evaluation"]}')
    res, __ = evaluate.evaluate_prompts(prompts, eval_template, eval_data, demos_template, few_shot_data, conf['evaluation']['method'], conf['evaluation'], model)

    print('Finished evaluating.')

    demo_fn = evaluate.demo_function(eval_template, conf['demo'], model)

    return res, demo_fn, model, tokenizer


def evaluate_prompts(prompts, eval_template, eval_data, demos_template, few_shot_data, conf,
                     base_conf='configs/default.yaml'):
    """
    Function to evaluate a list of prompts.
    Parameters:
        prompts: The list of prompts to evaluate.
        eval_template: The template for the evaluation queries.
        eval_data: The data to use for evaluation.
        eval_method: The evaluation method to use. ('likelihood')
        conf: The configuration dictionary.
        base_conf: The base configuration file.
    Returns:
        A list of prompts and their scores, sorted by score.
    """

    conf = config.update_config(conf, base_conf)

    # Generate prompts
    eval_template = template.EvalTemplate(eval_template)
    demos_template = template.DemosTemplate(demos_template)

    print('Evaluating prompts...')
    res, _ = evaluate.evaluate_prompts(
        prompts, eval_template, eval_data, demos_template, few_shot_data, conf['evaluation']['method'],
        conf['evaluation'])

    print('Finished evaluating.')

    return res


def estimate_cost(eval_template,
                  demos_template,
                  prompt_gen_data,
                  eval_data,
                  conf,
                  base_conf='configs/default.yaml',
                  few_shot_data=None,
                  prompt_gen_template=None,
                  eval_query=None):
    conf = config.update_config(conf, base_conf)

    max_prompt_len = conf['generation']['model']['gpt_config']['max_tokens']
    num_prompts = conf['generation']['num_prompts_per_subsample'] * \
                  conf['generation']['num_subsamples']
    eval_method = conf['evaluation']['method']

    if eval_method == 'bandits':
        num_prompts_per_round = conf['evaluation']['num_prompts_per_round']
        if num_prompts_per_round < 1:
            num_prompts_per_round = int(
                num_prompts * num_prompts_per_round)
        num_evals = conf['evaluation']['rounds'] * \
                    num_prompts_per_round * \
                    conf['evaluation']['base_eval_config']['num_samples']
    else:
        num_evals = conf['evaluation']['num_samples'] * num_prompts

    # Compute cost of prompt generation
    queries = get_generation_query(
        eval_template, demos_template, conf, prompt_gen_data, prompt_gen_template, num_query=50)

    query_cost = 0
    for query in queries:
        query_cost += llm.gpt_get_estimated_cost(
            conf['generation']['model'], query, max_prompt_len)

    total_query_cost = query_cost / len(queries) * num_prompts

    # Compute cost of evaluation
    if few_shot_data is None:
        few_shot_data = prompt_gen_data

    queries = get_evaluation_query(
        eval_template, demos_template, conf, eval_data, few_shot_data, eval_query, num_query=50)

    if conf['evaluation']['method'] == 'bandits':
        model_name = conf['evaluation']['base_eval_config']['model']
    else:
        model_name = conf['evaluation']['model']

    query_cost = 0
    for query in queries:
        query_cost += llm.gpt_get_estimated_cost(model_name, query, 0)

    total_eval_cost = query_cost / len(queries) * num_evals

    return total_query_cost + total_eval_cost


def get_generation_query(eval_template,
                         demos_template,
                         conf,
                         prompt_gen_data,
                         prompt_gen_template=None,
                         num_query=1):
    # Generate prompts
    eval_template = template.EvalTemplate(eval_template)
    demos_template = template.DemosTemplate(demos_template)
    if prompt_gen_template is None:
        prompt_gen_template = eval_template.convert_to_generation_template()
    else:
        prompt_gen_template = template.GenerationTemplate(prompt_gen_template)

    # First, generate a few prompt queries:
    queries = []
    for _ in range(num_query):
        subsampled_data = data.subsample_data(
            prompt_gen_data, conf['generation']['num_demos'])
        queries.append(generate.get_query(prompt_gen_template,
                                          demos_template, subsampled_data))

    return queries


def get_evaluation_query(eval_template,
                         demos_template,
                         conf,
                         eval_data,
                         few_shot_data,
                         eval_query=None,
                         num_query=1
                         ):
    eval_template = template.EvalTemplate(eval_template)
    demos_template = template.DemosTemplate(demos_template)

    if conf['evaluation']['method'] == 'bandits':
        eval_base_method = conf['evaluation']['base_eval_method']
        num_few_shot = conf['evaluation']['base_eval_config']['num_few_shot']
    else:
        eval_base_method = conf['evaluation']['method']
        num_few_shot = conf['evaluation']['num_few_shot']

    if eval_query is None:
        if eval_base_method == 'likelihood':
            from automatic_prompt_engineer.evaluation import likelihood
            eval_query = likelihood.get_query
        else:
            raise ValueError(
                'Cannot estimate costs for: {}'.format(eval_base_method))

    max_prompt_len = conf['generation']['model']['gpt_config']['max_tokens']
    filler_prompt = 'GGGG' * max_prompt_len

    queries = []
    for _ in range(num_query):
        idx = random.randint(0, len(eval_data[0]) - 1)
        input_, output_ = eval_data[0][idx], eval_data[1][idx]
        demo_data = data.subsample_data(few_shot_data, num_few_shot)
        query = eval_query(filler_prompt, eval_template, input_,
                           output_, demo_data, demos_template)[0]
        queries.append(query)
    return queries
