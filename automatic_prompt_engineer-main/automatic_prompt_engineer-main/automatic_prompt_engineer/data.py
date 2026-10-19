import random
from datasets import load_from_disk

def load_dataset(path, split=None):
    ds = load_from_disk(path)
    if split:
        return ds[split]
    return ds

def subsample_data(data, subsample_size, mode = 'gen/eval', idxs = []):
    """
    Subsample data. Data is in the form of a tuple of lists.
    """
    inputs, outputs = data
    
    assert len(inputs) == len(outputs)

    indices = []
    
    if mode == 'few_shot':
        new_unfair_indices = []
        new_fair_indices = []

        size = int(min(subsample_size, len(idxs)))
        id = random.sample(idxs, size)
        new_unfair_indices.extend(id)

    elif mode == 'gen/eval':
        fair_indices = [i for i, label in enumerate(outputs) if label == 'fair']
        unfair_indices = [i for i in range(len(outputs)) if i not in fair_indices]
        
        if subsample_size > len(fair_indices) + len(unfair_indices):
            new_fair_indices = fair_indices
            new_unfair_indices = unfair_indices
            
        elif len(fair_indices) < int(subsample_size/2) and len(unfair_indices) >= int(subsample_size/2):
            new_fair_indices = fair_indices
            new_unfair_indices = random.sample(unfair_indices, subsample_size - len(fair_indices))

        elif len(unfair_indices) < int(subsample_size/2) and len(fair_indices) >= int(subsample_size/2):
            new_unfair_indices = unfair_indices
            new_fair_indices = random.sample(fair_indices, subsample_size - len(unfair_indices))

        else:
            new_fair_indices = random.sample(fair_indices, min(int(subsample_size/2), len(fair_indices)))
            new_unfair_indices = random.sample(unfair_indices, min(int(subsample_size/2), len(unfair_indices)))

    indices.extend(new_fair_indices)
    indices.extend(new_unfair_indices)
    
    random.shuffle(indices)

    inputs = [inputs[i] for i in indices]
    outputs = [outputs[i] for i in indices]

    return inputs, outputs, indices


def create_split(data, split_size):
    """
    Split data into two parts. Data is in the form of a tuple of lists.
    """
    inputs, outputs = data
    assert len(inputs) == len(outputs)
    indices = random.sample(range(len(inputs)), split_size)
    inputs1 = [inputs[i] for i in indices]
    outputs1 = [outputs[i] for i in indices]
    inputs2 = [inputs[i] for i in range(len(inputs)) if i not in indices]
    outputs2 = [outputs[i] for i in range(len(inputs)) if i not in indices]
    return (inputs1, outputs1), (inputs2, outputs2)
