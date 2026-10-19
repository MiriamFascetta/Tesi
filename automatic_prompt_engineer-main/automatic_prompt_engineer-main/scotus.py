from datasets import load_dataset


train_dataset = load_dataset("pasinit/scotus", revision="refs/convert/parquet")
print(train_dataset["train"][1])