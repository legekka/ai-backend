import torch
from modules.training import PTrainer
import modules.db_functions as dbf

discord_id = 148531387962490880

dataset = dbf.create_RPDataset(discord_id=discord_id)

from modules.utils import checkpoint_dataset_hash

model_dataset_hash = checkpoint_dataset_hash(f"models/RaterNNP_{discord_id}.pth")
dataset_hash = dataset.generate_hash()

if dataset_hash == model_dataset_hash:
    print("Dataset is up to date")
    exit()

trainer = PTrainer(dataset)
print("Dataset hash: " + dataset.generate_hash())

trainer.start_training()