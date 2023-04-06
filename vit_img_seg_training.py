from datasets import load_dataset
from datasets import load_metric
from transformers import ViTForImageClassification, ViTConfig
import torch
from transformers import ViTFeatureExtractor
from torchvision.transforms import Normalize, ToTensor, Compose
from transformers import TrainingArguments, Trainer
import numpy as np
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

def preprocess_train(example_batch):
    example_batch['pixel_values'] = [train_transforms(image.convert('RGB')) for image in example_batch['image']]
    return example_batch

def preprocess_val(example_batch):
    example_batch['pixel_values'] = [val_transforms(image.convert('RGB')) for image in example_batch['image']]
    return example_batch
    
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# load data
dataset = load_dataset("imagefolder", data_dir = "./train")
dataset = dataset.shuffle(seed=42)

# create helper dict
label2id, id2label = dict(), dict()
label2id['neg'] = 0
id2label[0] = 'neg'
label2id['pos'] = 1
id2label[1] = 'pos'

repo_name = './vit-finetuned'
feature_extractor = AutoFeatureExtractor.from_pretrained(repo_name)
model = AutoModelForImageClassification.from_pretrained(repo_name)
    
## load feature extractor
#feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch32-384')
#feature_extractor.size = (192, 32)

# transform and split data
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
train_transforms = Compose([ToTensor(), normalize,])
val_transforms = Compose([ToTensor(), normalize,])

splits = dataset['train'].train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

## define model
#state_dict = ViTForImageClassification.from_pretrained('google/vit-base-patch32-384', num_labels=2, ignore_mismatched_sizes=True).state_dict()
#state_dict['vit.embeddings.position_embeddings'] = torch.rand(1, 7, 768)
#
#configuration = ViTConfig(image_size=(192,32), patch_size=32, label2id=label2id, id2label=id2label)
#model = ViTForImageClassification(configuration)
#model.load_state_dict(state_dict)

# define metric
metric = load_metric("accuracy")

# training variables
batch_size = 64

args = TrainingArguments("vit-finetuned",
    remove_unused_columns=False,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# define trainer
trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

# train
train_results = trainer.train()
# rest is optional but nice to have
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

