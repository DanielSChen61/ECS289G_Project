# Many code snippets are brought over by the Hugging Face tutorial here: https://huggingface.co/blog/fine-tune-vit

from datasets import Dataset, DatasetDict
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split

import torch
import numpy as np
import glob
import random
import evaluate


def process_example(example):
    inputs = feature_extractor(example['image'], return_tensors='pt')
    inputs['labels'] = example['labels']
    return inputs

def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
    inputs['labels'] = example_batch['labels']
    return inputs

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

def compute_metrics(p):
    m1 = metric1.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids, average='macro')['precision']
    m2 = metric2.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids, average='macro')['recall']
    return {'precision': m1, 'recall': m2}


train_normal = glob.glob('../image_data/chest_xray/train/NORMAL/*.*')
train_bacteria = glob.glob('../image_data/chest_xray/train/PNEUMONIA_bacteria/*.*')
train_virus = glob.glob('../image_data/chest_xray/train/PNEUMONIA_virus/*.*')

test_normal = glob.glob('../image_data/chest_xray/test/NORMAL/*.*')
test_bacteria = glob.glob('../image_data/chest_xray/test/PNEUMONIA_bacteria/*.*')
test_virus = glob.glob('../image_data/chest_xray/test/PNEUMONIA_virus/*.*')

train_combined = train_normal + train_bacteria + train_virus
test_combined = test_normal + test_bacteria + test_virus

combined1 = []
combined2 = []

train_normal_labels = [0] * len(train_normal)
train_bacteria_labels = [1] * len(train_bacteria)
train_virus_labels = [2] * len(train_virus)
train_combined_labels = train_normal_labels + train_bacteria_labels + train_virus_labels

test_normal_labels = [0] * len(test_normal)
test_bacteria_labels = [1] * len(test_bacteria)
test_virus_labels = [2] * len(test_virus)
test_combined_labels = test_normal_labels + test_bacteria_labels + test_virus_labels

for file in train_combined:
    image = load_img(file, color_mode='rgb', target_size=(224, 224))
    combined1.append(image)

for file in test_combined:
    image = load_img(file, color_mode='rgb', target_size=(224, 224))
    combined2.append(image)

random.seed(1)
train_data_and_labels = list(zip(combined1, train_combined_labels))
test_data_and_labels = list(zip(combined2, test_combined_labels))
random.shuffle(train_data_and_labels)
random.shuffle(test_data_and_labels)

train_data, train_combined_labels = zip(*train_data_and_labels)
test_data, test_combined_labels = zip(*test_data_and_labels)

split_size = 0.2
train_data, val_data, train_combined_labels, val_combined_labels = train_test_split(train_data, train_combined_labels, test_size=split_size, random_state=1)

train_ds = Dataset.from_dict({"image": train_data, "labels": train_combined_labels})
val_ds = Dataset.from_dict({"image": val_data, "labels": val_combined_labels})
test_ds = Dataset.from_dict({"image": test_data, "labels": test_combined_labels})

ds = DatasetDict({
    'train': train_ds,
    'validation': val_ds,
    'test': test_ds
})

model_name_or_path = 'google/vit-base-patch16-224-in21k'
# model_name_or_path = 'google/vit-large-patch16-224'

feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

prepared_ds = ds.with_transform(transform)  

metric1 = evaluate.load('precision')
metric2 = evaluate.load('recall')

labels = ['normal', 'pneumonia_bacteria', 'pneumonia_virus']

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)

training_args = TrainingArguments(
  output_dir='./vit-base-xray',
  per_device_train_batch_size=32,
  evaluation_strategy='steps',
  num_train_epochs=14,
  fp16=True,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=8e-7,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds['train'],
    eval_dataset=prepared_ds['validation'],
    tokenizer=feature_extractor,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics('train', train_results.metrics)
trainer.save_metrics('train', train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(prepared_ds['test'])
trainer.log_metrics('eval', metrics)
trainer.save_metrics('eval', metrics)
