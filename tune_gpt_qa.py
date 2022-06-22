"""
Script for tuning GPT with pasted HF utils
Originally for DSC291 HW
"""
import time
import datetime
import os
import math
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from typing import Optional, Union
from datasets import Dataset
from copy import deepcopy
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase
from transformers import AutoTokenizer,AutoConfig
from transformers import Trainer
from transformers.trainer_utils import PredictionOutput,EvalLoopOutput
from transformers import EarlyStoppingCallback, TrainerCallback 

def has_tensor(obj) -> bool:
    """
    Given a possibly complex data structure,
    check if it has any torch.Tensors in it.
    Credit: AllenNLP
    """
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, dict):
        return any(has_tensor(value) for value in obj.values())
    elif isinstance(obj, (list, tuple)):
        return any(has_tensor(item) for item in obj)
    else:
        return False

def move_to_device(obj, cuda_device):
    """
    Given a structure (possibly) containing Tensors on the CPU,
    move all the Tensors to the specified GPU (or do nothing, if they should be on the CPU).
    Credit: AllenNLP
    """

    if cuda_device == torch.device("cpu") or not has_tensor(obj):
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.to(cuda_device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, cuda_device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, cuda_device) for item in obj]
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # This is the best way to detect a NamedTuple, it turns out.
        return obj.__class__(*(move_to_device(item, cuda_device) for item in obj))
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, cuda_device) for item in obj)
    else:
        return obj

class ProgressCallback(TrainerCallback):

    def setup(self, total_epochs, print_every=1): 
        self.total_epochs = total_epochs 
        self.current_epoch = 0
        self.epoch_start_time = None
        self.current_step = 1
        self.global_start_time = time.time()
        self.print_every=print_every
        return self

    def on_step_begin(self, args, state, control, **kwargs):
        
        avg_time_per_step = (time.time() - self.global_start_time)/max(state.global_step,1 )
        eta = avg_time_per_step * (state.max_steps-state.global_step) / 3600
        if self.current_step % self.print_every == 0:
            print(
                'epoch: ', 
                self.current_epoch,
                ', step ',
                self.current_step, 
                '/', 
                state.max_steps // self.total_epochs, 
                '||', 
                datetime.datetime.now(),
                '|| ETA(hrs): ',
                round(eta,2)
                )
        self.current_step += 1
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        print('[ProgressCallback]: current epoch: ', self.current_epoch,' / ', self.total_epochs)
        self.current_epoch += 1
        self.current_step = 1
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        print('[ProgressCallback]: epoch', self.current_epoch,' / ', self.total_epochs, ' done')
        print("--- %s hours ---" % ((time.time() - self.epoch_start_time)/3600) )

class PreprocessFunction:

    def __init__(self, tokenizer, input_ids_max_length):
        self.tokenizer = tokenizer 
        self.input_ids_max_length = input_ids_max_length

    def __call__(self, example):
        # Tokenize the texts
        example['target'] = example['query'].replace("[MASK]",example['answer'][0])
        example['text'] = '<|BOS|>'+example['question']+'<|SEP|>'+example['target']+'<|endoftext|>'
        result = self.tokenizer(
            example['text'], 
            padding=False, 
            max_length=self.input_ids_max_length,
            truncation=True
        )
        return result


@dataclass
class CustomDataCollator:

    tokenizer: PreTrainedTokenizerBase
    padding: bool = True
    input_ids_max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    history_pad_token_id: int = 0

    def __call__(self, features):
        # get padding side
        padding_side = self.tokenizer.padding_side
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.input_ids_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        features['labels'] = features['input_ids'].clone()
        features['labels'].masked_fill_(
            mask = torch.logical_not(features['attention_mask']),
            value = -100
        )
        return features
    
class CustomTrainer(Trainer):
    
    def prediction_step(self,model,inputs,prediction_loss_only, ignore_keys):
        with torch.no_grad():
            if torch.cuda.is_available():
                output = model(
                    **move_to_device(inputs, torch.device('cuda'))
                )
            else:
                output = model(**inputs)
        
        return output.loss.detach().cpu()

    def evaluation_loop(
            self,
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        ):
            """
            Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
            Works both with or without labels.
            Modify this for tracking multiple customed losses
            """
            args = self.args
            model = self._wrap_model(self.model, training=False)
            self.callback_handler.eval_dataloader = dataloader
            model.eval()
            batch_size = dataloader.batch_size
            num_examples = self.num_examples(dataloader)
            print(f"***** Running evaluation loop *****")
            print(f"  Num examples = {num_examples}")
            print(f"  Batch size = {batch_size}")
            loss_host = []
            
            for step, inputs in enumerate(dataloader):
                loss = self.prediction_step(
                    model, 
                    inputs, 
                    prediction_loss_only, 
                    ignore_keys=ignore_keys
                )
                
                loss_host += loss.repeat(batch_size).tolist()
                self.control = self.callback_handler.on_prediction_step(args, self.state, self.control) 
            loss_host = torch.tensor(loss_host)
            
            count=0
            total=0
            acc_count=0
            correct_data = []
#             for prmpt,target in tqdm(zip(self.source_eval,self.target_eval),total = len(self.target_eval),position=0, leave=True):
            
            selected = []
            for example in tqdm(self.eval_dataset,total = len(self.eval_dataset),position=0, leave=True):
                prmpt = '<|BOS|>'+example['question']+'<|SEP|>'
                target = example['target']+'<|endoftext|>'
                generated = torch.tensor(self.tokenizer.encode(prmpt)).unsqueeze(0)
                device = torch.device("cuda")
                generated = generated.to(device)
                output = model.generate(generated, 
#                                 do_sample=True,   
#                                 min_length=50, 
                                max_length=128,
                                num_beams=1,
                                #top_k=30,                                 
                                #top_p=0.7,        
                                temperature=0.9,
                                )
                text = self.tokenizer.decode(output[0]).split('<|SEP|>')[1]
                if count<10 and text!=target: #example['answer'][0] in text: #
                    print(f"\n text:{text} \n vs target:{target}")
                    count+=1
                if text==target:
                    acc_count+=1
#                     correct_data.append(prmpt+text)
                    selected.append(example)
                total+=1
            with open('correct_answered.jsonl','w') as fout:
                for item in selected:
                    fout.write(json.dumps(item)+'\n')
                fout.close()
            print(f"acc is {acc_count/total*1.0}")
                
                    
            metrics = {
                'eval_loss':torch.mean(loss_host).item()
            }
            print('[CustomTrainer]: Evaluation done)', metrics)
            output = EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_examples)
            return output

def run_gpt():
    import os
    import numpy as np
    from transformers import AutoTokenizer
    from transformers import GPT2LMHeadModel, TrainingArguments
    from datasets import load_dataset, Dataset

    # env vars
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["HF_DATASETS_CACHE"]="./huggingface_cache"
    os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    
    load_model = True
    load_model_path = './saved_gpt_models_sp/checkpoint-105471'
    
    add_special_token = True
    SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",                   
                    "sep_token": "<|SEP|>"}
    
    # load data and tokenize
    gpt_dataset = load_dataset(
        path='./kmir_with_questions',
        data_files={
            'train': 'kmir_dev_cloze.jsonl',
            'test': 'kmir_test_cloze.jsonl',
            'valid': 'kmir_train_cloze.jsonl'
#             'train': 'kmir_dev_cloze_sp.txt',
#             'test': 'kmir_test_cloze_sp.txt',
#             'valid': 'kmir_train_cloze_sp.txt'
            }
        )
    gpt_dataset = gpt_dataset.filter(lambda x:x['relType'] == 'Triple Completion')
    print(len(gpt_dataset['train']))
    print(len(gpt_dataset))
    print(len(gpt_dataset['valid']))
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if add_special_token:
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
        print("Special tokens added")
    tokenizer.pad_token = tokenizer.eos_token
    
    print('tokenizing')
    train_dataset = gpt_dataset['train'].map(
            PreprocessFunction(
                tokenizer,
                input_ids_max_length = 256
                ),
            batched=False,
            desc="Running tokenizer on dataset"
    )
    
    # note here we just use "test" for valid, since we don't need tuning
    valid_dataset = gpt_dataset['train'].map(
            PreprocessFunction(
                tokenizer,
                input_ids_max_length = 256
                ),
            batched=False,
            desc="Running tokenizer on dataset"
    )
    
    # load model and train 20 samples. eval on 10 samples
    if add_special_token:
        config = AutoConfig.from_pretrained('gpt2', 
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False)
    else: 
        config = AutoConfig.from_pretrained('gpt2',                                     
                                            pad_token_id=tokenizer.eos_token_id,
                                            output_hidden_states=False)
    if load_model:
        model = GPT2LMHeadModel.from_pretrained(load_model_path)
    else:
        model = GPT2LMHeadModel.from_pretrained('gpt2',config=config)
    if add_special_token:
        #Special tokens added, model needs to be resized accordingly
        model.resize_token_embeddings(len(tokenizer))

    
    trainer = CustomTrainer(
    model=model,
    train_dataset=train_dataset, 
    eval_dataset=train_dataset.shuffle(seed=42).select(range(5000)), 
    tokenizer=tokenizer,
    data_collator=CustomDataCollator(
            tokenizer=tokenizer,
            input_ids_max_length=256
        ),
        args=TrainingArguments(
            load_best_model_at_end = True,
            output_dir = './saved_gpt_models_sp',
            save_strategy = 'epoch',
            evaluation_strategy = 'epoch',
            per_device_train_batch_size=12,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            num_train_epochs=10,
            save_total_limit =3,
            gradient_accumulation_steps = 1,
            logging_steps=1000
        )
    )
#     import random
#     random.seed(42)
#     qa = list(map(lambda x:x.split('<|SEP|>'),random.sample(gpt_dataset['train']['text'],4000)))
#     trainer.source_eval = [x[0]+'<|SEP|>' for x in qa]
#     trainer.target_eval = [x[1] for x in qa]
#     trainer.tokenizer = tokenizer
    # evalute before tuning
#     print('perplexity before tuning')
#     print(np.exp(trainer.evaluate()['eval_loss']))
    
    # train
    #trainer.train()
    
    # evalute after tuning
    print('perplexity after tuning')
    print(np.exp(trainer.evaluate()['eval_loss']))

if __name__ == '__main__':
    run_gpt()

