import os
import sys
import random
from argparse import Namespace
import logging

from tqdm import tqdm
import torch as T
import torch.nn as nn
import numpy as np

#sys.path.append('../')
from torch.utils.data import DataLoader

from data.card import TransactionDataset
from models import CLTrial_Bert
from scripts.utils import random_split_dataset
import data.datacollator as datacoll



logging.basicConfig(
                    level = logging.INFO,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger(__name__)


config = vars(Namespace(cached=False, checkpoint=0, data_extension='', data_fname='card_transaction.v2', data_root='./data/credit_card/', data_type='card', do_eval=False, do_train=True, field_ce=True, field_hs=64, flatten=False, jid=1, lm_type='bert', log_dir='sam/logs', mlm=True, mlm_prob=0.15, nrows=None, num_train_epochs=3, output_dir='sam', save_steps=500, seed=9, skip_user=False, stride=5, user_ids=None, vocab_file='vocab.nb'))
config['data_root'] = "./dataset/credit_card/"
config['output_dir'] = "sample"
config['log_dir'] = "sample/logs"
os.makedirs(config['output_dir'], exist_ok=True)
os.makedirs(config['log_dir'], exist_ok=True)

seed = config['seed']
random.seed(seed)  # python
np.random.seed(seed)  # numpy
T.manual_seed(seed)  # torch
if T.cuda.is_available():
    T.cuda.manual_seed_all(seed)  # torch.cuda

dataset = TransactionDataset(root=config['data_root'],
                            fname=config['data_fname'],
                            fextension="",
                            vocab_dir=config['output_dir'],
                            nrows=None,
                            user_ids=None,
                            seq_len=20,
                            mlm=True,
                            cached=config['cached'],
                            stride=10,
                            flatten=config['flatten'],
                            return_labels=True,
                            skip_user=True)

vocab = dataset.vocab
custom_special_tokens = vocab.get_special_tokens()

totalN = len(dataset)
totalN = len(dataset)
trainN = int(0.6 * totalN)

valtestN = totalN - trainN
valN = int(valtestN * 0.5)
testN = valtestN - valN
lengths = [trainN, valN, testN]
log.info(f"# lengths: train [{trainN}]  valid [{valN}]  test [{testN}]")
log.info("# lengths: train [{:.2f}]  valid [{:.2f}]  test [{:.2f}]".format(trainN / totalN, valN / totalN,
                                                                               testN / totalN))
train_dataset, eval_dataset, test_dataset = random_split_dataset(dataset, lengths)

tab_net = CLTrial_Bert.ClTrial(custom_special_tokens,
                                  vocab=vocab,
                                  field_ce=config['field_ce'],
                                  flatten=config['flatten'],
                                  ncols=dataset.ncols,
                                  field_hidden_size=config['field_hs']
                                  )
collactor_cls = "TransDataCollatorForLanguageModeling"
data_collator = datacoll.TransDataCollatorForLanguageModeling(
        tokenizer=tab_net.tokenizer, mlm=True, mlm_probability=config['mlm_prob']
    ) 
train_dataloader = DataLoader(
            train_dataset,
            batch_size=10,
            collate_fn=data_collator)

model = tab_net.model
optim_params = {'betas': (0.9, 0.999), 'eps': 1e-08, 'lr': 5e-05}
optim = T.optim.AdamW(model.parameters(), **optim_params)

total_loss = 0
for inps in tqdm(train_dataloader):
    log.debug(inps.keys())
    log.debug(inps['Ouput'])

    log.debug(inps.keys())
    log.debug(inps['input_ids'].shape)
    log.debug(inps['masked_lm_labels'].shape)
    log.debug(inps['masked_lm_labels'], )
    optim.zero_grad()
    log.debug(inps['input_ids'].shape)
    labels = inps.pop("Ouput")
    model.train()
    inps['input_ids'] = inps['input_ids']
    inps['masked_lm_labels'] = inps['masked_lm_labels']
    outputs =model(**inps)
    log.debug('out',outputs)
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    log.debug(loss)
    loss.backward()
    optim.step()
    total_loss += loss.item()

tot_loss = total_loss/len(train_dataloader)
print('Training Loss',tot_loss)
