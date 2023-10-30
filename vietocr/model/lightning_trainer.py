from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS, OptimizerLRScheduler
from vietocr.optim.labelsmoothingloss import LabelSmoothingLoss
from torch.optim import Adam, SGD, AdamW
from vietocr.tool.translate import translate, batch_translate_beam_search
from vietocr.tool.utils import download_weights

import random
import torch
from vietocr.loader.dataloader import Collator
from torch.utils.data import DataLoader
import torch.optim as optim

from vietocr.tool.utils import compute_accuracy
from vietocr.optim.scheduler import CosineAnealingWarmRestartsWeightDecay
import numpy as np
import os
import pytorch_lightning as pl
import wandb


def seed_worker(worker_id):
    np.random.seed(worker_id * 10 + 42)
    random.seed(worker_id * 10 + 42)


class LightningTrainer(pl.LightningModule):
    def __init__(self, config, model, vocab, trainset, validset, pretrained=True):
        super().__init__()
        self.config = config
        self.trainset = trainset
        self.validset = validset
        self.model = model
        self.vocab = vocab
        self.num_iters = config['trainer']['iters']
        self.beamsearch = config['predictor']['beamsearch']
        self.num_workers = self.config['dataloader'].pop('num_workers') if 'num_workers' in self.config['dataloader'] else os.cpu_count()
        self.validation_step_outputs = []

        self.metrics = config['trainer']['metrics']
        if pretrained:
            weight_file = download_weights(config['pretrain'], quiet=config['quiet'])
            self.load_weights(weight_file)

        self.criterion = LabelSmoothingLoss(len(self.vocab), padding_idx=self.vocab.pad, smoothing=0.1)

    def forward(self, img, tgt_input, tgt_padding_mask):
        return self.model(img, tgt_input, tgt_key_padding_mask=tgt_padding_mask)

    def load_weights(self, filename):
        saved_state_dict = torch.load(filename, map_location=torch.device(self.device))
        model_state_dict = self.model.state_dict()
        state_dict = {}

        for name, param in model_state_dict.items():
            if name not in saved_state_dict:
                print('{} not found'.format(name))
            elif saved_state_dict[name].shape != param.shape:
                print('{} missmatching shape, required {} but found {}'.format(name, param.shape, saved_state_dict[name].shape))
            else:
                state_dict[name] = saved_state_dict[name]

        self.model.load_state_dict(state_dict, strict=False)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = AdamW(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, total_steps=self.num_iters, **self.config['optimizer'])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
                'name': 'train/learning_rate'
            }
        }

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        g = torch.Generator()
        g.manual_seed(self.config['trainer']['seed'])
        return DataLoader(
            self.trainset,
            batch_size=self.config['trainer']['batch_size'],
            num_workers=self.num_workers,
            collate_fn=Collator(self.config['aug']['masked_language_model']),
            worker_init_fn=seed_worker,
            shuffle=False,
            drop_last=False,
            **self.config['dataloader']
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.validset,
            batch_size=self.config['trainer']['valid_batch_size'],
            num_workers=self.num_workers,
            collate_fn=Collator(False),
            worker_init_fn=seed_worker,
            shuffle=False,
            drop_last=False,
            **self.config['dataloader']
        )

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(self.config)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        img = batch['img']
        tgt_input = batch['tgt_input']
        tgt_output = batch['tgt_output']
        tgt_padding_mask = batch['tgt_padding_mask']

        outputs = self(img, tgt_input, tgt_padding_mask)
        outputs = outputs.view(-1, outputs.size(2))
        tgt_output = tgt_output.view(-1)
        loss = self.criterion(outputs, tgt_output)
        self.log('train/loss', loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        img = batch['img']
        tgt_input = batch['tgt_input']
        tgt_output = batch['tgt_output']
        tgt_padding_mask = batch['tgt_padding_mask']

        outputs = self(img, tgt_input, tgt_padding_mask)
        outputs = outputs.flatten(0, 1)
        tgt_output = tgt_output.flatten()
        loss = self.criterion(outputs, tgt_output)

        if self.beamsearch:
            translated_sentence = batch_translate_beam_search(batch['img'], self.model)
        else:
            translated_sentence, prob = translate(batch['img'], self.model)
        pred_sent = self.vocab.batch_decode(translated_sentence.tolist())
        actual_sent = self.vocab.batch_decode(batch['tgt_output'].tolist())

        table_row = None
        if len(self.validation_step_outputs) < 10:
            table_row = [batch['filenames'][0], img[0].cpu().numpy().transpose(1, 2, 0), actual_sent[0], pred_sent[0]]

        self.validation_step_outputs.append({
            'loss': loss.item(),
            'pred_sent': pred_sent,
            'actual_sent': actual_sent,
            'table_row': table_row
        })

        return loss

    def on_validation_epoch_end(self) -> None:
        val_losses = torch.Tensor([out['loss'] for out in self.validation_step_outputs])
        pred_sents = sum([out['pred_sent'] for out in self.validation_step_outputs], [])
        actual_sents = sum([out['actual_sent'] for out in self.validation_step_outputs], [])
        acc_full_seq = compute_accuracy(actual_sents, pred_sents, mode='full_sequence')
        acc_per_char = compute_accuracy(actual_sents, pred_sents, mode='per_char')
        table_rows = [out['table_row'] for out in self.validation_step_outputs if out['table_row'] is not None]
        for i in range(len(table_rows)):
            table_rows[i][1] = wandb.Image(table_rows[i][1])
        self.log_dict({
            'val/loss': val_losses.mean().item(),
            'val/acc_full_seq': acc_full_seq,
            'val/acc_per_char': acc_per_char,
        }, prog_bar=True, logger=True, sync_dist=True)
        self.logger.log_table(key="valid samples", columns=["filenames", "image", "label", "pred"], data=table_rows)
