import argparse
import os
import time

import torch
from pytorch_lightning import seed_everything

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

from log import logger
from model.ner_model import NERBaseAnnotator
from utils.reader import CoNLLReader

conll_iob = {'B-ORG': 0, 'I-ORG': 1, 'B-MISC': 2, 'I-MISC': 3, 'B-LOC': 4, 'I-LOC': 5, 'B-PER': 6, 'I-PER': 7, 'O': 8}
wnut_iob = {'B-CORP': 0, 'I-CORP': 1, 'B-CW': 2, 'I-CW': 3, 'B-GRP': 4, 'I-GRP': 5, 'B-LOC': 6, 'I-LOC': 7, 'B-PER': 8, 'I-PER': 9, 'B-PROD': 10, 'I-PROD': 11, 'O': 12}
resume_iob = {'M-RACE': 0,  'B-PRO': 1,  'S-ORG': 2,  'B-LOC': 3,  'B-CONT': 4,  'M-CONT': 5,  'E-LOC': 6,  'M-PRO': 7,  'M-LOC': 8,  'M-TITLE': 9,  'B-ORG': 10,  'M-ORG': 11,  'E-ORG': 12,  'E-RACE': 13,  'B-EDU': 14,  'S-NAME': 15,  'B-TITLE': 16,  'S-RACE': 17,  'B-NAME': 18,  'B-RACE': 19,  'E-NAME': 20,  'O': 21,  'E-CONT': 22,  'M-EDU': 23,  'E-TITLE': 24,  'E-EDU': 25,  'M-NAME': 26,  'E-PRO': 27}
weibo_iob = {'O': 0,  'B-PER.NOM': 1,  'E-PER.NOM': 2,  'B-LOC.NAM': 3,  'E-LOC.NAM': 4,  'B-PER.NAM': 5,  'M-PER.NAM': 6,  'E-PER.NAM': 7,  'S-PER.NOM': 8,  'B-GPE.NAM': 9,  'E-GPE.NAM': 10,  'B-ORG.NAM': 11,  'M-ORG.NAM': 12,  'E-ORG.NAM': 13,  'M-PER.NOM': 14,  'S-GPE.NAM': 15,  'B-ORG.NOM': 16,  'E-ORG.NOM': 17,  'M-LOC.NAM': 18,  'M-ORG.NOM': 19,  'B-LOC.NOM': 20,  'M-LOC.NOM': 21,  'E-LOC.NOM': 22,  'B-GPE.NOM': 23,  'E-GPE.NOM': 24,  'M-GPE.NAM': 25,  'S-PER.NAM': 26,  'S-LOC.NOM': 27}
msra_iob = {'O': 0,  'S-NS': 1,  'B-NS': 2,  'E-NS': 3,  'B-NT': 4,  'M-NT': 5,  'E-NT': 6,  'M-NS': 7,  'B-NR': 8,  'M-NR': 9,  'E-NR': 10,  'S-NR': 11,  'S-NT': 12}
ontonotes_iob = {'E-PER': 0,  'E-GPE': 1,  'E-LOC': 2,  'M-ORG': 3,  'E-ORG': 4,  'S-ORG': 5,  'B-GPE': 6,  'O': 7,  'M-PER': 8,  'M-LOC': 9,  'B-PER': 10,  'M-GPE': 11,  'S-LOC': 12,  'B-ORG': 13,  'S-PER': 14,  'B-LOC': 15,  'S-GPE': 16}

def parse_args():
    p = argparse.ArgumentParser(description='Model configuration.', add_help=False)
    p.add_argument('--train', type=str, help='Path to the train data.', default=None)
    p.add_argument('--test', type=str, help='Path to the test data.', default=None)
    p.add_argument('--dev', type=str, help='Path to the dev data.', default=None)

    p.add_argument('--out_dir', type=str, help='Output directory.', default='.')
    p.add_argument('--iob_tagging', type=str, help='IOB tagging scheme', default='wnut')

    p.add_argument('--max_instances', type=int, help='Maximum number of instances', default=-1)
    p.add_argument('--max_length', type=int, help='Maximum number of tokens per instance.', default=50)

    p.add_argument('--encoder_model', type=str, help='Pretrained encoder model to use', default='xlm-roberta-large')
    p.add_argument('--model', type=str, help='Model path.', default=None)
    p.add_argument('--model_name', type=str, help='Model name.', default=None)
    p.add_argument('--stage', type=str, help='Training stage', default='fit')
    p.add_argument('--prefix', type=str, help='Prefix for storing evaluation files.', default='test')

    p.add_argument('--batch_size', type=int, help='Batch size.', default=128)
    p.add_argument('--gpus', type=int, help='Number of GPUs.', default=1)
    p.add_argument('--cuda', type=str, help='Cuda Device', default='cuda:0')
    p.add_argument('--epochs', type=int, help='Number of epochs for training.', default=5)
    p.add_argument('--lr', type=float, help='Learning rate', default=1e-5)
    p.add_argument('--dropout', type=float, help='Dropout rate', default=0.1)

    return p.parse_args()


def get_tagset(tagging_scheme):
    if 'conll' in tagging_scheme:
        return conll_iob
    elif 'wnut' in tagging_scheme:
        return  wnut_iob
    elif 'resume' in tagging_scheme:
        return resume_iob
    elif 'ontonotes' in tagging_scheme:
        return ontonotes_iob
    elif 'msra' in tagging_scheme:
        return msra_iob
    elif 'weibo' in tagging_scheme:
        return weibo_iob


def get_out_filename(out_dir, model, prefix):
    model_name = os.path.basename(model)
    model_name = model_name[:model_name.rfind('.')]
    return '{}/{}_base_{}.tsv'.format(out_dir, prefix, model_name)


def write_eval_performance(eval_performance, out_file):
    outstr = ''
    added_keys = set()
    for out_ in eval_performance:
        for k in out_:
            if k in added_keys or k in ['results', 'predictions']:
                continue
            outstr = outstr + '{}\t{}\n'.format(k, out_[k])
            added_keys.add(k)

    open(out_file, 'wt').write(outstr)
    logger.info('Finished writing evaluation performance for {}'.format(out_file))


def get_reader(file_path, max_instances=-1, max_length=50, target_vocab=None, encoder_model='xlm-roberta-large'):
    if file_path is None:
        return None
    reader = CoNLLReader(max_instances=max_instances, max_length=max_length, target_vocab=target_vocab, encoder_model=encoder_model)
    reader.read_data(file_path)

    return reader


def create_model(train_data, dev_data, tag_to_id, batch_size=64, dropout_rate=0.1, stage='fit', lr=1e-5, encoder_model='xlm-roberta-large', num_gpus=1):
    return NERBaseAnnotator(train_data=train_data, dev_data=dev_data, tag_to_id=tag_to_id, batch_size=batch_size, stage=stage, encoder_model=encoder_model,
                            dropout_rate=dropout_rate, lr=lr, pad_token_id=train_data.pad_token_id, num_gpus=num_gpus)


def load_model(model_file, tag_to_id=None, stage='test'):
    if ~os.path.isfile(model_file):
        model_file = get_models_for_evaluation(model_file)

    hparams_file = model_file[:model_file.rindex('checkpoints/')] + '/hparams.yaml'
    model = NERBaseAnnotator.load_from_checkpoint(model_file, hparams_file=hparams_file, stage=stage, tag_to_id=tag_to_id)
    model.stage = stage
    return model, model_file


def save_model(trainer, out_dir, model_name='', timestamp=None):
    out_dir = out_dir + '/lightning_logs/version_' + str(trainer.logger.version) + '/checkpoints/'
    if timestamp is None:
        timestamp = time.time()
    os.makedirs(out_dir, exist_ok=True)

    outfile = out_dir + '/' + model_name + '_timestamp_' + str(timestamp) + '_final.ckpt'
    trainer.save_checkpoint(outfile, weights_only=True)

    logger.info('Stored model {}.'.format(outfile))
    return outfile


def train_model(model, out_dir='', epochs=10, gpus=1):
    trainer = get_trainer(gpus=gpus, out_dir=out_dir, epochs=epochs)
    trainer.fit(model)
    return trainer


def get_trainer(gpus=4, is_test=False, out_dir=None, epochs=10):
    seed_everything(42)
    if is_test:
        return pl.Trainer(gpus=1) if torch.cuda.is_available() else pl.Trainer(val_check_interval=100)

    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=gpus, deterministic=True, max_epochs=epochs, callbacks=[get_model_earlystopping_callback()],
                             default_root_dir=out_dir, distributed_backend='ddp', checkpoint_callback=False)
        trainer.callbacks.append(get_lr_logger())
    else:
        trainer = pl.Trainer(max_epochs=epochs, default_root_dir=out_dir)

    return trainer


def get_lr_logger():
    lr_monitor = LearningRateMonitor(logging_interval='step')
    return lr_monitor


def get_model_earlystopping_callback():
    es_clb = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=3,
        verbose=True,
        mode='min'
    )
    return es_clb


def get_models_for_evaluation(path):
    if 'checkpoints' not in path:
        path = path + '/checkpoints/'
    model_files = list_files(path)
    models = [f for f in model_files if f.endswith('final.ckpt')]

    return models[0] if len(models) != 0 else None


def list_files(in_dir):
    files = []
    for r, d, f in os.walk(in_dir):
        for file in f:
            files.append(os.path.join(r, file))
    return files


