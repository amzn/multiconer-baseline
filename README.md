# MULTI-CONER NER Baseline.

This code repository provides you with baseline approach for Named Entity Recognition (NER). In this repository the following functionalities are provided:

- CoNLL data readers
- Usage of any HuggingFace pre-trained transformer models
- Training and Testing through Pytorch-Lightning  

Below a more detailed description on how to use this code is provided.

### Running the Code

#### Arguments:
```
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
p.add_argument('--epochs', type=int, help='Number of epochs for training.', default=5)
p.add_argument('--lr', type=float, help='Learning rate', default=1e-5)
p.add_argument('--dropout', type=float, help='Dropout rate', default=0.1)
``` 

#### Running 

###### Train a XLM-RoBERTa base model
```
python -m ner_baseline.train_model --train train.txt --dev dev.txt --out_dir . --model_name xlmr_ner --gpus 1 \
                                   --epochs 2 --encoder_model xlm-roberta-base --batch_size 64 --lr 0.0001
```

###### Evaluate the trained model
```
python -m ner_baseline.evaluate --test test.txt --out_dir . --gpus 1 --encoder_model xlm-roberta-base \
                                --model MODEL_FILE_PATH --prefix xlmr_ner_results

```

### Setting up the code environment

```
$ pip install -r requirements.txt
```

# License 
The code under this repository is licensed under the [Apache 2.0 License](https://github.com/amzn/multiconer-baseline/blob/main/LICENSE).
