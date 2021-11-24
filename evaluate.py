import time

from torch.utils.data import DataLoader

from utils.utils import parse_args, get_reader, load_model, get_trainer, get_out_filename, write_eval_performance, get_tagset

if __name__ == '__main__':
    timestamp = time.time()
    sg = parse_args()

    # load the dataset first
    test_data = get_reader(file_path=sg.test, target_vocab=get_tagset(sg.iob_tagging), max_instances=sg.max_instances, max_length=sg.max_length, encoder_model=sg.encoder_model)

    model, model_file = load_model(sg.model, tag_to_id=get_tagset(sg.iob_tagging))
    trainer = get_trainer(is_test=True)
    out = trainer.test(model, test_dataloaders=DataLoader(test_data, batch_size=sg.batch_size, collate_fn=model.collate_batch))

    # use pytorch lightnings saver here.
    eval_file = get_out_filename(sg.out_dir, model_file, prefix=sg.prefix)
    write_eval_performance(out, eval_file)
