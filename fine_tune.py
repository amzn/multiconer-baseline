import time

from utils.utils import get_reader, train_model, save_model, parse_args, get_tagset, load_model

if __name__ == '__main__':
    timestamp = time.time()
    sg = parse_args()
    out_dir_path = sg.out_dir + '/' + sg.model_name

    # load the dataset first
    train_data = get_reader(file_path=sg.train, target_vocab=get_tagset(sg.iob_tagging), encoder_model=sg.encoder_model, max_instances=sg.max_instances, max_length=sg.max_length)
    model, model_file = load_model(sg.model, tag_to_id=get_tagset(sg.iob_tagging), stage='finetune')
    model.train_data = train_data

    trainer = train_model(model=model, out_dir=out_dir_path, epochs=sg.epochs)

    # use pytorch lightnings saver here.
    out_model_path = save_model(trainer=trainer, out_dir=out_dir_path, model_name=sg.model_name, timestamp=timestamp)
