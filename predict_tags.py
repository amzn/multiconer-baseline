import time

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.utils import parse_args, get_reader, load_model, get_out_filename, get_tagset

if __name__ == '__main__':
    timestamp = time.time()
    sg = parse_args()

    # load the dataset first
    test_data = get_reader(file_path=sg.test, target_vocab=get_tagset(sg.iob_tagging), max_instances=sg.max_instances, max_length=sg.max_length)

    model, model_file = load_model(sg.model, tag_to_id=get_tagset(sg.iob_tagging))
    model = model.to(sg.cuda)
    # use pytorch lightnings saver here.
    eval_file = get_out_filename(sg.out_dir, model_file, prefix=sg.prefix)

    test_dataloaders = DataLoader(test_data, batch_size=sg.batch_size, collate_fn=model.collate_batch, shuffle=False, drop_last=False)
    out_str = ''
    index = 0
    for batch in tqdm(test_dataloaders, total=len(test_dataloaders)):
        pred_tags = model.predict_tags(batch, device=sg.cuda)

        for pred_tag_inst in pred_tags:
            out_str += '\n'.join(pred_tag_inst)
            out_str += '\n\n\n'
        index += 1
    open(eval_file, 'wt').write(out_str)
