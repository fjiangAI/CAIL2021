import json
import json
import random
import numpy as np
import torch
import os
import argparse

from torch.nn.utils.rnn import pad_sequence
import sys

sys.path.append("../")
from module.tokenizer import T5PegasusTokenizer

from classfier import Classifier

input_path = './input/input.json'  # origin_input file path
output_path = './output/result4.json'  # output file path
model_path = './output_dir/model-4'


def choice_answer(classifier, question, candidate_list):
    answer_list, class_index = classifier.predict(question, candidate_list)
    answer = answer_list[class_index]
    if len(answer) < 3:
        return "您好，"
    else:
        return answer


def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # 获取设备信息
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed=42)
    classifier = Classifier(vocab_path='./t5_pegasus_torch/vocab.txt', model_path=model_path, device=0, rs_max_len=200,
                            max_len=512)
    with open(output_path, 'w', encoding='utf8') as fw:
        with open(input_path, 'r', encoding="utf8") as f:
            for line in f:
                data = json.loads(line)
                id = data.get('id')
                answer = choice_answer(classifier, data.get('question'), data.get('candidates'))
                rst = dict(
                    id=id,
                    answer=answer
                )
                fw.write(json.dumps(rst, ensure_ascii=False) + '\n')
