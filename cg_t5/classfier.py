import torch
from torch.nn.utils.rnn import pad_sequence
from module.tokenizer import T5PegasusTokenizer


class Classifier:
    def __init__(self, vocab_path, model_path, device=0, rs_max_len=200, max_len=512):
        self.tokenizer = T5PegasusTokenizer.from_pretrained(vocab_path)
        self.model = torch.load(model_path)
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.rs_max_len = rs_max_len
        self.max_len = max_len
        self.generate_max_len = rs_max_len

    def convert_feature(self, sample, tokenizer, rs_max_len, max_len):
        """
            数据处理函数
            Args:
                sample: 一个字典，格式为{"du1": du1, "du2": du2}
            Returns:
            """
        input_ids = []
        du1_tokens = tokenizer.tokenize(sample["du1"])
        du2_tokens = tokenizer.tokenize(sample["du2"])
        # 判断如果正文过长，进行截断
        while len(du1_tokens) + len(du2_tokens) > max_len - rs_max_len - 4:
            if len(du1_tokens) > len(du2_tokens):
                du1_tokens = du1_tokens[:-1]
            else:
                du2_tokens = du2_tokens[:-1]
        # 生成模型所需的input_ids和token_type_ids
        input_ids.append(tokenizer.cls_token_id)
        input_ids.extend(tokenizer.convert_tokens_to_ids(du1_tokens))
        input_ids.append(tokenizer.sep_token_id)
        input_ids.extend(tokenizer.convert_tokens_to_ids(du2_tokens))
        input_ids.append(tokenizer.sep_token_id)
        return input_ids

    def generate_results(self, output, tokenizer):
        title_list = []
        for i in range(len(output)):
            title = ''.join(tokenizer.decode(output[i][1:], skip_special_tokens=True)).replace(' ', '')
            title_list.append(title)
        return title_list

    def search_max_index(self,x):
        x=x.cpu()
        idx = torch.LongTensor([[1]] * x.shape[0])
        out = torch.gather(x, 1, idx)
        index = out.argmax(dim=0)
        return int(index)
    def predict(self, question, candidate_list):
        input_list = []
        du1_list = []
        du2_list = []
        title_list = []
        input_tensors=[]
        for num, candidate in enumerate(candidate_list):
            question = self.clear_string(question)
            candidate = self.clear_string(candidate)
            sample = {"du1": question, "du2": candidate}
            input_ids = self.convert_feature(sample, self.tokenizer, self.rs_max_len, self.max_len)
            input_list.append(torch.tensor(input_ids, dtype=torch.long))
        with torch.no_grad():
            input_list = pad_sequence(input_list, batch_first=True, padding_value=0)
            input_tensors = torch.tensor(input_list).long().to(self.device)
            text_output, class_output = self.model.generate(input_tensors,
                                                                decoder_start_token_id=self.tokenizer.cls_token_id,
                                                                eos_token_id=self.tokenizer.sep_token_id,
                                                                max_length=self.generate_max_len,return_prob=True)
            title_list = self.generate_results(text_output, self.tokenizer)
            class_index = self.search_max_index(class_output)
        return title_list, class_index

    def clear_string(self, sentence):
        sentence = sentence.replace("\n", "")
        sentence = sentence.replace("\r", "")
        sentence = sentence.replace("\t", "")
        return sentence
