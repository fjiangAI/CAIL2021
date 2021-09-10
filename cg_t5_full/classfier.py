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

    def find_longest_du(self, dus):
        max_length = 0
        max_index = -1
        all_length = 0
        for index, du in enumerate(dus):
            all_length += len(du)
            if len(du) > max_length:
                max_length = len(du)
                max_index = index
        return max_index, all_length

    def convert_feature(self, sample, tokenizer, rs_max_len, max_len):
        """
            数据处理函数
            Args:
                sample: 一个字典，格式为{"du1": du1, "du2": du2}
            Returns:
            """
        input_ids = []
        du1_tokens = tokenizer.tokenize(sample["du1"])
        du2_tokens = []
        for du2 in sample["du2"]:
            du2_token = self.tokenizer.tokenize(du2)
            du2_tokens.append(du2_token)
        # 判断如果正文过长，进行截断
        max_index, all_length = self.find_longest_du(du2_tokens)
        while len(du1_tokens) + all_length > max_len - rs_max_len - (3 + len(du2_tokens)):
            if len(du1_tokens) > len(du2_tokens[max_index]):
                du1_tokens = du1_tokens[:-1]
            else:
                du2_tokens[max_index] = du2_tokens[max_index][:-1]
            max_index, all_length = self.find_longest_du(du2_tokens)
        # 生成模型所需的input_ids和token_type_ids
        input_ids.append(self.tokenizer.cls_token_id)
        input_ids.extend(self.tokenizer.convert_tokens_to_ids(du1_tokens))
        input_ids.append(self.tokenizer.sep_token_id)
        for du2_token in du2_tokens:
            input_ids.extend(self.tokenizer.convert_tokens_to_ids(du2_token))
            input_ids.append(self.tokenizer.sep_token_id)
        return input_ids

    def generate_results(self, output, tokenizer):
        title_list = []
        for i in range(len(output)):
            title = ''.join(tokenizer.decode(output[i][1:], skip_special_tokens=True)).replace(' ', '')
            title_list.append(title)
        return title_list

    def find_smallest_du(self, candidate_list):
        min_length = 10000
        min_index = -1
        for index, du in enumerate(candidate_list):
            if len(du) < min_length:
                min_length = len(du)
                min_index = index
        return min_index

    def clear_strings(self, candidates):
        new_candidates = []
        for candidate in candidates:
            new_candidate = self.clear_string(candidate)
            new_candidates.append(new_candidate)
        while len(new_candidates) > 4:
            small_index = self.find_smallest_du(new_candidates)
            new_candidates.pop(small_index)
        return new_candidates

    def predict(self, question, candidate_list):
        input_list = []
        title_list = []
        input_tensors=[]
        question = self.clear_string(question)
        candidate_list = self.clear_strings(candidate_list)
        sample = {"du1": question, "du2": candidate_list}
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
        return title_list[0]

    def clear_string(self, sentence):
        sentence = sentence.replace("\n", "")
        sentence = sentence.replace("\r", "")
        sentence = sentence.replace("\t", "")
        return sentence
