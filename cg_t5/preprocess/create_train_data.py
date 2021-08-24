from origin_input.evaluate import Evaluator
import json


class Creator:
    def __init__(self):
        self.sample_list = []
        self.evaluator = Evaluator()

    def clear_string(self, sentence):
        sentence = sentence.replace("\n", "")
        sentence = sentence.replace("\r", "")
        sentence = sentence.replace("\t", "")
        return sentence

    def cal_rouge(self, answer, candidate):
        score = self.evaluator.compute_rouge(answer, candidate)
        rouge_1 = score["rouge-1"]
        rouge_2 = score["rouge-2"]
        rouge_l = score["rouge-l"]
        rouge_all = 0.2 * rouge_1 + 0.4 * rouge_2 + 0.4 * rouge_l
        return rouge_all

    def get_label(self, answer, candidate):
        rouge_all = self.cal_rouge(answer, candidate)
        index = 0
        if rouge_all < 0.35:
            index = 0
        else:
            index = 1
        return index

    def get_labels(self, answer, candidate_list):
        rouge_list = []
        labels = []
        for candidate in candidate_list:
            candidate = self.clear_string(candidate)
            rouge_all = self.cal_rouge(answer, candidate)
            rouge_list.append(rouge_all)
        max_index = rouge_list.index(max(rouge_list))
        for index, rouge in enumerate(rouge_list):
            if index == max_index:
                labels.append(1)
            else:
                labels.append(0)
        return labels

    def process_sample(self, question, candidate_list, answer):
        question = self.clear_string(question)
        answer = self.clear_string(answer)
        labels = self.get_labels(answer, candidate_list)
        for index, candidate in enumerate(candidate_list):
            candidate = self.clear_string(candidate)
            sample = {"du1": question, "du2": candidate, "rs": answer,"label": labels[index]}
            self.sample_list.append(sample)

    def write_to_file(self, des_file):
        with open(des_file, mode='w', encoding='utf-8') as fw:
            fw.write(json.dumps(self.sample_list, ensure_ascii=False))


if __name__ == '__main__':
    output_path = "../data_dir/test_sample.json"
    input_path = "test.json"
    creator = Creator()
    with open(input_path, 'r', encoding="utf8") as f:
        for index, line in enumerate(f):
            print("正在处理第" + str(index) + "样本")
            data = json.loads(line)
            id = data.get('id')
            question = data.get("question")
            candidate_list = data.get('candidates')
            answer = data.get('answer')
            creator.process_sample(question, candidate_list, answer)
            creator.write_to_file(output_path)
