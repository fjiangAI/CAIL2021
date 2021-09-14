from origin_input.evaluate import Evaluator
import json
import heapq


class Creator:
    def __init__(self):
        self.sample_list = []
        self.evaluator = Evaluator()
        self.stop_list = ["电联", "当面", "私聊", "付费咨询",
                          "vx", "*", "我是", "欢迎来电", "来电咨询", "电话咨询",
                          "我的", "我们是", "我是", "电话", "微同号", "解答", "详细沟通"]

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

    def select_top_candidate(self, answer, candidate_list):
        rouge_list = []
        new_candidate_list = []
        max_index = 0
        for candidate in candidate_list:
            candidate = self.clear_string(candidate)
            rouge_all = self.cal_rouge(answer, candidate)
            rouge_list.append(rouge_all)
        top_index = heapq.nlargest(4, range(len(rouge_list)), rouge_list.__getitem__)
        for i in range(len(candidate_list)):
            if i in top_index:
                if i == top_index[0]:
                    max_index = len(new_candidate_list)
                temp_candidate = self.clear_string(candidate_list[i])
                new_candidate_list.append(temp_candidate)
        return new_candidate_list, max_index

    def clear_strings(self, candidates):
        new_candidates = []
        for candidate in candidates:
            new_candidate = self.clear_string(candidate)
            new_candidates.append(new_candidate)
        return new_candidates

    def has_stop_word(self, sentence):
        for stop_word in self.stop_list:
            if sentence.__contains__(stop_word):
                return True
        return False

    def delete_AD(self, candidate_list):
        new_candidate_list = []
        for candidate in candidate_list:
            if self.has_stop_word(candidate):
                continue
            new_candidate_list.append(candidate)
        return new_candidate_list

    def process_sample(self, question, candidate_list, answer):
        question = self.clear_string(question)
        answer = self.clear_string(answer)
        #candidate_list=self.delete_AD(candidate_list)
        new_candidate_list, max_index = self.select_top_candidate(answer, candidate_list)
        sample = {"du1": question, "du2": new_candidate_list, "rs": answer, "label": max_index}
        self.sample_list.append(sample)

    def write_to_file(self, des_file):
        with open(des_file, mode='w', encoding='utf-8') as fw:
            fw.write(json.dumps(self.sample_list, ensure_ascii=False))


if __name__ == '__main__':
    output_path = "../data_dir/train_sample.json"
    input_path = "./data_second_train.json"
    input_path2 = "./data_first_train.json"
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
    with open(input_path2, 'r', encoding="utf8") as f:
        for index, line in enumerate(f):
            print("正在处理第" + str(index) + "样本")
            data = json.loads(line)
            id = data.get('id')
            question = data.get("question")
            candidate_list = data.get('candidates')
            answer = data.get('answer')
            creator.process_sample(question, candidate_list, answer)
    creator.write_to_file(output_path)
