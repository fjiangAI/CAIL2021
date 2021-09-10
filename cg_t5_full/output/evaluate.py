from rouge.rouge import Rouge
import json
import sys

sys.setrecursionlimit(100000)


class Evaluator:
    def __init__(self):
        self.rouge = Rouge()
        self.answer = {"golden": [], "predict": []}
        self.result = {}
        self.question = []
        self.score = []

    def clear_string(self, sentence):
        sentence = sentence.replace("\n", "")
        sentence = sentence.replace("\r", "")
        sentence = sentence.replace("\t", "")
        return sentence

    def read_answer(self, file_name, mode="golden"):
        with open(file_name, 'r', encoding="utf8") as f:
            for line in f:
                data = json.loads(line)
                self.answer[mode].append(data.get('answer'))
                if mode == "golden":
                    self.question.append(self.clear_string(data.get("question")))

    def compute_rouge(self, source, target):
        """计算rouge-1、rouge-2、rouge-l
        """
        source, target = ' '.join(source), ' '.join(target)
        try:
            scores = self.rouge.get_scores(hyps=source, refs=target)
            return {
                'rouge-1': scores[0]['rouge-1']['f'],
                'rouge-2': scores[0]['rouge-2']['f'],
                'rouge-l': scores[0]['rouge-l']['f'],
            }
        except ValueError:
            return {
                'rouge-1': 0.0,
                'rouge-2': 0.0,
                'rouge-l': 0.0,
            }

    def compute_rouges(self):
        sources = self.answer["golden"]
        targets = self.answer["predict"]
        scores = {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }
        for source, target in zip(sources, targets):
            score = self.compute_rouge(source, target)
            for k, v in scores.items():
                scores[k] = v + score[k]
            new_score = 0.2 * score["rouge-1"] + 0.4 * score["rouge-2"] + 0.4 * score["rouge-l"]
            self.score.append(new_score)
        self.result = {k: v / len(targets) for k, v in scores.items()}
        self.result["rouge-all"] = 0.2 * self.result["rouge-1"] + 0.4 * self.result["rouge-2"] + 0.4 * self.result[
            "rouge-l"]

    def show_result(self):
        print(self.result)

    def write_detail(self, des_file_name):
        with open(des_file_name, encoding='utf-8', mode='w') as fw:
            for question, golden, predict, score in zip(self.question, self.answer["golden"], self.answer["predict"],
                                                        self.score):
                fw.write(question + "\t" + golden + "\t" + predict + "\t" + str(score) + "\n")


if __name__ == '__main__':
    input_path = '../data_dir/test.json'  # input file path
    for i in range(10):
        output_path = './cg_t5_full2/result' + str(i) + '.json'  # output file path
        evaluator = Evaluator()
        evaluator.read_answer(input_path, "golden")
        evaluator.read_answer(output_path, "predict")
        evaluator.compute_rouges()
        evaluator.show_result()
        evaluator.write_detail("./cg_t5_full2/detail.tsv")