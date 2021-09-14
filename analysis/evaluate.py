from rouge.rouge import Rouge
import json
import sys
sys.setrecursionlimit(100000)

class Evaluator:
    def __init__(self):
        self.rouge = Rouge()
        self.answer = {"golden": [], "predict": []}
        self.result = {}

    def read_answer(self, file_name, mode="golden"):
        with open(file_name, 'r', encoding="utf8") as f:
            for line in f:
                data = json.loads(line)
                self.answer[mode].append(data.get('answer'))

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

        self.result = {k: v / len(targets) for k, v in scores.items()}
        self.result["rouge-all"] = 0.2 * self.result["rouge-1"] + 0.4 * self.result["rouge-2"] + 0.4 * self.result["rouge-l"]
    def show_result(self):
        print(self.result)

if __name__ == '__main__':
    input_path = './input/input.json'  # input file path
    output_path = './lawformer/result2.json'  # output file path
    evaluator = Evaluator()
    evaluator.read_answer(input_path, "golden")
    evaluator.read_answer(output_path, "predict")
    evaluator.compute_rouges()
    evaluator.show_result()
