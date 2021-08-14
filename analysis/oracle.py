import json

from evaluate import Evaluator

input_path = './input/input.json'  # input file path
output_path = './output/result.json'  # output file path


class Oracle:
    def __init__(self):
        self.evaluator = Evaluator()

    def calculate_rouge_all(self, score):
        rouge_all = 0.2 * score["rouge-1"] + 0.4 * score["rouge-2"] + 0.4 * score["rouge-l"]
        return rouge_all

    def get_longest_sentence(self,candidate_list):
        answer = ""
        for candidate in candidate_list:
            if len(candidate) > len(answer):
                answer = candidate
        return answer

    def select_answer(self, candidates_list, real_answer):
        oracle_answer = ""
        max_rouge_all = 0.0
        max_score = None
        select_answer_index=-1
        for index, candidate in enumerate(candidates_list):
            score = self.evaluator.compute_rouge(real_answer, candidate)
            rouge_all = self.calculate_rouge_all(score)
            if rouge_all > max_rouge_all:
                oracle_answer = candidate
                max_rouge_all = rouge_all
                max_score = score
                select_answer_index=index
        if max_score==None:
            oracle_answer="您好，"
            score = self.evaluator.compute_rouge(real_answer, oracle_answer)
            rouge_all = self.calculate_rouge_all(score)
            max_rouge_all = rouge_all
            max_score = score
        return oracle_answer, max_rouge_all, max_score,select_answer_index


if __name__ == "__main__":
    oracle = Oracle()
    with open(output_path, 'w', encoding='utf8') as fw:
        with open(input_path, 'r', encoding="utf8") as f:
            for line in f:
                data = json.loads(line)
                id = data.get('id')
                candidates_list = data.get('candidates')
                real_answer = data.get("answer")
                oracle_answer, max_rouge_all, max_score, select_answer_index = oracle.select_answer(candidates_list, real_answer)
                rst = {
                    "id": id,
                    "real_answer": real_answer,
                    "candidates":candidates_list,
                    "answer": oracle_answer,
                    "rouge-1": max_score["rouge-1"],
                    "rouge-2": max_score["rouge-2"],
                    "rouge-l": max_score["rouge-l"],
                    "rouge-all": max_rouge_all,
                    "select_answer_index":select_answer_index,
                }
                fw.write(json.dumps(rst, ensure_ascii=False) + '\n')
