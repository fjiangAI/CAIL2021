import json

from evaluate import Evaluator

input_path = './input/input.json'  # input file path
output_path = './output/length2.tsv'  # output file path
evaluator = Evaluator()


def calculate_rouge_all(score):
    rouge_all = 0.2 * score["rouge-1"] + 0.4 * score["rouge-2"] + 0.4 * score["rouge-l"]
    return rouge_all

def get_length(question, candidate, answer):
    strings = question
    strings += candidate
    strings += answer
    return strings

if __name__ == '__main__':
    length = []
    with open(output_path, 'w', encoding='utf8') as fw:
        with open(input_path, 'r', encoding="utf8") as f:
            for line in f:
                data = json.loads(line)
                id = data.get('id')
                question = data.get("question")
                question=question.replace("\n","")
                question = question.replace("\r", "")
                candidates_list = data.get('candidates')
                real_answer = data.get("answer")
                real_answer = real_answer.replace("\n", "")
                for candidate in candidates_list:
                    candidate = candidate.replace("\n", "")
                    temp_length = len(get_length(question, candidate, real_answer))
                    score = evaluator.compute_rouge(real_answer, candidate)
                    rouge_1=score["rouge-1"]
                    rouge_2 = score["rouge-2"]
                    rouge_l = score["rouge-l"]
                    rouge_all = calculate_rouge_all(score)
                    fw.write(str(question) + "\t" +str(candidate) + "\t"+ str(real_answer) + "\t" +
                             str(rouge_1)+"\t"+str(rouge_2)+"\t"+str(rouge_l)+"\t"+ str(rouge_all)+"\t"+
                             str(len(question)) +"\t"+str(len(question+candidate)) +"\t"+str(len(real_answer))+ "\t" +str(temp_length) + "\n")
