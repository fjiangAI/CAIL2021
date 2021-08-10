import json
from random import choice

input_path = './input/input.json'  # input file path
output_path = './output/result.json'  # output file path

stop_list = ["电联", "当面", "私聊", "付费咨询", "微信",
             "vx", "*", "我是", "欢迎来电", "来电咨询","电话咨询",
             "我的","我们是","我是","电话","微同号","解答","详细沟通"]
select_list = ["您好", "你好"]


def has_stopword(sentence, stop_list):
    for stop_word in stop_list:
        if sentence.__contains__(stop_word):
            return True
    return False


def get_longest_sentence(candidate_list):
    answer = ""
    for candidate in candidate_list:
        if len(candidate) > len(answer):
            answer = candidate
    return answer


def choice_answer(sentence_list):
    answer = ""
    candidate_answer = []
    for sentence in sentence_list:
        sentence = sentence.replace("\n", "")
        if has_stopword(sentence, stop_list):
            continue
        candidate_answer.append(sentence)
    for candidate in candidate_answer:
        for select_word in select_list:
            if candidate.__contains__(select_word):
                if len(candidate) > len(answer):
                    answer = candidate
    if len(answer) == 0:
        answer = get_longest_sentence(candidate_answer)
        if len(answer) == 0:
            answer = sentence_list[0]
        answer = "您好，" + answer
    return answer


if __name__ == "__main__":
    with open(output_path, 'w', encoding='utf8') as fw:
        with open(input_path, 'r', encoding="utf8") as f:
            for line in f:
                data = json.loads(line)
                id = data.get('id')
                answer = choice_answer(data.get('candidates'))
                rst = dict(
                    id=id,
                    answer=answer
                )
                fw.write(json.dumps(rst, ensure_ascii=False) + '\n')
