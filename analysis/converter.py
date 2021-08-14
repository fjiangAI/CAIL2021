import json


class Converter:
    def __init__(self):
        pass

    def json2tsv(self, input_file, output_file):
        with open(output_file, 'w', encoding='utf8') as fw:
            with open(input_file, 'r', encoding="utf8") as f:
                for line in f:
                    data = json.loads(line)
                    id = data.get('id')
                    oracle_answer = data.get('answer')
                    oracle_answer=oracle_answer.replace("\n","")
                    real_answer = data.get("real_answer")
                    real_answer=real_answer.replace("\n","")
                    rouge_all = data.get("rouge-all")
                    selected_index=data.get("select_answer_index")
                    fw.write(str(id) + "\t" + str(oracle_answer) + "\t" + str(real_answer) + "\t" + str(rouge_all) + "\t"+str(selected_index)+'\n')

if __name__ == '__main__':
    input_file="./output/result.json"
    output_file = "./output/result.tsv"
    converter=Converter()
    converter.json2tsv(input_file=input_file,output_file=output_file)