class Split:
    def __init__(self):
        self.lines = {"train": [], "test": []}

    def read_file(self, file):
        with open(file, encoding='utf-8', mode='r') as fr:
            lines = fr.readlines()
            for index, line in enumerate(lines):
                if index < 10000:
                    self.lines["train"].append(line)
                else:
                    self.lines["test"].append(line)

    def write_file(self, des_file, mode="train"):
        with open(des_file, encoding='utf-8', mode='w') as fw:
            for line in self.lines[mode]:
                fw.write(line)


if __name__ == '__main__':
    input_file = "input.json"
    train_file = "train.json"
    test_file = "../input/input.json"
    splitor = Split()
    splitor.read_file(input_file)
    splitor.write_file(train_file, "train")
    splitor.write_file(test_file, "test")
