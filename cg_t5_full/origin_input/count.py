import json
import heapq

if __name__ == '__main__':
    rouge_list=[4,2,6]
    top_index = heapq.nlargest(4, range(len(rouge_list)), rouge_list.__getitem__)
    print(top_index)