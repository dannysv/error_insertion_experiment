import json
import random
import codecs
from mutils.utils import removeall
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('nsample', 
                        type=int, 
                        help='the number of instances to be selected on the sample', 
                        default=2000)
    args=parser.parse_args()
    nsample = args.nsample 
    with open('../error_insertion/abstracts_cleaned_erro-0.25.json', 'r') as f:
        data = json.load(f)

    sample = random.sample(data.keys(), nsample)
    removeall('./gt')
    removeall('./ocr')
    #out_gt = codecs.open('./gt/abstracts.txt', 'w')
    #out_ocr = codecs.open('./ocr/abstracts.txt', 'w')


    for item in sample:
        out_gt = codecs.open('./gt/'+str(item)+'.txt', 'w')
        out_ocr = codecs.open('./ocr/'+str(item)+'.txt', 'w')
        out_gt.write(data[item]['abstracts_pt'].lower()+'\n')
        out_ocr.write(data[item]['abstracts_pt_error'].lower()+'\n')
    print('ok')
