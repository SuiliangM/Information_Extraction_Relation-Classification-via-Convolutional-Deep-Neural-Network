# coding=utf-8

'''
Preprocess original Sem-eval task8 data
'''

from glob import glob
import os
import re


def process_question(question):
    '''

    '''
    question = question.replace("'", " '")
    question = question.replace(",", " ,")
    question = question.replace(".", " .")
    question = question.split(' ')
    e1_begin = e1_end = e2_begin = e2_end = 0
    for i, item in enumerate(question):
        if item.startswith('<e1>'):
            e1_begin = i
        if item.endswith('</e1>'):
            e1_end = i
        if item.startswith('<e2>'):
            e2_begin = i
        if item.endswith('</e2>'):
            e2_end = i

    def remove_tag(x):
        x = x.replace('<e1>', '')
        x = x.replace('</e1>', '')
        x = x.replace('<e2>', '')
        x = x.replace('</e2>', '')
        return x
    question = list(map(remove_tag, question))
    return question, e1_begin, e1_end, e2_begin, e2_end


def process_file(in_filename, out_filename):
    max_len = 0
    max_distance = 0
    with open(in_filename, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for i in range(0, len(lines), 4):
        relation = lines[i+1].strip()
        question = lines[i].strip().split('\t')[1][1:-1]
        question, e1_begin, e1_end, e2_begin, e2_end = process_question(
            question)
        max_len = max(max_len, len(question))
        max_distance = max(max_distance, e1_end)
        max_distance = max(max_distance, len(question) - e1_end)
        max_distance = max(max_distance, e2_end)
        max_distance = max(max_distance, len(question) - e2_end)
        new_lines.append('{}\t{}\t{}\t{}\t{}\t{}\n'.format(' '.join(question),
                                                           e1_begin, e1_end, e2_begin, e2_end, relation))
    with open(out_filename, 'w') as f:
        f.writelines(new_lines)
    print("Max length: {}".format(max_len))
    print("Max distance: {}".format(max_distance))


def structure_2007(filename):
    output = open(filename.replace('raw', 'structured'), 'w')
    lines = open(filename).readlines()

    block = []

    for line in lines:
        if line == '\n':
            if len(block) == 2:
                block.append('Comment:\n')

            # Block is complete
            first_line = block[0]
            first_line = first_line.split()
            right = ' '.join(first_line[1:])
            first_line = first_line[0] + '\t' + right + '\n'
            block[0] = first_line

            
            second_line = block[1]
            pattern = re.compile(' = "(false)|(true)"')
            label = pattern.search(second_line).group()
            block[1] = label + '\n'

            for item in block:
                output.write(item)
            block = []
            output.write('\n')

        else:
            block.append(line)

def split_2007(filename):
    '''
    Split the 2007 
    '''
    lines = open(filename).readlines()
    train_name = filename.replace('processed', 'split')
    train_name = train_name.replace('relation-', '')
    dev_name = train_name.replace('train', 'dev')
    train_output = open(train_name, 'w')
    dev_output = open(dev_name, 'w')

    train_output.writelines(lines[:120])
    dev_output.writelines(lines[120:])


def process_embedding(filename, out_path):
    '''
    Convert the downloaded embedding file into two files
    '''
    lines = open(filename).readlines()
    out_embedding = open(out_path + '/embeddings.txt', 'w')
    out_wordlist = open(out_path + '/words.lst', 'w')
    for line in lines:
        word = line.split()[0]
        embed = line.split()[1:]
        out_wordlist.write(word + '\n')
        out_embedding.write(" ".join(embed) + '\n')

def binarize_2010(filename, train_test):
    relations = [[], [], [], [], [], [], [], []]    
    lines = open(filename).readlines()
    for line in lines:
        label = line.split('\t')[-1].strip()
        line = line.split('\t')
        if 'Cause-Effect' in label:
            relations[0].append('\t'.join(line[:-1]) + '\ttrue\n')
        elif 'Instrument-Agency' in label: 
            relations[1].append('\t'.join(line[:-1]) + '\ttrue\n')
        elif 'Product-Producer' in label:
            relations[2].append('\t'.join(line[:-1]) + '\ttrue\n')
        elif 'Entity-Origin' in label:
            relations[3].append('\t'.join(line[:-1]) + '\ttrue\n')
        elif 'Theme-Tool' in label:
            relations[4].append('\t'.join(line[:-1]) + '\ttrue\n')
        elif 'Component-Whole' in label:
            relations[5].append('\t'.join(line[:-1]) + '\ttrue\n')
        elif 'Content-Container' in label:
            relations[6].append('\t'.join(line[:-1]) + '\ttrue\n')
        elif 'Other' in label:
            relations[7].append('\t'.join(line[:-1]) + '\tfalse\n')
            
    for i, relation in enumerate(relations):
        output = open('data/2010/bin_raw_2010/{}-{}'.format(train_test, i+1), 'w')
        output.writelines(relation)
        output.close()

def merge_bin_2010(filename):
    '''
    '''
    negatives = open('data/2010/bin_raw_2010/test/test-8').readlines()
    positive = open(filename).readlines()
    lines = positive + negatives
    lines = list(set(lines))
    output = open(filename.replace('bin_raw', 'merge_bin'), 'w')
    output.writelines(lines)
    output.close()


if __name__ == '__main__':
    # process_file("./data/TRAIN_FILE.TXT", "./data/train.txt")
    # process_file(
    #     'SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT', 'data/test.txt')
    
    # for filename in files:
        # structure_2007(filename)
        # process_file(filename.replace('raw', 'structured'), filename.replace('structured', 'processed'))
        # process_file(filename, filename.replace('structured', 'processed'))
        # split_2007(filename)

    # binarize_2010('data/2010/train.txt', 'train')
    # binarize_2010('data/2010/test.txt', 'test')

    files = glob("data/2010/bin_raw_2010/test/*")
    for filename in files:
        merge_bin_2010(filename)