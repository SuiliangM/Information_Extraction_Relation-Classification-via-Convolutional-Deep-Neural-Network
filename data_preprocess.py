import os
from tqdm import tqmd

class Data(object):
    def __init__(self, training_file_folder, is_binary):
        self.train_folder = training_file_folder
        self.is_binary = is_binary
        
    def read_multi_relation(self):
        for input_training_file in tqdm(os.listdir(self.train_folder)):
            with open(input_training_file, "r") as rfp:
                lines = rfp.readlines()
                
                for line in tqdm(lines):
                    ## split line by \t
                    
                    elements = line.strip().split('\t')
                    
                    relation = elements[0] ## the first element is the relation
                    
                    sentence = elements[3] ## the third element is the sentence
                    
                    
            
        
    def process_binary_relation(self, )    