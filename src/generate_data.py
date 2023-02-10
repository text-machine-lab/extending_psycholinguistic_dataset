from os import truncate
from unittest.util import _MAX_LENGTH
from  transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import random
import argparse
import subprocess
import json


def get_arguments():
    """
    Get arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='neg', help="'neg' for neg-simp, 'role' for role-88")
    parser.add_argument("--key", default=None, help="key for gpt3 request")
    args = parser.parse_args()
    return args

def generate_role(key):
    """
    generate role pairs using raw role-88
    """
    data = []
    with open("data/original_datasets/ROLE-88.txt","r") as f:
        data = f.readlines() # readlines() returns a list of items, each item is a line in your file
    NUM_SAMPLES = 4 # Number of in context samples
    NUM_PROMPTING = 10 # How many times we are prompting gpt3 with NUM_SAMPLES

    for i in range(NUM_PROMPTING):  
        random_num_list = []
        while len(random_num_list) < NUM_SAMPLES:
            random_num_list.append(random.randrange(1, len(data), 2))
            random_num_list == list(set(random_num_list)) # list of random integer between item 1 till len of data
        print(random_num_list)
        # create the prompt. choose sample pairs using the random_num_list
        prompt = "The task is to reverse the role in the sentences. Generate more sentences like this: "
        for item in random_num_list:
            sample1 = data[item][5:].replace("|","\t").strip().split('\t')
            sample1 = sample1[0] + sample1[1]
        
            sample2 = data[item+1][5:].replace("|","\t").strip().split('\t')
            sample2 = sample2[0] + sample2[1]

            random_samples = sample1 + ',' + sample2 + ','
            prompt = prompt + random_samples

        # pass this prompt to the curl command to gpt3
        curl_req = 'curl https://api.openai.com/v1/completions \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ' + key + '" \
        -d \'{"model": "text-davinci-002", "prompt": ' + '"' + prompt + '"' +', "temperature": 0.64, "max_tokens": 100}\''
        
        # Getting resposnse from gpt3
        gpt3result = subprocess.check_output(curl_req, shell=True)
        gpt3result = json.loads(gpt3result)

        # extracting the output text from gpt3 response
        gpt3result = gpt3result['choices'][0]['text']
        print(gpt3result)
        
        # dumping the extracted text to a text file
        file = open("data/role-1500-unfiltered.txt", "a") 
        file.write(gpt3result)
        # Next the role-1500-unfiltered.txt file is manually cleaned
   
def generate_negsimp_template():
    """
    Generates the dataset NEG-SIMP using categories and subcategories through a template from the original paper 
    """
    file = pd.read_csv('data/original_datasets/neg-simp-categories.csv')
    cat_subcat = dict(zip(file['Category'], file['Subcategory']))
    
    sent = []
    for cat, subcat in cat_subcat.items():
        subcat = subcat.split(',')
        for item in subcat:
            # using the template
            aff = 'A ' + item + " is (a/an) " + cat.lower() + ','
            neg = 'A ' + item + " is not (a/an) " + random.choice([x.lower() for x in file['Category'] if x != cat]) + ','
            sent.append(aff)
            sent.append(neg)
    textfile = open("data/neg-simp-temp-unfiltered.txt", "w")
    # Next the neg-simp-temp-unfiltered.txt file is manually cleaned

    for element in sent:
        textfile.write(element + "\n")
    textfile.close()


def generate_negsimp_gen(key):
    """
    generate neg-1500-simp-gen-unfiltered using just GPT3 API
    """
    NUM_SAMPLES = 4 # Number of in context samples
    NUM_PROMPTING = 200 # How many times we are prompting gpt3 with NUM_SAMPLES

    data = []
    with open("data/original_datasets/NEG-136-SIMP.txt","r") as f:
        data = f.readlines() # readlines() returns a list of items, each item is a line in your file
    
    for i in range(NUM_PROMPTING):  
        random_num_list = []
        while len(random_num_list) < NUM_SAMPLES:
            random_num_list.append(random.randrange(0, len(data), 2))
            random_num_list == list(set(random_num_list)) # list of random integer between item 1 till len of data
        print(random_num_list)
        # create the prompt. choose sample pairs using the random_num_list
        prompt = "The task is to generate affirmative sentences and its negation. The object of the sentence should be a hypernym of the subject of the sentence. Generate more sentence pairs like these: "
        for item in random_num_list:
            data[item] = data[item].replace('\n',' ')
            data[item+1] = data[item+1].replace('\n',' ')
            random_samples = data[item] + data[item+1]
            prompt = prompt + random_samples

        # pass this prompt to the curl command to gpt3
        curl_req = 'curl https://api.openai.com/v1/completions \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ' + key + '" \
        -d \'{"model": "text-davinci-002", "prompt": ' + '"' + prompt + '"' +', "temperature": 1, "max_tokens": 150}\''
        
        # Getting resposnse from gpt3
        gpt3result = subprocess.check_output(curl_req, shell=True)
        gpt3result = json.loads(gpt3result)

        # extracting the output text from gpt3 response
        gpt3result = gpt3result['choices'][0]['text']
    
        # dumping the extracted text to a text file
        file = open("data/neg-1500-simp-gen-unfiltered.txt", "a") 
        file.write(gpt3result)
        # Next the neg-1500-simp-gen-unfiltered.txt file is manually cleaned

def main():
    args = get_arguments()
    key = args.key
    if args.dataset == 'negsimp_template':
        generate_negsimp_template()
    elif args.dataset == 'negsimp_gen':
        generate_negsimp_gen(key)
    elif args.dataset == 'role':
        generate_role(key)
        

if __name__ == "__main__":
    main()