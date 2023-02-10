from dis import Instruction
from numpy import mod
import transformers
import torch
import argparse
import re
import subprocess
import json
import ast


def get_args():
    """ Set hyperparameters """
    parser = argparse.ArgumentParser()
  
    parser.add_argument("file_path", help="file path of the extended dataset e.g. data/neg_simp_generated.txt")
    parser.add_argument("model_name_or_path", help="Huggingface pretrained model name/path")
    parser.add_argument("--prediction", type=bool, default=False, help="set it to True if we need predictions too")
    parser.add_argument("--key", help="key for openai gpt3")
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    return args

def process_data(file_path, modeldir):
    """Convert the sentences into source and label

    Arguments 
        file_path (string) : file path
        modeldir (string) : model name
    
    Return
        source (list) : convert e.g.The librarian documented which journalist the celebrities had avoided, 
                                    The librarian documented which celebrities the journalist had interviewed, 

                            to
        
                            ['The librarian documented which journalist the celebrities had [MASK]',
                             'The librarian documented which celebrities the journalist had [MASK]']
        
        label (list) : e.g.     ['avoided', 'interviewed']
    """
    source = []
    label = []
    
    # Read line by line, remove (, ), , and replace last word with [MASK] and add to source list
    # Append target word to label list
    with open(file_path) as f:
        text = f.readlines()
        for line in text:
            line= line.replace(',', "")
            splitted_line = line.split(" ")
            line_label = splitted_line[-1]
            if modeldir.startswith("roberta"):
                line_source = line.replace(line_label, '<mask> .')
            elif modeldir.startswith("gpt"):
                line_source = line.replace(line_label, '')
            elif modeldir.startswith('t5'):
                line_source = line.replace(line_label, '<extra_id_0>.')
            else:
                line_source = line.replace(line_label, '[MASK].')
            label.append(line_label.rstrip("\n"))
            source.append(line_source)
    return source, label

def predictions(modeldir, device, source, label, k, file_path):
    """
    Generates prediction for the generated dataset and store them in a file
    Arguments:

    modeldir [str] : model name
    device [str] : cpu or cuda
    source [list] : list of source sentences e.g. The librarian documented which journalist the celebrities had [MASK]'
    label [list] : list of gold labels
    k [int] : top k value minimum value 10 
    file_path [str] : file path where original source and label are present
    """    
    # Define model and tokenizer
    print("Running experiment for ", modeldir)
    tokenizer = transformers.AutoTokenizer.from_pretrained(modeldir)
    if modeldir.startswith("gpt2"):
        model = transformers.GPT2LMHeadModel.from_pretrained(modeldir).to(device)
    elif modeldir.startswith("t5"):
        model = transformers.T5ForConditionalGeneration.from_pretrained(modeldir).to(device)
    else: # for BERT, RoBERTa, ALBERT, DISTILBERT
        model = transformers.AutoModelForMaskedLM.from_pretrained(modeldir).to(device)
    model.eval()
    
    # Getting top k predictions from the model
    top_predictions = []
    # x = 5 # index used to pick predictions, it only changes for roberta as its first predcition is always a space or ',' or '.'
    index = 0
    
    for item in source:
        # item is e.g. 'The librarian documented which journalist the celebrities had [MASK]' 
        tokenized_input = tokenizer(item, return_tensors="pt")
        tokenized_input = tokenized_input.to(device)
        print(device)
        if modeldir.startswith('t5'):
            decoder_ids = tokenizer("<pad> <extra_id_0>", add_special_tokens=False, return_tensors="pt").input_ids
            decoder_ids = decoder_ids.to(device) 
            predictions = model(input_ids=tokenized_input.input_ids, decoder_input_ids=decoder_ids)
        else:
            predictions = model(**tokenized_input)

        if modeldir.startswith('roberta'):
            token = '<mask>'
            x = 6
            index = 1
        else:
            token ='[MASK]'

        if modeldir.startswith('gpt'):
            mask_index = -2  # -2 is position of last token
            index = 1
        elif modeldir.startswith('t5'): 
            mask_index = 1   # 1 is position of <extra_id_0>
        else: 
            for i, tok in enumerate(tokenized_input['input_ids'].reshape(-1)):
                if tok == tokenizer.convert_tokens_to_ids(token): 
                    mask_index = i       
                    
        predictions = predictions.logits

        softpred = torch.softmax(predictions[0, mask_index],0)
        top_inds = torch.argsort(softpred, descending=True)[:k].cpu().numpy()
        top_tok_preds = tokenizer.decode(top_inds, skip_special_tokens=True)
        top_predictions.append(top_tok_preds)
  
    # Save all prediction in a file, for negation it includes both affirmative and negation predictions
    filename = file_path.split('/')[-1].split('.')[0]
        
    file_allpred = open("predictions/{}/{}.txt".format(filename, modeldir), 'w')
    for i in range(len(top_predictions)):
        list_top_pred = top_predictions[i].split(' ') # e.g. ['fish', 'trout', 'species', 'mineral', 'protein']
        file_allpred.writelines([str(list_top_pred),'\n'])
    print("prediction saved for ", modeldir)

def evaluation(file_path, modeldir, label):
    '''
    Computes accuracy and sensitivity for the model predictions

    Arguments:

    file_path [str] : file path where original source and label are present
    modeldir [str] : model name
    label [list] : list of gold labels
    '''
    topkindex = 5
    index = 0
    # Open files    
    file = open("result.txt", 'a')

    file_path = file_path.split('/')   # e.g. NEG-1500-SIMP.txt
    dataset = file_path[1].split('.')[0]   # e.g. NEG-1500-SIMP

    # Open files where predictions are stored
    pred_file = open("predictions/{}/{}.txt".format(dataset, modeldir), 'r')
    lines = pred_file.readlines()
    top_predictions = []
    for line in lines:
        line  = ast.literal_eval(line)
        top_predictions.append(line)

    # set the index
    if ('roberta' in modeldir) or ('gpt' in modeldir):
        index = 1
        topkindex = 6

    
    # #of flips for negation
    if 'NEG' in dataset:
        flip = 0
        file_allsens = open('sensitivity.txt', 'a')
        for i in range(0, len(top_predictions), 2):
            list_top_pred = top_predictions[i][index] # e.g. ['fish', 'trout', 'species', 'mineral', 'protein']
            neg_list_top_pred  = top_predictions[i+1][index]
            if list_top_pred != neg_list_top_pred:
                flip += 1
        file_allsens.writelines(["{} --->", modeldir , " | #flipped = ", str(flip), "\n".format(dataset)])
    
    # take all sentences for role and only affirmative for neg-simp
    step = 1
    if 'NEG' in dataset:
        step = 2 # for neg as we need only affirmative sentences
        
    # for ROLE-1500
    # Accuracy for top 1, 5, 10 and 20 predictions
    topkmatch = 0
    top10match = 0
    top5match = 0
    top1match = 0

    for i in range(0, len(top_predictions), step):
        list_top_pred = top_predictions[i] # e.g. ['fish', 'trout', 'species', 'mineral', 'protein']
        if label[i] in list_top_pred:
            topkmatch += 1
        if label[i] in list_top_pred[:10]:
            top10match += 1
        if label[i] in list_top_pred[:topkindex]:
            top5match += 1
        if label[i] == list_top_pred[index]:
            top1match += 1
            
    # print(topkmatch)
    topk_accuracy = step * topkmatch / (len(top_predictions))
    top10_accuracy = step * top10match / (len(top_predictions))
    top5_accuracy = step * top5match / (len(top_predictions))
    top1_accuracy = step * top1match / (len(top_predictions))

    print("model = ", modeldir)
    print("Top 20 match = ", topk_accuracy)
    print("Top 10 match = ", top10_accuracy)
    print("Top 5 match = ", top5_accuracy)
    print("Top 1 match = ", top1_accuracy)
    
    file.writelines([dataset," | ", modeldir, " | ", str(topk_accuracy),  " | ", str(top10_accuracy),  " | ", str(top5_accuracy), " | ", str(top1_accuracy), '\n\n'])
    print("Completed experiment for ", modeldir)

def gpt3_predictions(file_path, key):
    """
    Get top 5 predictions from Open AI gpt3 for generated datasets
    Arguments
        file_path
        key : OPEN AI gpt 3 key
    
    Description
        top1pred [list] : top 5 predictions from gpt3 
        top5pred [list] : top 5 predictions from gpt3 . Note top 5 are not in decreaing order of prob thats 
                                why we have top1 different than first element of top5.
    """
    
    source, label = process_data(file_path, 'gpt')
    top1pred = []
    top5pred = []
    Instruction = "The goal is to complete the given sentence with one english word. Avoid punctuation or new line or space."
    for item in source:
    # item is e.g. 'The librarian documented which journalist the celebrities had '
        
        item = item.strip()
        prompt = Instruction + item 
    
        curl_req = 'curl https://api.openai.com/v1/completions \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ' + key + '" \
        -d \'{"model": "text-davinci-002", "prompt": ' + '"' + prompt + '"' +', "temperature": 0.5, "n": 1, "max_tokens": 1, "logprobs": 5}\''

        # # Getting resposnse from gpt3
        gpt3result = subprocess.check_output(curl_req, shell=True)
        gpt3result = json.loads(gpt3result)

        # extracting the output text from gpt3 response
        top1pred_gpt3 = gpt3result['choices'][0]['logprobs']['tokens'][0]
        top5pred_gpt3 = gpt3result['choices'][0]['logprobs']['top_logprobs'][0]
        top5pred_gpt3 = list(top5pred_gpt3.keys())
        top1pred.append(top1pred_gpt3)
        top5pred.append(top5pred_gpt3)

        dataset = file_path.split('/')[-1].split('.')[0] 

        file = open("predictions/{}/gpt3_top1_prediction.txt".format(dataset), 'w')
        filetop5 = open("predictions/{}/gpt3_top5_prediction.txt".format(dataset), 'w')

    for i in range(0, len(label)):
        top5pred[i] = [item.strip() for item in top5pred[i]]
        file.writelines([top1pred[i].replace("\n",''),"\n"])
        filetop5.writelines([str(top5pred[i]),"\n"])

def gpt3_evaluation(label, file_path):
    """
    modeldir [str] : model name
    key [str] :open ai gpt3 key
    source [list] : list of source sentences e.g. The librarian documented which journalist the celebrities had [MASK]'
    label [list] : list of gold labels
    file_path [str] : file path where original source and label are present
    """

    file_path = file_path.split('/')       # e.g. NEG-1500-SIMP.txt
    dataset = file_path[1].split('.')[0]   # e.g. NEG-1500-SIMP

    # Read the predictions. Top 1 and top 5 are different as top 5 are not in order of probability.
    pred_file_top1 = open("predictions/{}/gpt3_top1_prediction.txt".format(dataset), 'r')
    pred_file_top5 = open("predictions/{}/gpt3_top5_prediction.txt".format(dataset), 'r')

    top1matchgpt3 = 0
    top5matchgpt3 = 0
    flipped = 0 # for neg sensitivity
    step = 1
   
    if 'NEG' in dataset:
        step = 2  

    # Open files where predictions are stored
    lines = pred_file_top1.readlines()
    top1_predictions = []
    for line in lines:
        top1_predictions.append(line)

    lines = pred_file_top5.readlines()
    top5_predictions = []
    for line in lines:
        line  = ast.literal_eval(line)
        top5_predictions.append(line)

    # Compute accuracy
    for i in range(0, len(top1_predictions), step):
        if label[i] == top1_predictions[i].strip():
            top1matchgpt3 +=1
            if 'neg' in dataset:
                top1_predictions[i] != top1_predictions[i+1]
                flipped += 1

        
        if label[i] in top5_predictions[i]:
            top5matchgpt3 += 1

    file = open("result.txt", 'a')
    file.writelines([dataset," top1 accuracy = ", str(step*top1matchgpt3/len(label)), " | ", "top5 accuracy = ",  str(step*top5matchgpt3/len(label)), "\n"])

    print("flipped", flipped)
    print("top1 match", top1matchgpt3)
    if "NEG" in file_path[1]:
        if top1matchgpt3 != 0:
            file.writelines(["  flipped ",str(flipped), " | top1match=", str(top1matchgpt3), "\n"])

def main():
    args = get_args()
    file_path = args.file_path
    modeldir = args.model_name_or_path
    key = args.key
    device = args.device
    source, label = process_data(file_path, modeldir)
    k = 20
   
    if modeldir.startswith("gpt3"):
        if args.prediction == False:
            gpt3_predictions(file_path, key) # get predictions from gpt3
        gpt3_evaluation(label, file_path) # evaluate gpt3
    else:
        if args.prediction == False:
            predictions(modeldir, device, source, label, k, file_path)
        evaluation(file_path, modeldir, label)
 
if __name__ == '__main__':
    main()