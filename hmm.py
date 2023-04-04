'''
Done by:
Kaavya Senthil Kumar (A0246227E)
Lam Wen Jett (A0234935Y)
Ng Han Leong Jordan (A0233839W)
Tasneem Benazir D/O Shahul Hameed (A0238339W)
'''

from collections import defaultdict

''''
GLOBAL VARIABLES using training data (twitter_train.txt)
'''
label_tag_counter = defaultdict(float) #dict where {tag: count}
label_tag_token = defaultdict(lambda: defaultdict(float)) #dict where {tag:{token: count}}
label_token = defaultdict(float) #dict where {token: count}

unseen_token_tag_prob = defaultdict(float) #dict where {tag: probability}

delta = 1 # change from 0.01, 0.1, 1, 10 etc to test

total_num_words = 0 #instantiating total number of tokens in training data

with open("twitter_train.txt", 'r', encoding="utf-8") as inputfile:
    for line in inputfile: #from training data, populate the first 3 dicts
        if not line.isspace():
            token, tag = line.strip().split('\t')
            #token = token.lower()
            label_tag_counter[tag] += 1
            label_tag_token[tag][token] += 1
            label_token[token] += 1

    total_num_words = len(label_token)

    for tag, tag_value in label_tag_counter.items(): #populate 4th dict, probability of each tag for all unseen tokens
        prob = delta/(tag_value + delta * (total_num_words + 1))
        unseen_token_tag_prob[tag] = prob

'''
Q2a) helper function to write to naive_output_probs.txt

    using the dicts, find the emission probability of each tag for each token in the training data and write it to naive_output_probs.txt

    the file is sorted according to tags

    unseen token probability is not added
'''
def naive_output_probs():
    with open("naive_output_probs.txt", 'w', encoding="utf-8") as outputfile:
        for tag, tag_value in label_tag_counter.items():
            for token, token_value in label_tag_token[tag].items():
                prob = (token_value + delta) / (tag_value + delta * (total_num_words + 1))
                outputfile.write(f"{tag}\t{token}\t{prob}\n")

# Implement the six functions below
'''
Q2b) Predicting tags for test data (twitter_dev_no_tag.txt) using naive_output_probs.txt

Q2c) accuracy = 908/1378 = 0.6589259796806967
'''
def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):

    naive_output_probs()

    label_tag_prob = defaultdict(lambda: defaultdict(float)) #dict where {token: {tag: prob}} using naive_output_probs.txt
    most_likely_tag_for_unseen_tokens = max(unseen_token_tag_prob, key=unseen_token_tag_prob.get)
    
    with open(in_output_probs_filename, 'r', encoding="utf-8") as probinputfile:
        for line in probinputfile: #populating the above dict
            tag, token, prob = line.strip().split('\t')
            label_tag_prob[token][tag] = prob

    with open(in_test_filename,'r', encoding="utf-8") as tokeninputfile: #opening test data (twitter_dev_no_tag.txt)
        with open(out_prediction_filename, 'w', encoding="utf-8") as tagoutputfile: #writing to output file (naive_predictions.txt)
            for line in tokeninputfile:
                if line.isspace(): #for empty lines
                    tagoutputfile.write("\n")
                else:
                    token = line.strip()
                    #token = token.lower()
                    if token in label_tag_prob:
                        most_likely_tag = max(label_tag_prob[token], key = label_tag_prob[token].get)
                        tagoutputfile.write(f"{most_likely_tag}\n")
                    else:
                        tagoutputfile.write(f"{most_likely_tag_for_unseen_tokens}\n")       

'''
Q3a) Compute the RHS using for a given word by finding the probability for each tag from naive_output_probs.txt and multiplying it with
    the count of that tag in twitter_train.txt, then find the most probable tag by finding the max value out of all the tags. 
    
    Mathematically, it is computed as argmax j (P(x=w|y=j)*P(y=j))/P(x=w). For a seen token, P(x=w|y=j) is taken from naive_output_probs.txt
    and it is multiplied with P(y=j) which is the count of that respective tag in twitter_train.txt. For all unseen tokens, P(x=w|y=j) is 
    obtained via the smoothing formula, and then multiplied with P(y=j) as above. For all tokens, P(x=w) can be ignored as it is a constant
    for a given word and we are maximising.

Q3c) accuracy = 952/1378 = 0.690856313497823
'''
def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):

    most_likely_tag_unseen_token = "" #for most likely tag for unseen tokens
    max_prob = 0

    for tag in unseen_token_tag_prob:
        current_prob = unseen_token_tag_prob[tag] * label_tag_counter[tag]
        if current_prob > max_prob:
            most_likely_tag_unseen_token = tag
            max_prob = current_prob
    
    label_tag_prob = defaultdict(lambda: defaultdict(float)) #dict where {token: {tag: prob}} using naive_output_probs.txt

    with open(in_output_probs_filename, 'r', encoding="utf-8") as probinputfile:
        for line in probinputfile: #populating the above dict
            tag, token, prob = line.strip().split('\t')
            label_tag_prob[token][tag] = prob
    
    with open(in_test_filename,'r', encoding="utf-8") as tokeninputfile: #opening test data (twitter_dev_no_tag.txt)
        with open(out_prediction_filename, 'w', encoding="utf-8") as tagoutputfile: #writing to output file (naive_predictions2.txt)
            for line in tokeninputfile:
                if line.isspace():
                    tagoutputfile.write("\n")
                else:
                    token = line.strip()
                    #token = token.lower()
                    if token in label_tag_prob:
                        most_likely_tag = ""
                        max_prob = 0
                        for tag in label_tag_prob[token]:
                            current_prob = float(label_tag_prob[token][tag]) * label_tag_counter[tag]
                            if current_prob > max_prob:
                                most_likely_tag = tag
                                max_prob = current_prob
                        tagoutputfile.write(f"{most_likely_tag}\n")
                    else:
                        tagoutputfile.write(f"{most_likely_tag_unseen_token}\n")
    pass

'''
to be done
'''
def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    pass

def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):
    pass



def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)



def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''

    ddir = 'c:/Users/tasne/OneDrive/Desktop/Y2S2/BT3102/Project/project-files' #your working dir

    in_train_filename = f'{ddir}/twitter_train.txt'

    naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'

    naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    # trans_probs_filename =  f'{ddir}/trans_probs.txt'
    # output_probs_filename = f'{ddir}/output_probs.txt'

    # in_tags_filename = f'{ddir}/twitter_tags.txt'
    # viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    # viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
    #                 viterbi_predictions_filename)
    # correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    # print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    # trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    # output_probs_filename2 = f'{ddir}/output_probs2.txt'

    # viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    # viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
    #                  viterbi_predictions_filename2)
    # correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    # print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')

    
    
if __name__ == '__main__':
    run()
