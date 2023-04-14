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
def output_probs_helper(in_output_probs_filename):
    with open(in_output_probs_filename, 'w', encoding="utf-8") as outputfile:
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

    output_probs_helper(in_output_probs_filename)

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

'''
Q4a) Considering START and STOP state for trans_probs.txt
'''

def viterbi_helper(in_output_probs_filename, in_trans_probs_filename):
    output_probs_helper(in_output_probs_filename)
    trans_dict = defaultdict(lambda: defaultdict(float))
    with open("twitter_train.txt", "r", encoding="utf-8") as testdata:
        with open(in_trans_probs_filename, "w", encoding="utf-8") as trans_probs:
                #first line only
                tag_i = "START"
                tag_j = testdata.readline().strip().split("\t")[1]
                trans_dict[tag_i][tag_j] = 1

                tweet_counter = 0

                for line in testdata:
                    #for all lines until second last
                    if not line.isspace(): #transition from one state to another
                        if tag_j == "STOP" : #if previous line was a blank
                            tag_j = "START"
                        tag_i = tag_j
                        tag_j = line.strip().split("\t")
                        if len(tag_j) == 1: #end of test data reached
                            break 
                        tag_j = tag_j[1]
                        trans_dict[tag_i][tag_j] += 1
                    else: #transition from one state to STOP state
                        tag_i = tag_j
                        tag_j = "STOP"
                        trans_dict[tag_i][tag_j] += 1
                        tweet_counter += 1

                label_tag_counter["START"] = tweet_counter
                label_tag_counter["STOP"] = tweet_counter
            
                for tag_i, inner_dict in trans_dict.items():
                    for tag_j, count in inner_dict.items():
                        prob = (count + delta)/(label_tag_counter[tag_i] + delta * (total_num_words + 1)) #smoothing formula where delta = 1
                        trans_probs.write(f"{tag_i}\t{tag_j}\t{prob}\n")
                      
'''
Q4c) accuracy = 1034/1378 = 0.7503628447024674
'''
def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    
    viterbi_helper(in_output_probs_filename, in_trans_probs_filename)

    output_probs_dict = defaultdict(lambda: defaultdict(float))
    trans_probs_dict = defaultdict(lambda: defaultdict(float))

    with open(in_output_probs_filename, 'r', encoding="utf-8") as probinputfile:
        for line in probinputfile:
            tag, token, prob = line.strip().split('\t')
            output_probs_dict[token][tag] = float(prob)
    
    with open(in_trans_probs_filename, "r", encoding="utf-8") as transinputfile:
        for line in transinputfile:
            tag_i, tag_j, prob = line.strip().split("\t")
            trans_probs_dict[tag_i][tag_j] = float(prob)
    
    with open(in_tags_filename, "r", encoding="utf-8") as tagsfile:
        tag_list = tagsfile.read().split()
    
    with open(in_test_filename, "r", encoding="utf-8") as testdata:
        with open(out_predictions_filename, "w", encoding="utf-8") as outputfile:
            pi = defaultdict(lambda: defaultdict(float))
            BP = defaultdict(lambda: defaultdict(float))
            tokens = []

            for line in testdata:
                if not line.isspace(): #gathering all tokens of one tweet in a list
                    tokens.append(line.strip())
                else: #when the end of the tweet is reached
                    for index in range(0, len(tokens) + 1): #processing all the tokens in the tweet
                        if index <= len(tokens) - 1:
                            token = tokens[index]
                            trans_prob = 0
                            output_prob = 0
                            if index == 0: #for the first word of the tweet
                                for tag_j in tag_list:
                                    if tag_j in trans_probs_dict["START"]:
                                        trans_prob = trans_probs_dict["START"][tag_j]
                                    else:
                                        trans_prob = delta / (label_tag_counter["START"] + delta * (total_num_words + 1))
                                    if token in output_probs_dict:
                                        output_prob = output_probs_dict[token][tag_j]
                                    else:
                                        output_prob = delta / (label_tag_counter[tag_j] + delta * (total_num_words + 1))
                                    prob = trans_prob * output_prob
                                    pi[index + 1][tag_j] = prob
                                    BP[index + 1][tag_j] = "*"

                            else: #for every other word in between including last word
                                for tag_j in tag_list:
                                    find_max = defaultdict(float)
                                    for tag_i in tag_list:
                                        prev_prob = pi[index][tag_i]
                                        if tag_j in trans_probs_dict[tag_i]:
                                            trans_prob = trans_probs_dict[tag_i][tag_j]
                                        else:
                                            trans_prob = delta / (label_tag_counter[tag_i] + delta * (total_num_words + 1))
                                        if token in output_probs_dict:
                                            output_prob = output_probs_dict[token][tag_j]
                                        else:
                                            output_prob = delta / (label_tag_counter[tag_j] + delta * (total_num_words + 1))
                                        prob = prev_prob * trans_prob * output_prob
                                        find_max[tag_i] = prob
                                    max_tag_i = max(find_max, key = find_max.get)
                                    max_prob = find_max[max_tag_i]
                                    pi[index + 1][tag_j] = max_prob
                                    BP[index + 1][tag_j] = max_tag_i
                        
                        else: #for the stop state
                            find_max = defaultdict(float)
                            for tag_i in tag_list:
                                prev_prob = pi[index][tag_i]
                                if "STOP" in trans_probs_dict[tag_i]:
                                    trans_prob = trans_probs_dict[tag_i]["STOP"]
                                else:
                                    trans_prob = delta / (label_tag_counter[tag_i] + delta * (total_num_words + 1))
                                prob = prev_prob * trans_prob
                                find_max[tag_i] = prob
                            final_BP = max(find_max, key=find_max.get)
                            max_prob = find_max[final_BP]

                            sequence = []
                            sequence.append(final_BP)
                            for i in reversed(range(1, index + 1)):
                                tag = BP[i][final_BP]
                                if tag == "*":
                                    continue
                                sequence.append(tag)
                                final_BP = tag
                            sequence.reverse()
                            for tag in sequence:
                                outputfile.write(f"{tag}\n")
                            outputfile.write("\n")

                            pi = defaultdict(lambda: defaultdict(float)) #reset
                            BP = defaultdict(lambda: defaultdict(float)) #reset
                            tokens = [] #reset

'''
Q5a) Observations we have made from twitter.txt:
    1. By ignoring capitalisation, it can increase the accuracy of the model. For example, "yeah", "Yeah" and "YEAH" all appears in the
     twitter_train.txt with the tag "!". Therefore, we can consolidate these into one probability under "yeah" so improve the prediction of the token
     with the tag "!".

     2. We observe that for this data, users are all defined as "@USER_XXXXX" with the text after the underscore being the unique ID to each user.
     But since every user should be tagged with "@", we can ignore the unique ID and check if the token starts with "@" as this will mean that the tag would
     also be "@". Furthermore, we observe that even the word "@" is tagged with "@" so there is not a need to check the subsequent characters. 

     3. All urls can be detected by checking if the token starts with "http". Thus, any token that starts with "http" has to be tagged with "U" and we can
     disregard all the characters afterwards.

     4. All retweets have a token "RT" and the tag "~". Thus, any token that is "RT" can be given the tag "~".

     If the above conditions are met, we will assign these tags with a probability of 1 in the pi matrix because we will need this state to have the maximum 
     probability when generating the pi value for the next token's tags so that the tag will be "selected" as a backpointer.

Q5c) accuracy = 1108/1378 = 0.8040638606676342
'''
##### Recompute the inital output probability by lower casing every token

label_tag_counter_new = defaultdict(float) #dict where {tag: count}
label_tag_token_new = defaultdict(lambda: defaultdict(float)) #dict where {tag:{token: count}}
label_token_new = defaultdict(float) #dict where {token: count}
unseen_token_tag_prob_new = defaultdict(float) #dict where {tag: probability}

total_num_words_new = 0 #instantiating total number of tokens in training data

with open("twitter_train.txt", 'r', encoding="utf-8") as inputfile:
    for line in inputfile: #from training data, populate the first 3 dicts
        if not line.isspace():
            token, tag = line.strip().split('\t')
            
            ##### change all token to lower case
            token = token.lower()
            
            label_tag_counter_new[tag] += 1
            label_tag_token_new[tag][token] += 1
            label_token_new[token] += 1

    total_num_words_new = len(label_token_new)

    for tag, tag_value in label_tag_counter_new.items(): #populate 4th dict, probability of each tag for all unseen tokens
        prob = delta/(tag_value + delta * (total_num_words_new + 1))
        unseen_token_tag_prob_new[tag] = prob
       
def output_probs_helper2(in_output_probs_filename):
    with open(in_output_probs_filename, 'w', encoding="utf-8") as outputfile:
        for tag, tag_value in label_tag_counter_new.items():
            for token, token_value in label_tag_token_new[tag].items():
                prob = (token_value + delta) / (tag_value + delta * (total_num_words_new + 1))
                outputfile.write(f"{tag}\t{token}\t{prob}\n")

def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):

    ##### override the old output_probs txt with lowercases token probabilities
    viterbi_helper(in_output_probs_filename, in_trans_probs_filename)
    output_probs_helper2(in_output_probs_filename)

    output_probs_dict = defaultdict(lambda: defaultdict(float))
    trans_probs_dict = defaultdict(lambda: defaultdict(float))

    with open(in_output_probs_filename, 'r', encoding="utf-8") as probinputfile:
        for line in probinputfile:
            tag, token, prob = line.strip().split('\t')
            output_probs_dict[token][tag] = float(prob)

    with open(in_trans_probs_filename, "r", encoding="utf-8") as transinputfile:
        for line in transinputfile:
            tag_i, tag_j, prob = line.strip().split("\t")
            trans_probs_dict[tag_i][tag_j] = float(prob)
    
    with open(in_tags_filename, "r", encoding="utf-8") as tagsfile:
        tag_list = tagsfile.read().split()
    
    with open(in_test_filename, "r", encoding="utf-8") as testdata:
        with open(out_predictions_filename, "w", encoding="utf-8") as outputfile:
            pi = defaultdict(lambda: defaultdict(float))
            BP = defaultdict(lambda: defaultdict(float))
            tokens = []

            for line in testdata:
                if not line.isspace(): #gathering all tokens of one tweet in a list
                    tokens.append(line.strip())
                else: #when the end of the tweet is reached
                    for index in range(0, len(tokens) + 1): #processing all the tokens in the tweet
                        if index <= len(tokens) - 1:
                            token = tokens[index].lower() ##### change the incoming tokens to lowercases as well
                            trans_prob = 0
                            output_prob = 0
                            if index == 0: #for the first word of the tweet
                                for tag_j in tag_list:
                                    if tag_j in trans_probs_dict["START"]:
                                        trans_prob = trans_probs_dict["START"][tag_j]
                                    else:
                                        trans_prob = delta / (label_tag_counter["START"] + delta * (total_num_words + 1))
                                    if token in output_probs_dict:
                                        output_prob = output_probs_dict[token][tag_j]
                                    else:
                                        output_prob = delta / (label_tag_counter[tag_j] + delta * (total_num_words + 1))
                                    prob = trans_prob * output_prob

                                    ##### check if token was @user_XXXX
                                    if token[0] == "@" and tag_j == "@":
                                        pi[index + 1]["@"] = 1
                                        BP[index + 1]["@"] = "*"
                                    ##### check if url
                                    elif (len(token) > 4 and token[:4] == "http") and tag_j == "U":
                                        pi[index + 1]["U"] = 1
                                        BP[index + 1]["U"] = "*"
                                    ##### check if retweet
                                    elif token == "rt" and tag_j == "~":
                                        pi[index + 1]["~"] = 1
                                        BP[index + 1]["~"] = "*"
                                    else: 
                                        pi[index + 1][tag_j] = prob
                                        BP[index + 1][tag_j] = "*"

                            else: #for every other word in between including last word
                                for tag_j in tag_list:
                                    find_max = defaultdict(float)
                                    for tag_i in tag_list:
                                        prev_prob = pi[index][tag_i]
                                        if tag_j in trans_probs_dict[tag_i]:
                                            trans_prob = trans_probs_dict[tag_i][tag_j]
                                        else:
                                            trans_prob = delta / (label_tag_counter[tag_i] + delta * (total_num_words + 1))
                                        if token in output_probs_dict:
                                            output_prob = output_probs_dict[token][tag_j]
                                        else:
                                            output_prob = delta / (label_tag_counter[tag_j] + delta * (total_num_words + 1))
                                        prob = prev_prob * trans_prob * output_prob
                                        find_max[tag_i] = prob
                                    max_tag_i = max(find_max, key = find_max.get)
                                    max_prob = find_max[max_tag_i]
                                    
                                    ##### check if token was @user_XXXX
                                    if token[0] == "@" and tag_j == "@":
                                        pi[index + 1]["@"] = 1
                                        BP[index + 1]["@"] = max_tag_i
                                    #### check if token is url
                                    elif (len(token) > 4 and token[:4] == "http") and tag_j == "U":
                                        pi[index + 1]["U"] = 1
                                        BP[index + 1]["U"] = max_tag_i
                                    #### check if retweet
                                    elif token == "rt" and tag_j == "~":
                                        pi[index + 1]["~"] = 1
                                        BP[index + 1]["~"] = max_tag_i
                                    else: 
                                        pi[index + 1][tag_j] = max_prob
                                        BP[index + 1][tag_j] = max_tag_i
                        
                        else: #for the stop state
                            find_max = defaultdict(float)
                            for tag_i in tag_list:
                                prev_prob = pi[index][tag_i]
                                if "STOP" in trans_probs_dict[tag_i]:
                                    trans_prob = trans_probs_dict[tag_i]["STOP"]
                                else:
                                    trans_prob = delta / (label_tag_counter[tag_i] + delta * (total_num_words + 1))
                                prob = prev_prob * trans_prob
                                find_max[tag_i] = prob
                            final_BP = max(find_max, key=find_max.get)
                            max_prob = find_max[final_BP]

                            sequence = []
                            sequence.append(final_BP)
                            for i in reversed(range(1, index + 1)):
                                tag = BP[i][final_BP]
                                if tag == "*":
                                    continue
                                sequence.append(tag)
                                final_BP = tag
                            sequence.reverse()
                            for tag in sequence:
                                outputfile.write(f"{tag}\n")
                            outputfile.write("\n")

                            pi = defaultdict(lambda: defaultdict(float)) #reset
                            BP = defaultdict(lambda: defaultdict(float)) #reset
                            tokens = [] #reset

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

    trans_probs_filename =  f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')

if __name__ == '__main__':
    run()
