from flask import Flask, jsonify, request

import logging
import numpy as np
import pandas as pd
import re
import spacy
import torch
import tokenizations
from transformers import BertModel

from scipy.special import softmax
from spacy.lang.en import English

import config
import openie_systems as openies
import util

from run import Runner
from tensorize import CorefDataProcessor
from common_utils import *
from term_selection_module import Classifier

app = Flask(__name__)

logging.getLogger().setLevel(logging.CRITICAL)

def setup(config_name, model_identifier, gpu_id):
    runner = Runner(config_name, gpu_id)
    model = runner.initialize_model(model_identifier)
    data_processor = CorefDataProcessor(runner.config)

    # Interactive input
    model.to(model.device)
    nlp = English()
    nlp.add_pipe('sentencizer')
    bert_tokenizer = data_processor.tokenizer
    return runner, model, data_processor, nlp, bert_tokenizer

def predict(doc, runner, model, data_processor, nlp, bert_tokenizer, seg_len=512):
    doc = get_document_from_string(str(doc), seg_len, bert_tokenizer, nlp) # Doc to Token
    tensor_examples, _ = data_processor.get_tensor_examples_from_custom_input([doc]) #Token to vector tensors
    predicted_clusters, _, _ = runner.predict(model, tensor_examples)
    subtokens = util.flatten(doc['sentences'])
    mentions = []
    for cluster in predicted_clusters[0]:
        mentions_str = [' '.join(subtokens[m[0]:m[1]+1]) for m in cluster]
        mentions.append(mentions_str)
    return doc, subtokens, predicted_clusters, mentions

def align_cluster(ori_text, cluster, bert_tokenizer, spacy_tokenizer):
    doc = get_document_from_string(ori_text, 512,bert_tokenizer, spacy_tokenizer)
    bert_tokens = util.flatten(doc['sentences'])
    spacy_tokens=[]
    spacy_doc = spacy_tokenizer(ori_text)
    for tokens in spacy_doc:
        spacy_tokens.append(tokens.text)
    a2b, _ = tokenizations.get_alignments(bert_tokens, spacy_tokens)
    recons = []
    for i in cluster:
        clustList = []
        for j in i:
            curPair=[]
            span = a2b[j[0]:j[1]+1]
            tempLen = len(span)
            start_pos = 0
            end_pos=tempLen-1
            while len(span[start_pos]) == 0:
                start_pos+=1
            while len(span[end_pos]) == 0:
                end_pos-=1
            curPair.append(span[start_pos][0])
            curPair.append(span[end_pos][0]+1)
            clustList.append(curPair)
        recons.append(clustList)
    return recons


def span_model_nn(df, tokenizer, model, max_len, device, use_gpu=True):
    for i, rows in df.iterrows():
        clust_text = split_text(rows['text'])
        for j, text in enumerate(clust_text):
            final_text = process_span(text)
            clust_text[j] = final_text
        clust_tokens = tokenizer.batch_encode_plus(clust_text, max_length = max_len, pad_to_max_length=True, truncation=True,return_token_type_ids=False)
        clust_seq = torch.tensor(clust_tokens['input_ids'])
        clust_mask = torch.tensor(clust_tokens['attention_mask'])
        with torch.no_grad():
            if use_gpu:
                preds = model(clust_seq.to(device), clust_mask.to(device))
                preds = preds.detach().cpu().numpy()
            else:
                preds = model(clust_seq, clust_mask)
                preds = preds.detach().numpy()
            preds = softmax(preds, axis=1)
            index = np.argmax(preds[:,1])
            conf =  preds[:,1][index]
            
        ori_index = index
        if conf < 0.5:
            index = -2
        df.at[i, 'Select'] = int(index)
        df.at[i, 'confidence'] = conf # For debugging purpose
        print("PREDICTION:", index, clust_text[ori_index], "Confidence =", conf)
    return df


device = torch.device("cuda")

print("Setting up Coreference Resolution Model...")
configname = config.coref_bert
modelid = config.coref_pretrained
gpuid = config.gpu_id
runner, model, data_processor, nlp, bert_tokenizer = setup(configname, modelid, gpuid)


print("Setting up Span Selection Model...")
configname = config.term_selection_bert
bert_config = util.initialize_config(configname)
classifier_config = load_json(config.ffnn_params)
model_path = config.ffnn_model
model_param = classifier_config[config.ffnn_model.split('/')[-1]]
max_seq_len=model_param['max_len']
bert = BertModel.from_pretrained(bert_config['bert_pretrained_name_or_path'])
cmodel = Classifier(bert)
cmodel.load_state_dict(torch.load(model_path))
cmodel = cmodel.to(device)

nlp_lg = spacy.load("en_core_web_lg")
print("Connecting to OpenIE4...")
openie = openies.OpenIE_4()


@app.route('/build', methods =['POST'])
def build():
    res_dict = {}
    #Stage 1 Coref Resolution
    clusterdf = pd.DataFrame(columns=["article_id", "cluster_id", "spans", "text"])
    textjson = request.json
    text = textjson['text']
    text = clean_article(text)
    cur_article = remove_URL_email(text)
    if len(cur_article)> config.article_length_limit: # Discard article if length exceeds > certain characters
        print("No article processed")
        return jsonify(res_dict)
    # print(cur_article)
    _, _, cluster, menStr = predict(cur_article, runner, model, data_processor, nlp, bert_tokenizer)
    cluster_id=0
    for i, clust in enumerate(menStr):
        temp_dict = {'article_id': 1,
                    'cluster_id': cluster_id,
                    'spans': str(list(cluster[0][i])),
                    'text': str(clust)}
        clusterdf=clusterdf.append(temp_dict, ignore_index=True)
        cluster_id+=1
    torch.cuda.empty_cache()
    # print(clusterdf)

    # Stage 2 Find Representative Term
    df = span_model_nn(clusterdf, bert_tokenizer, cmodel, max_seq_len, device)
    article_clusters_resolved = {}
    article_id = '1'
    final_clust = []
    selected = []
    for i, rows in df.iterrows():
        this_clust = split_spans(rows['spans'])
        final_clust.append(this_clust)
        selected.append(rows['Select'])
    article_clusters_resolved[article_id] = {}
    article_clusters_resolved[article_id]['clusters'] = final_clust
    article_clusters_resolved[article_id]['selected'] = selected
    # Align tokens
    aligned_cluster = align_cluster(cur_article,
                                article_clusters_resolved['1']['clusters'],
                                bert_tokenizer,
                                nlp_lg)
    article_clusters_resolved['1']['clusters']=aligned_cluster

    # Stage 3
    clust = article_clusters_resolved
    text_body = clean_sent(cur_article)
    doc=nlp_lg(text_body)
    span_by_sentence = []
    articles_extraction =[]
    ori_triples=[]
    for sent in doc.sents: # Process Sentence by sentence
            spans_in_this_sentence = []
            from_cluster = []
            resolve_as = []
            sentence_start = sent[0].i
            sentence_end = sent[-1].i
            # Populate spans that are in this sentence from coref results
            for index, j in enumerate(clust['1']['clusters']):
                for k in j:
                    if k[0] >= sentence_start and k[1]<=sentence_end:
                        spans_in_this_sentence.append(k)
                        from_cluster.append(index)
                        resolve_as.append(clust['1']['selected'][index])
            cleaned_sent = str(sent)
            # Deduplicate sub-spans
            sub_span = [True]*len(spans_in_this_sentence)
            for index1, z in enumerate(spans_in_this_sentence):
                for index2, y in enumerate(spans_in_this_sentence):
                    if index1==index2:
                        continue
                    if z[0]>=y[0] and z[1]<=y[1]:
                        sub_span[index1]=False
            sub_span_len = len(sub_span)
            for index1, z in enumerate(reversed(sub_span)):
                if not z:
                    spans_in_this_sentence.pop(sub_span_len-index1-1)
                    from_cluster.pop(sub_span_len-index1-1)
                    resolve_as.pop(sub_span_len-index1-1)         

            # Extract Semantic Triples
            extracted_relations = get_rel_rest(openie, cleaned_sent)
            for ext in extracted_relations: #For Each Relation
                relation_dict = {}
                this_sub = untokenize(ext['sub'])
                this_rel = untokenize(ext['rel'])
                this_obj = untokenize(ext['obj'])
                # print("sub:", this_sub,"\nrel:", this_rel,"\nobj:", this_obj, '\nconf:', ext['conf'])
                relation_dict['conf'] = ext['conf']
                relation_dict['sub'] = this_sub
                relation_dict['rel'] = this_rel
                relation_dict['obj'] = this_obj
                ori_triples.append(dict(relation_dict))
                ori_sub = this_sub
                ori_obj = this_obj
                sub_editable = [1]*len(this_sub) # Used to track which part of subject has been edited
                obj_editable = [1]*len(this_obj) # Used to track which part of object has been edited
                further_edit_sub = True
                further_edit_obj = True
                for n, m in enumerate(spans_in_this_sentence): #Check each coref spans of this sentence against the extracted relation
                    this_text = str(doc[m[0]:m[1]])
                    clust_origin = from_cluster[n]
                    resolve_index = int(resolve_as[n])
                    if resolve_index == -2:
                        continue
                    resolved_span = clust['1']['clusters'][clust_origin][resolve_index]
                    resolved_text = str(doc[resolved_span[0]:resolved_span[1]])

                    
                    if this_text == this_sub: #this_text is the coreferent span
                        this_sub = resolved_text
                        further_edit_sub = False
                    elif bool(re.search(r'\b{}\b'.format(this_text), ori_sub)) and further_edit_sub:
                        matching_span = re.search(r'\b{}\b'.format(this_text), ori_sub).span() # Check against original subject
                        if all(sub_editable[matching_span[0]:matching_span[1]]): # Only portion of span that has not been overwritten before
                            this_sub = re.sub(r'\b{}\b'.format(this_text), r'{}'.format(resolved_text), this_sub)
                            sub_editable[matching_span[0]:matching_span[1]]=[0]*(matching_span[1]-matching_span[0]) # Mark part of text modified


                    if this_text == this_obj: #this_text is the coreferent span
                        this_obj = resolved_text
                        further_edit_obj = False
                    elif bool(re.search(r'\b{}\b'.format(this_text), ori_obj)) and further_edit_obj:
                        matching_span = re.search(r'\b{}\b'.format(this_text), ori_obj).span() #Check against original object
                        if all(obj_editable[matching_span[0]:matching_span[1]]):
                            this_obj = re.sub(r'\b{}\b'.format(this_text), r'{}'.format(resolved_text), this_obj)
                            obj_editable[matching_span[0]:matching_span[1]]=[0]*(matching_span[1]-matching_span[0]) # Mark part of text modified


                relation_dict['sub'] = this_sub
                relation_dict['obj'] = this_obj
                articles_extraction.append(relation_dict)
    all_extractions = articles_extraction

    if len(all_extractions)==0:
        print("Nothing to extract. Terminating")
        return jsonify(res_dict)
    all_extractions_w_counts = count_entities(all_extractions)
    ori_extractions_w_counts = count_entities(ori_triples)

    res_dict['ori'] = ori_extractions_w_counts
    res_dict['proc'] = all_extractions_w_counts
    
    return jsonify(res_dict)
        
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)