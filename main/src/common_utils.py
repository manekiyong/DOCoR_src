import re
import json
from preprocess import get_document

# Contains all the functions that are used across multiple stages

#Used in: Coref, Score model for Best Span
def remove_URL_email(text): #Rationale: URLs & Emails do not provide much information; they waste token space.
    text=re.sub(r'http\S+', '', text)
    text = re.sub(r'\S*@\S*\s?', '', text)
    while "  " in text:
        text = text.replace("  ", " ")
    return text

def get_document_from_string(string, seg_len, bert_tokenizer, spacy_tokenizer, genre='nw'): #News Genre
    doc_key = genre  # See genres in experiment config
    doc_lines = []

    # Build doc_lines
    for token in spacy_tokenizer(string):
        cols = [genre] + ['-'] * 11
        cols[3] = token.text
        doc_lines.append('\t'.join(cols))
        if token.is_sent_end:
            doc_lines.append('\n')
    doc = get_document(doc_key, doc_lines, 'english', seg_len, bert_tokenizer)
    return doc

def load_json(fileName):
    with open(fileName, encoding='utf-8') as f:
        this_json = json.load(f)
    return this_json

def export_json(this_dict, fileName):
    with open(fileName, "w", encoding='utf-8') as outfile:
        json.dump(this_dict, outfile, indent=4)

def clean_article(text_body):
    text_body = text_body.replace("\n\n", " ")
    text_body = text_body.replace("\n", " ")
    text_body = text_body.replace("—", "-")
    text_body = text_body.replace("’","'")
    text_body = text_body.replace("‘","'")
    text_body = text_body.replace("“", "\"")
    text_body = text_body.replace("”", "\"")
    return text_body

def process_span(span):
    final_text = span
    temp_list = re.findall(r"'(.*?)'", final_text) # Replace ' some text ' to 'some text'
    for j in temp_list:
        final_text=final_text.replace(j, j.strip())
    temp_list = re.findall(r"\((.*?)\)", final_text) # Replace ( some text ) to (some text)
    for j in temp_list:
        final_text=final_text.replace(j, j.strip())
    final_text=re.sub(r"\b ' s \b","'s ", final_text) #Apostrophe 1 
    final_text=re.sub(r"\b ' s$\b","'s ", final_text) #Apostrophe 1
    final_text=re.sub(r"\b ’ s \b","'s ", final_text) #Apostrophe 2 (Different type of apostrophe)
    final_text=re.sub(r"\b ’ s$\b","'s ", final_text) #Apostrophe 2
    final_text=final_text.replace(" - ", "-") # Replace some - text to some-text
    final_text=final_text.replace(" , ", ", ") # Replace some , text to some, text
    final_text=re.sub(r"[.,!?\\-]$", "", final_text) #Strip punctuating at ending
    return final_text.strip()

def split_text(text):
    text = text.replace("\"", "\'")
    temp_list = text.split("\', \'")
    temp_list[0]=temp_list[0][2:]
    temp_list[-1]=temp_list[-1][:-2]
    return temp_list

def split_spans(spans):
    temp_list = spans[1:-1].split("), (")
    temp_list[0]=temp_list[0][1:]
    temp_list[-1]=temp_list[-1][:-1]
    final_list=[]
    for i in temp_list:
        temp_list=[]
        temp_span = i.split(", ")
        start = int(temp_span[0])
        end = int(temp_span[1])
        temp_list.append(start)
        temp_list.append(end)
        final_list.append(temp_list)
    return final_list

def get_rel_rest(openie, text):
    extracted = openie.extract(text)
    return extracted

def clean_sent(sent):
    sent = sent.replace("—", "-")
    sent = sent.replace("’","'")
    sent = sent.replace("\“", "\"")
    sent = sent.replace("\”", "\"")
    return sent

def untokenize(text):
    text = re.sub(r" ([^a-zA-Z\d\s.,!?]) ", r" \1", text) #Handle mostly apostrophe cases
    text = re.sub(r" ([.,?!]) ", r"\1 ", text)
    text = re.sub(r" n't\b", r"n't", text)
    text = re.sub(r" 've\b", r"'ve", text)
    text = re.sub(r" 'll\b", r"'ll", text)
    text = re.sub(r" 'd\b", r"'d", text)
    text = re.sub(r" 're\b", r"'re", text)
    text = re.sub(r" 's\b", r"'s", text)
    text = re.sub(r" 'm\b", r"'m", text)
    return text
    
def count_entities(articles_extraction):
    entity_count = {}
    for triple in articles_extraction:
        if not triple['sub'] in entity_count:
            entity_count[triple['sub']]=1
        else:
            if entity_count[triple['sub']]==5:
                entity_count[triple['sub']]=5
            else:
                entity_count[triple['sub']]+=1
        if not triple['obj'] in entity_count:
            entity_count[triple['obj']]=1
        else:
            if entity_count[triple['obj']]==5:
                entity_count[triple['obj']]=5
            else:
                entity_count[triple['obj']]+=1

    articles_extraction_w_count = []
    for triple in articles_extraction:
        triple['subj_count']=entity_count[triple['sub']]
        triple['obj_count']=entity_count[triple['obj']]
        articles_extraction_w_count.append(triple)
    return articles_extraction_w_count

def get_unique_node_rls(articles_extraction):
    entity_count = {}
    no_of_rls = len(articles_extraction)
    for triple in articles_extraction:
        if not triple['sub'] in entity_count:
            entity_count[triple['sub']]=1
        else:
            entity_count[triple['sub']]+=1
        if not triple['obj'] in entity_count:
            entity_count[triple['obj']]=1
        else:
            entity_count[triple['obj']]+=1
    # print("No. of Unique Node: ", len(entity_count))
    # print("No. of Rls", no_of_rls)
    return len(entity_count), no_of_rls