

# General Configurations
gpu_id = 0

#########################################
#### Stage 1 Coreference Model Setup ####
#########################################
coref_bert = 'spanbert_large'
coref_pretrained = 'May22_23-31-16_66000'
article_length_limit = 2400 # Limit to be set depending on GPU availability.

#####################################################
#### Stage 2 Representative Term Selection Setup ####
#####################################################
term_selection_bert = coref_bert
term_selection_method = 'ffnn'
ffnn_params = '/models/span_selection/term_selection_model_params.json'

ffnn_model = '/models/span_selection/term_selection_fine_tuned.pt'

#############################################################################
#### Stage 3 OpenIE, Mention Substitution & Knowledge Graph Construction ####
#############################################################################

openie = 'openie4'
men_sub = 'after'
export_triples = True
