import json
import requests
# from stanfordcorenlp import StanfordCoreNLP
# from pyclausie import ClausIE

class OpenIE_Reverb:
    '''
    To run Reverb, you need run up the RESTful service
    cd /home/kc/NLP/openie_extraction/tools/reverb
    java -jar reverb_restful.jar --server.port=8010
    '''
    def __init__(self, server_port=8010):
        server_url = 'http://localhost:' + str(server_port)
        if server_url[-1] == '/':
            server_url = server_url[:-1]
        self.server_url = server_url
        self.extract_context = '/reverb'

    def extract(self, text):
        try:
            requests.get(self.server_url)
        except requests.exceptions.ConnectionError:
            raise Exception('Check whether you have started the ReVerb server')

        try:
            r = requests.post(self.server_url + self.extract_context, data=text, verify=False)
            extractions = json.loads(r.text)
            reverb_triple = []
            print(extractions)
            for triple in extractions['extractions']:
                reverb_triple.append({"sub": triple['sub'], "rel": triple['rel'], "obj": triple['obj'],
                                      "conf": float(triple['conf'])})
        except:
            reverb_triple = []
            print('Reverb failed: ', text)

        return reverb_triple


class OpenIE_4:
    '''
    To run Openie4, you need run up the RESTful service
    cd /home/kc/NLP/openie_extraction/tools/openie4
    java -jar openie4_restful.jar --binary --httpPort 8020
    '''
    def __init__(self, server_port=8020):
        server_url = 'http://openie4:'+str(server_port)
        if server_url[-1] == '/':
            server_url = server_url[:-1]
        self.server_url = server_url
        self.extract_context = '/getExtraction'

    def extract(self, text, properties=None):
        assert isinstance(text, str)
        if properties is None:
            properties = {}
        else:
            assert isinstance(properties, dict)

        try:
            requests.get(self.server_url)
        except requests.exceptions.ConnectionError:
            raise Exception('Check whether you have started the OpenIE4 server')

        data = text.encode('utf-8')

        try:
            r = requests.post(
                self.server_url + self.extract_context, params={
                    'properties': str(properties)
                }, data=data, headers={'Connection': 'close'})
            extractions = json.loads(r.text)
        except:
            extractions = []
            print('Openie 4 failed: ', text)

        openie4_triple = []

        for extraction in extractions:
            conf = float(extraction['confidence'])
            sub = extraction['extraction']['arg1']['text']
            rel = extraction['extraction']['rel']['text']
            obj = extraction['extraction']['arg2s']
            if len(obj) > 0:
                obj = obj[0]['text']
            else:
                obj = ''

            openie4_triple.append({"sub": sub, "rel": rel, "obj": obj, "conf": conf})

        return openie4_triple

class OpenIE_5:
    '''
    To run openIE 5, you need at least 12G memory
    see https://github.com/vaibhavad/python-wrapper-OpenIE5 for more information
    cd /home/kc/NLP/openie_extraction/tools/openie5
    java -Xmx12g -XX:+UseConcMarkSweepGC -jar openie-assembly-5.0-SNAPSHOT.jar --httpPort 8030
    '''
    def __init__(self, server_port=8030):
        server_url = 'http://localhost:'+str(server_port)
        if server_url[-1] == '/':
            server_url = server_url[:-1]
        self.server_url = server_url
        self.extract_context = '/getExtraction'

    def extract(self, text, properties=None):
        assert isinstance(text, str)
        if properties is None:
            properties = {}
        else:
            assert isinstance(properties, dict)

        try:
            requests.get(self.server_url)
        except requests.exceptions.ConnectionError:
            raise Exception('Check whether you have started the OpenIE5 server')

        try:
            data = text.encode('utf-8')
            r = requests.post(
                self.server_url + self.extract_context, params={
                    'properties': str(properties)
                }, data=data, headers={'Connection': 'close'})
            extractions = json.loads(r.text)
        except:
            extractions = []
            print('Openie 5 failed: ', text)

        openie5_triple = []

        for extraction in extractions:
            # print(extraction)
            confidence = extraction['confidence']
            extraction = extraction['extraction']
            sub_string = extraction['arg1']['text']
            rel_string = extraction['rel']['text']
            obj_list = extraction['arg2s']
            object_string = []
            for object in obj_list:
                object_string.append(object['text'])
            if len(object_string) == 0:
                continue
            else:
                openie5_triple.append({"sub": sub_string, "rel": rel_string, "obj": object_string[0],
                                                "conf": confidence})

        return openie5_triple


# class OpenIE_Stanford:
#     '''
#     To run stanford core nlp service, must navigate to its directory and run the java pipeline service
#     cd /home/kc/NLP/openie_extraction/tools/stanford
#     java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
#     '''

#     def __init__(self, server_port=9000):
#         self.server = StanfordCoreNLP('http://localhost', port=server_port)
#         self.pros = {'annotators': 'openie', 'pinelineLanguage': 'en'}

#     def extract(self, text):
#         # call stanford NLP to get openIE triples, NERs, etc..
#         try:
#             sentences = json.loads(self.server.annotate(text, properties=self.pros))
#             stanford_triple = []
#             for sent in sentences['sentences']:
#                 if 'openie' in sent.keys():
#                     for extraction in sent['openie']:
#                         # print(extraction)
#                         sub_string = extraction['subject']
#                         rel_string = extraction['relation']
#                         obj_string = extraction['object']
#                         stanford_triple.append({"sub": sub_string, "rel": rel_string, "obj": obj_string})

#         except:
#             print('Openie Stanford failed: ', text)
#             stanford_triple = []

#         return stanford_triple


# class OpenIE_ClausIE():
#     def __init__(self):
#         self.cl = ClausIE.get_instance()

#     def extract(self, text):
#         clausie_triple = []
#         try:
#             triples = self.cl.extract_triples([text])
#             for triple in triples:
#                 sub_string = getattr(triple, 'subject')
#                 rel_string = getattr(triple, 'predicate')
#                 obj_string = getattr(triple, 'object')
#                 clausie_triple.append({"sub": sub_string, "rel": rel_string, "obj": obj_string})
#         except ValueError:
#             print('Clausie failed: ', text)
#             pass

#         return clausie_triple