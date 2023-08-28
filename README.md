## System Requirements
- Approx 5GB of GPU VRAM might be needed to load both Coreference & Span Selection model. 

## Preparing Files
- Download Coreference Model weights from [**HERE**](https://drive.google.com/file/d/1pW4IVlQFaSkTYavHbY9-7xl6HoSoStIe/view?usp=sharing) and put in `/main/models/`
    - Note: SpanBERT Large and the coreference resolution weights (`model_May22_23-31-16_66000.bin`) contained within this `.zip` folder are from [this repository](https://github.com/lxucs/coref-hoi), just zipping it here for convenience sake.
    - span_selection model are the one developed as part of this work. 
- Download RESTful OpenIE4 Java file from [**HERE**](https://drive.google.com/file/d/1r5OB6ygWNw0ByXKreoUUkxhxhReXeMxL/view?usp=sharing) and put in `/openie/`


## Start up
```
cd build
docker-compose up
```

## Inference
This can be done on your host machine
```
import requests
import json

text = """Your text here"""

url = "http://localhost:5000/build"
data = {'text': text}
r = requests.post(url,
                  data=json.dumps(data), 
                  headers={'Content-type': 'application/json'})

response = json.loads(r.text)
preprocessed_triples = response['ori'] # Raw semantic triple extracted from OpenIE4
processed_triples = response['proc'] # Processed semantic triple after term selection & alignment

# Compare the original triple with the processed triple
for ori, proc in zip(preprocessed_triples,processed_triples):
    print(f"Original : {ori['sub']} --- {ori['rel']} --> {ori['obj']}")
    print(f"Processed: {proc['sub']} --- {proc['rel']} --> {proc['obj']}")
    print("")
```