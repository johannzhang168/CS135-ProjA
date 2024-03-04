
import torch
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import MaxAbsScaler


def extract_awesome_features(texts):
    processed_texts = [text[1] for text in texts]  
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(processed_texts, padding=True, truncation=True, return_tensors='pt', max_length=1024)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    features = torch.mean(embeddings, dim=1)
    features_numpy = features.numpy()
    scaler = MaxAbsScaler()
    features_numpy = scaler.fit_transform(features_numpy)
    return features_numpy
