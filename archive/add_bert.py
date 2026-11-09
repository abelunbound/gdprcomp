#Install dependencies
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import numpy as np


# Set up access to large langauage model as applicable

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


#Error log
# OSError: [Errno 28] No space left on device.
# -- Action taken: cleared up 5GB on drive C: by moving everything in my downloads folder to drive E:


from transformers import AutoTokenizer, AutoModel

model_name_legal = "nlpaueb/legal-bert-base-uncased"
tokenizer_legal = AutoTokenizer.from_pretrained(model_name_legal)
model_legal = AutoModel.from_pretrained(model_name_legal)


#Function to get the text from appropriate source

def get_text_from_policy_doc(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = ' '.join([p.get_text() for p in soup.find_all('p')])
    return text

#
# url = 'https://www.theatlantic.com/privacy-policy/'
# url2 = 'https://www.bbc.co.uk/programmes/articles/3VpmkXtXNhkRpQCq0zsWT7T/privacy-notice'
# url3 = 'https://www.bbc.co.uk/usingthebbc/privacy-policy/'

# Tokenize, Encode and create chunks

# Define maximum sequence length supported by the model
max_seq_length = 512  # Change according to the model's limitation

#Get text from policy document
policy_document_text = get_text_from_policy_doc(url)

# Tokenize and encode the policy document
encoded_policy = tokenizer_legal.encode_plus(policy_document_text, return_tensors='pt', padding=True, truncation=True)

# Split the encoded policy document into chunks
chunk_size_bert = max_seq_length - 2  # Account for [CLS] and [SEP] tokens
input_ids = encoded_policy['input_ids']
num_chunks = -(-len(input_ids[0]) // chunk_size_bert)  # Ceiling division to determine the number of chunks

# Create Embeddings
# Process each chunk separately

# Empty embeddings list
policy_embeddings = []

for i in range(num_chunks):
    start = i * chunk_size_bert
    end = min((i + 1) * chunk_size_bert, len(input_ids[0]))

    # Get a chunk of input_ids
    input_chunk = input_ids[:, start:end]

    # Get BERT embeddings for the chunk
    with torch.no_grad():
        chunk_outputs = model_legal(input_chunk)
    chunk_embeddings = chunk_outputs.last_hidden_state[:, 0, :]

    policy_embeddings.append(chunk_embeddings)

# Concatenate embeddings from all chunks
policy_embeddings = torch.cat(policy_embeddings, dim=0)

# Example conditions obtained from the law (as a dictionary)
law_conditions = {
    "Data Processing Records": "The policy should document how data processing activities are recorded - purposes, data categories, recipients.",
    "Data Subject Rights": "The policy should address and respect data subjects' rights in processing their data.",
    "Consent Mechanisms": "The policy should detail procedures for obtaining, recording, and managing consent for data processing.",
    "Purpose Limitation": "The policy should define lawful purposes for data collection and processing to avoid misuse.",
    "Data Minimisation": "The policy should emphasize collecting only necessary personal data for specified purposes.",
    "Data Security Measures": "The policy should outline security steps -encryption, access controls - for protecting personal data.",
    "Data Breach Response": "The policy should specify procedures for reporting and responding to data breaches.",
    "Data Protection Impact Assessments": "The policy should explain when and how DPIAs are conducted for high-risk activities.",
    "Cross-Border Data Transfers": "The policy should address data protection during international data transfers.",
    "Records of Processing Activities": "The policy should maintain and reference records of processing activities as required.",
    "Data Protection Officer": "The policy should state the appointment of a Data Protection Officer or contact.",
    "Third-Party Data Processors": "The policy should cover use of third-party processors and compliance agreements.",
    "Training and Awareness": "The policy should highlight employee training on data protection importance.",
    "Data Subject Requests": "The policy should outline procedures for handling data subject requests effectively.",
    "Data Retention and Deletion": "The policy should define data retention periods and secure deletion procedures.",
    "Privacy by Design and Default": "The policy should incorporate privacy principles into all data processing.",
    "Updates and Revisions": "The policy should specify how policy updates and revisions are managed and communicated.",
    "Regulatory Reporting": "The policy should address obligations for reporting data processing to authorities.",
    "Incident Documentation": "The policy should mandate documentation of incidents and response actions taken.",
    "Accountability and Governance": "The policy should outline measures demonstrating compliance and governance.",
    "Policy Accessibility": "The policy should ensure easy access to GDPR policy documents for stakeholders.",
    "Legal Basis for Processing": "The policy should provide clear explanations of legal grounds for data processing.",

}

# Calculate similarity scores for each condition
condition_scores = []
results_bert = {}
similarity_scores = []
chunk_scores = []

for criterion, condition in law_conditions.items():
    encoded_condition = tokenizer_legal.encode_plus(condition, return_tensors='pt', padding=True, truncation=True)
    condition_outputs = model_legal(**encoded_condition)
    condition_embeddings = condition_outputs.last_hidden_state[:, 0, :]

    # Convert PyTorch tensors to NumPy arrays
    policy_numpy = policy_embeddings.detach().cpu().numpy()
    condition_numpy = condition_embeddings.detach().cpu().numpy()

    # Calculate cosine similarity between policy document and condition embeddings (using NumPy arrays)
    similarity_scores = cosine_similarity(policy_numpy, condition_numpy).flatten()

    # Assign scores based on similarity for each chunk
    chunk_scores = [10 if score > 0.49 else 0 for score in similarity_scores]
    condition_score = min(sum(chunk_scores), 10)
    condition_scores.append(condition_score)
    similarity_scores = similarity_scores
    chunk_scores = chunk_scores

    results_bert[criterion] = condition_score