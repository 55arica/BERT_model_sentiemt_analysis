from transformers import BertTokenizer, BertForSequenceClassification
import torch

# ----------------------------------------------------------------------------------------------------


tokenzier = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# ----------------------------------------------------------------------------------------------------

def detect_sentiment(sentence):
  inputs = tokenzier(sentence, return_tensors='pt', padding = True, truncation=True, max_length=512)

  output = model(**inputs)
  logits = output.logits

  predicted_class = torch.argmax(logits, dim=1).item()
  sentiment = 'positive' if predicted_class == 1 else 'negative'
  return sentiment


# ------------------------------------------------------------------------------------------------------  

example_sentences = [
    "I love this product! It's amazing.",
    "This is the worst experience I've ever had.",
    "The movie was okay, not great but not terrible.",
]

for sentence in example_sentences:
  sentiment = detect_sentiment(sentence)
  print(f"Sentence: {sentence}\nSentiment: {sentiment}\n")



#              =================================2 ============================================================================================



from transformers import BertTokenizer, BertForSequenceClassification
import torch


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)


sentence = "I love this product!"

inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)

outputs = model(**inputs)

logits = outputs.logits
print("Logits:", logits)

predicted_class = torch.argmax(logits, dim=1).item()
sentiment = "positive" if predicted_class == 1 else "negative"
print(f"Sentence: {sentence}")
print(f"Sentiment: {sentiment}")
