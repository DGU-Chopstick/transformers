from transformers import BertTokenizer, BertForTokenClassification
import torch

tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = BertForTokenClassification.from_pretrained(
    "dbmdz/bert-large-cased-finetuned-conll03-english"
)
# print(PreTrainedModel.__doc__)
# help(model)
# print(PreTrainedModel.__doc__)
inputs = tokenizer(
    "HuggingFace is a company based in Paris and New York",
    add_special_tokens=False,
    return_tensors="pt",
)
# BertForTokenClassification.get_config_glen("test")
with torch.no_grad():
    logits = model(**inputs).logits

predicted_token_class_ids = logits.argmax(-1)

# Note that tokens are classified rather then input words which means that
# there might be more predicted token classes than words.
# Multiple token classes might account for the same word
predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
# predicted_tokens_classes
# ['O', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC']
# print(predicted_tokens_classes)
# print(model.config)
print(model)
# print("model")
# modeldata=model
# print(modeldata)
# print("model(**inputs)")
# modelwithargumentdata=model(**inputs)
# print(modelwithargumentdata)

# labels = predicted_token_class_ids
# loss = model(**inputs, labels=labels).loss
# print(round(loss.item(), 2))
