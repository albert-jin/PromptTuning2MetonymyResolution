from openprompt.plms import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm("t5", "t5-base")
line = 'blé :traitement gonflement.'.split(' ')
words = []
sents = []
for token in line:
    words.extend(tokenizer.tokenize(token))
    sents.append((tokenizer.tokenize(token)))
print(sents)
print(words)
print(tokenizer.convert_tokens_to_ids(words))
