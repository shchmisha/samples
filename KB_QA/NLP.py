import pandas as pd
import openai
import os
import fileinput
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, GPT2Config
import torch
import numpy as np
openai.api_key = "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"

'''
Main call: completion
response = openai.Completion.create(
  model="text-davinci-003",
  temperature=0,
  prompt=SOME_PROMPT,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
'''

NEWLINES_RE = re.compile(r"\n{2,}")  # two or more "\n" characters

def split_paragraphs(input_text=""):
    no_newlines = input_text.strip("\n")  # remove leading and trailing "\n"
    split_text = NEWLINES_RE.split(no_newlines)  # regex splitting

    paragraphs = [p + "\n" for p in split_text if p.strip()]
    # p + "\n" ensures that all lines in the paragraph end with a newline
    # p.strip() == True if paragraph has other characters than whitespace

    return paragraphs



class API:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key

    def request(self, prompt):
        result = ""

        while True:
            response = openai.Completion.create(
                model="text-davinci-003",
                temperature=0,
                prompt=prompt,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            prompt += response["choices"][0]["text"]
            result += response["choices"][0]["text"]

            if response["choices"][0]["finish_reason"] == "stop":
                break
        return result

    def extract_doc_context(self, text):
        query = text + "\nlist the primary topics of the text above:"
        response = self.request(query)
        return response

    def extract_paragraph_context(self, context, text):
        query = "given text:\n" + text + "\nlist the primary topics from the context that appear in the text:"
        response = self.request(query)
        return response

    def compare_contexts(self, context1, context2):
        query = "given list 1:\n" + context1 + "\nqnd given list 2:\n" + context2 + "\nare the topics in the two lists similar: yes/no"
        response = self.request(query)
        return response

    def extract_request_context(self, context, text):
        query = "given the context:\n"+context +" and the question:\n" + text+ "\nlist the primary topics of the question above:"
        response = self.request(query)
        return response

    def extract_request_operation(self, context, request):
        # query = "given the context:\n"+context +" and given request:\n" + request+"\nwhat is the operation in the request summarize/enhance/paraphrase/explain/answer:"
        query = "given request:\n" + request+"\nwhat is the request asking for? summarize, enhance, paraphrase, explain, answer: "
        response = self.request(query)
        return response

    def answer_question(self, context, text):
        query = "given the context:\n"+context + "\n" + text
        response = self.request(query)
        return response

    def reformat_text(self, context, text, format):
        query = "given the context:\n"+context + "\nand text" + text + "\n"+format+"the text above:"
        response = self.request(query)
        return response

    def embed_text(self, text):
        return openai.Embedding.create(
            model='text-embedding-ada-002',
            input=text
        )

    def augment_doc(self, document, context, request, request_operation):
        # augmetnign doc: choose topic in document and perform (summarize/enhance/paraphrase/explain)
        # aplit the text into some 
        # paraphrase each one given the context prpvided
        paragraphs = split_paragraphs(document)
        request_context = self.extract_request_context(context, request)
        # request_format = self.extract_request_operation(context, request)
        for i in range(len(paragraphs)):
            # check which topics from the document occur in the paragraph
            paragraph_context = self.extract_paragraph_context(context, paragraphs[i])
            if self.compare_contexts(request_context, paragraph_context):
                paragraphs[i] = self.reformat_text(context, paragraphs[i], request_operation)
        print(paragraphs)
        return ''.join(paragraphs)

# when enhanicing/explainign a topic, loop through all parts of the document, and if its topic mathces the topic requested, then augment it


# doc_context + topic -> format 
# could use the openai
# could use embeddings and RNN?


# doc_context -> questions
# could use openai as well. create a fine tune fore each user? expensive but possible
# 
def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)

class ML:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.loaded_models = {}

    def create_model(self, user_id):
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        torch.save(model.state_dict(), './DB/'+user_id+"/"+'model')

    def load_model(self, user_id):
        model = GPT2LMHeadModel(GPT2Config())
        model.load_state_dict(torch.load('./DB/'+user_id+"/"+'model'))

        optimizer = AdamW(model.parameters(), lr=0.0001)
        return model, optimizer

    def start_session(self, user_id):
        model, optimizer = self.load_model(user_id)
        self.loaded_models[user_id] = (model, optimizer)
    
    def close_session(self, user_id):
        model, _ = self.loaded_models[user_id]
        torch.save(model.state_dict(), './DB/'+user_id+'model')
        self.loaded_models.pop(user_id)

    def train_model(self, user_id, prompt, response):
        model, optimizer = self.loaded_models[user_id]

        inputs = prompt + response
        inputs = torch.tensor(self.tokenizer.encode(inputs)).unsqueeze(0)

        outputs = model(inputs, labels=inputs)
        loss, logits = outputs[:2]                        
                    
        optimizer.zero_grad()
        loss.backward()
        optimizer.zero_grad()

    def generate(self, user_id, prompt):
        model, _ = self.loaded_models[user_id]
        cur_ids = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)

        for i in range(100):
            outputs = model(cur_ids, labels=cur_ids)
            loss, logits = outputs[:2]
            softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(from only one in this case) batch and the last predicted embedding
            if i < 3:
                n = 20
            else:
                n = 3
            next_token_id = choose_from_top(softmax_logits.detach().numpy(), n=n) #Randomly(from the topN probability distribution) select the next word
            cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long() * next_token_id], dim = 1) # Add the last word to the running sequence

            if next_token_id in self.tokenizer.encode('<|endoftext|>'):
                joke_finished = True
                break


        output_list = list(cur_ids.squeeze().detach().numpy())
        output_text =self.tokenizer.decode(output_list)
        return output_text
