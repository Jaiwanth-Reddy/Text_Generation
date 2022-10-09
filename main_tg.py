from transformers import pipeline
import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg as guten

print(guten.fileids())

length_of_output = 100

prompt = input('Please input the prompt'.title())

nltk.download('gutenberg')
from nltk.corpus import gutenberg as guten
print(guten.fileids())

text = input('Please pick from the above texts')
corpus = list(nltk.corpus.gutenberg.raw(text))

from random import randint

start = randint(0,len(corpus) - len(prompt))

selected_part_of_corpus = corpus[start:start + len(prompt)]

input_to_model = selected_part_of_corpus + prompt

neo_generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')

res = generator(input_to_model, max_length=length_of_output, do_sample=True, temperature=0.9)

gpt_neo = res[0]['generated_text']

print('GPT Neo Text:\n' + gpt_neo)

generator = pipeline('text-generation', model='gpt2')

gpt_2_text = generator(input_to_model, max_length=length_of_output)

gpt_2 = gpt_2_text[0]['generated_text']

print('GPT 2 Text:\n' + gpt_2)

import rouge

def get_rogue_score(original,generated):
  rouge_score = rouge.Rouge()
  score = rouge_score.get_scores(str(original), str(generated), avg=True)       
  return(score['rouge-1']['f'], 2)

from nltk.translate.bleu_score import corpus_bleu

bleu_score_neo = corpus_bleu(prompt, str(gpt_neo))

bleu_score_gpt_2 = corpus_bleu(prompt, str(gpt_2))

rouge_score_neo = get_rogue_score(prompt,gpt_neo)

rouge_score_gpt_2 = get_rogue_score(prompt,gpt_2)

print('Comparing the two models\nUsing BLEU scores:\nGPT Neo ' + str(bleu_score_neo) + '\nGPT_2 ' + str(blue_score_gpt_2) + '\nUsing ROUGE Scores:\nGPT Neo:' + str(rogue_score_neo) + '\nGPT_2 ' + str(rogue_score_gpt_2))