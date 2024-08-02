import os
import json
import tqdm
from divconq import DivConqSummary
from openai import OpenAI
import time

client = OpenAI()

DATA_PATH = 'cuimc_data/output'
OUTPUT_PATH = 'output_data/'

def GPT_only_baseline(domain_prompt, abstract_list):
    st = time.time()
    prompt = ""
    for abstract in abstract_list:
        prompt += abstract + "\n"
    try:
        system_prompt = f"You are a summarization assistant. Your sole task is to provide detailed and accurate summaries of the input text provided by the user that preserves all the ideas in the text. You must not miss or leave out any details in the original text in your summary, all ideas must be present in the summary. The following are one or more articles on various topics. Here are some additional specific context on what the summarizes are: {domain_prompt}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4
        )
        et = time.time()
        print("Baseline time:", et-st)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

result_list = []

for filename in tqdm.tqdm(os.listdir(DATA_PATH)):
    if filename.endswith('.json'):
        file_path = os.path.join(DATA_PATH, filename)
        
        with open(file_path, 'r') as file:
            data = json.load(file)
       
        abstract_list = []
        for i in range(0, len(data)):
            full_abstract = data[i]["Title"] + ": " + data[i]["Abstract"]
            abstract_list.append(full_abstract)
        researcher_name = filename.replace(".json", "")
        domain_prompt = f"Given the following abstracts written by {researcher_name}, summarize their research focus/directions based on these papers."

        # GPT only baseline
        baseline_response = GPT_only_baseline(domain_prompt, abstract_list)
        
        # divconq
        divconq = DivConqSummary(domain_prompt, abstract_list, 10)
        results = divconq.run()

        curr = {}
        curr["Name"] = researcher_name
        curr["NumAbstracts"] = len(abstract_list)
        curr["GPT4Baseline"] = baseline_response
        curr["DivConq"] = results[0]
        result_list.append(curr)

with open(OUTPUT_PATH + 'results.json', 'w', encoding='utf-8') as f:
    json.dump(result_list, f, ensure_ascii=False, indent=4)
