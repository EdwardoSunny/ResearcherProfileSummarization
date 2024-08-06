import os
import json
import tqdm
from divconq import DivConqSummary
from openai import OpenAI
import time

client = OpenAI()

DATA_PATH = 'cuimc_data/output'
OUTPUT_PATH = 'output_data/'

def GPT_only_baseline(abstract_list):
    st = time.time()
    prompt = ""
    for abstract in abstract_list:
        prompt += abstract + "\n"
    try:
        prompt = "You are an assistant that explains and summaries the research directions of researchers. Your sole task is to provide detailed and accurate description of the researcher's directions based on some descriptive text on studies the researcher has done. Ensure any research directions you explain are atomic, there should not be combination directions such as AI in dermatology, Machine Learning and Neuroscience, instead should be separated like AI, dermatology, etc. Break up interdisciplinary fields into atomic topics/fields for clarity. Please list all atomic topics and directions first then begin explaining details for each field and topic you listed. You must cover ALL areas of the researcher's works and NOT miss any details. Only answer with the description without saying anything extra. Answer in a long detailed paragraph. Do NOT use any markdown syntax or formatting, just use plain text. Here are some descriptions of the works the researcher has done: \n\n" + prompt

        messages = [
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
person = "Weng, Chunhua"
for filename in tqdm.tqdm(os.listdir(DATA_PATH)):
    if filename.endswith(f'{person}.json'):
        file_path = os.path.join(DATA_PATH, filename)
        
        with open(file_path, 'r') as file:
            data = json.load(file)
       
        abstract_list = []
        for i in range(0, len(data)):
            full_abstract = data[i]["Title"] + ": " + data[i]["Abstract"]
            abstract_list.append(full_abstract)
        researcher_name = filename.replace(".json", "")
        # GPT only baseline
        baseline_response = GPT_only_baseline(abstract_list)
        
        # divconq
        divconq = DivConqSummary(abstract_list, 10)
        results = divconq.run()

        curr = {}
        curr["Name"] = researcher_name
        curr["NumAbstracts"] = len(abstract_list)
        curr["GPT4Baseline"] = baseline_response
        curr["DivConq"] = results[0]
        result_list.append(curr)
        break


with open(OUTPUT_PATH + 'results.json', 'w', encoding='utf-8') as f:
    json.dump(result_list, f, ensure_ascii=False, indent=4)
