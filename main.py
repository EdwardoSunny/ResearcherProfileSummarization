import os
import json
import tqdm
from divconq import DivConqSummary
from openai import OpenAI
import time

client = OpenAI()

DATA_PATH = 'human_eval_data/'
OUTPUT_PATH = 'output_data/'

def GPT_only_baseline(abstract_list):
    st = time.time()
    prompt = ""
    for abstract in abstract_list:
        prompt += abstract + "\n"
    try:
        example = "This researcher's research directions can be categorized into the following atomic topics and fields: deep learning, convolutional neural networks (CNNs)... Deep learning is a central focus of this researcher's research, exploring the development and optimization of neural networks that can learn from large amounts of data... Convolutional neural networks (CNNs) are a specific type of neural network that LeCun has significantly advanced. His work in this area involves improving the design and efficiency of CNNs, which are especially effective for image and video recognition tasks. This includes the development of novel CNN architectures and techniques to enhance their capability to process and understand visual information. Self-supervised learning is another key area of LeCun's research..."

        prompt = f"You are an assistant that explains and summaries the research directions of researchers. Your sole task is to provide detailed and accurate description of the researcher's directions based on summaries of the research the researcher has done. Break up interdisciplinary fields into atomic topics/fields for clarity. You must cover ALL areas of the researcher's works and NOT miss any details. Only answer with the description without saying anything extra. Answer in a long detailed paragraph. Ensure any research directions you explain are atomic, there should not be combination directions such as AI in dermatology, Machine Learning and Neuroscience, instead should be separated like AI, dermatology, machine learning, neuroscience etc. Do NOT use any markdown syntax or formatting, just use plain text. Please list all atomic topics and directions first then begin explaining details for each field and topic you listed. Here are an example of a description you should generate: \n\n {example} \n\n Here are some research abstract summaries of the researcher: \n\n" + prompt

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
for filename in tqdm.tqdm(os.listdir(DATA_PATH)):
    if filename.endswith('.json'):
        file_path = os.path.join(DATA_PATH, filename)
        
        with open(file_path, 'r') as file:
            data = json.load(file)
       
        abstract_list = []
        for i in range(0, len(data)):
            title = data[i]["Title"] 
            abstract = data[i]["Abstract"]
            if title is None: 
                title = ""
            if abstract is None: 
                abstract = ""
            full_abstract = title + ": " + abstract 
            abstract_list.append(full_abstract)
        researcher_name = filename.replace(".json", "")
        # GPT only baseline
        # baseline_response = GPT_only_baseline(abstract_list)
        
        # divconq
        divconq = DivConqSummary(abstract_list, 30)
        results = divconq.run()

        curr = {}
        curr["Name"] = researcher_name
        curr["NumAbstracts"] = len(abstract_list)
        # curr["GPT4Baseline"] = baseline_response
        curr["DivConq"] = results[0]
        result_list.append(curr)

with open(OUTPUT_PATH + 'yilu_results.json', 'w', encoding='utf-8') as f:
    json.dump(result_list, f, ensure_ascii=False, indent=4)
