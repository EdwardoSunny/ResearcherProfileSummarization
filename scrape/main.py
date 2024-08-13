from bs4 import BeautifulSoup
import requests
import tqdm
from openai import OpenAI
import json

client = OpenAI()


def get_GPT4_response(prompt, client):
    message = [
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=message,
    )

    return response.choices[0].message.content


url = "https://www.dbmi.columbia.edu/faculty/"
req = requests.get(url)

soup = BeautifulSoup(req.content, "html.parser")

results = {}
for link in tqdm.tqdm(soup.find_all("a")):
    link = link.get("href")
    if "dbmi.columbia.edu/profile/" in link:
        response = requests.get(link)
        if response.status_code == 200:
            # name = link.replace("https://www.dbmi.columbia.edu/profile/", "")
            # name = name.replace("/", "")
            # name = name.replace("-", " ")
            content = response.text
            soup = BeautifulSoup(content, "html.parser")
            div = soup.find("div", id="content")

            prompt = f"{div} \nPlease extract the name of this researcher. Only give me the name text, don't say anything else."
            name = get_GPT4_response(prompt, client)
            name = name.lower()

            prompt = f"{div} \nplease extract the description text from this. Only give me the description text, don't say anything else."
            description_only = get_GPT4_response(prompt, client)

            results[name] = description_only
            with open("results.json", "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
