import multiprocessing
from openai import OpenAI
import time
from lda import GroupLDA

client = OpenAI()

class DivConqSummary:
    def __init__(self, documents_list, num_topics):
        self.documents_list = documents_list
        self.num_topics = num_topics
        self.NUM_PROCESSES = len(self.documents_list)

    # maybe implement either reach word count or abstract count first? 
    def reorg_abstract_list(self, documents_list): 
        curr_abstract_count = 0
        result = []
        curr_abstract = ""
        grouplda = GroupLDA(self.documents_list)
        grouplda.train(self.num_topics)
        grouped_docs = grouplda.group_docs()
        count = 0
        for l in grouped_docs:
            curr = ""
            # print("CURR COUNT: ", len(l))
            for doc in l:
                curr += doc + "\n"
                count+=1
            result.append(curr)
        # print("COUNT:", count)
        return result
        
    def get_GPT_4_response(self, prompt):
        if (len(prompt) == 0):
           return "" 

        if ("You are an assistant that explains and summaries the research directions of researchers. Your sole task is to provide detailed and accurate description of the researcher" in prompt):
            print(prompt)
        try:
            messages = [
                {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.4
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

    def process_chunk(self, document):
        # Get the GPT response for each entry in the document_list

        input_prompt = "You are a summarization assistant that summaries the given research abstracts very very detailedly without losing information on any of the abstracts provided. Your sole task is to provide very detailed and accurate summaries of the input abstracts and preserves all the ideas in each abstract but condensing the information. You must not miss or leave out any details in the every original abstracts in your summary, all abstract topics must be present in the summary. The summary must be a concise long paragraph with all abstract details. Only answer with the summary, do not say anything extra. Here are the abstracts:\n\n" + document
        return self.get_GPT_4_response(input_prompt)

    def run(self):
        # Create a Pool of workers
        it = 0
        st = time.time()
        # reorganize reduces document list length
        self.documents_list = self.reorg_abstract_list(self.documents_list)
        self.NUM_PROCESSES = len(self.documents_list)
        with multiprocessing.Pool(processes=self.NUM_PROCESSES) as pool:
            # Map the function to the document list groups
            self.results = pool.map(self.process_chunk, self.documents_list)
            # End time
            # print(self.NUM_PROCESSES)
            # print(f"it {it}: document list length is " + str(len(self.documents_list)))
            it += 1
            # update document list to the list of results from each processes' summarization
            self.documents_list = self.results
        combined_summaries = ""
        for summary in self.documents_list:
            combined_summaries += summary + "\n"
        self.documents_list = [self.get_GPT_4_response("You are an assistant that explains and summaries the research directions of researchers. Your sole task is to provide detailed and accurate description of the researcher's directions based on some descriptive text on studies the researcher has done. Ensure any research directions you explain are atomic, there should not be combination directions such as AI in dermatology, Machine Learning and Neuroscience, instead should be separated like AI, dermatology, etc. Break up interdisciplinary fields into atomic topics/fields for clarity. Please list all atomic topics and directions first then begin explaining details for each field and topic you listed. You must cover ALL areas of the researcher's works and NOT miss any details. Only answer with the description without saying anything extra. Answer in a long detailed paragraph. Do NOT use any markdown syntax or formatting, just use plain text. Here are some descriptions of the works the researcher has done: \n\n" + combined_summaries)]
        
        et = time.time()
        # print("Multiprocessing time:", et-st)
        return self.documents_list 


if __name__ == "__main__":
    # debugging code
    divconq = DivConqSummary(["James Edward Harden Jr. (born August 26, 1989) is an American professional basketball player for the Los Angeles Clippers of the National Basketball Association (NBA). He is widely regarded as one of the greatest scorers and shooting guards in NBA history.", "The Pac-12 Conference Men's Basketball Player of the Year was an award given to the Pac-12 Conference's most outstanding player. The award was first given following the 1975–76 season, when the conference was known as the Pacific-8, and is determined by voting from the Pac-12 media and coaches. There have been two players honored multiple times: David Greenwood of UCLA and Sean Elliott of Arizona. Four freshmen have also won the award: Shareef Abdur-Rahim of California, Kevin Love of UCLA, Deandre Ayton of Arizona and Evan Mobley of USC", "James Madison, America's fourth President (1809-1817), significantly contributed to the Constitution's ratification by co-authoring The Federalist Papers with Alexander Hamilton and John Jay, earning him the title 'Father of the Constitution.' Born in 1751 in Orange County, Virginia, and educated at Princeton, Madison was a prominent figure in the Virginia Assembly, Continental Congress, and the Constitutional Convention. Despite his less charismatic public persona, his wife Dolley was widely admired. Madison played a key role in drafting the Bill of Rights and opposing Hamilton's financial policies, which led to the formation of the Republican Party. As Secretary of State, he confronted France and Britain over their violations of American shipping rights, leading to the Embargo Act of 1807. Elected President in 1808, he faced challenges including the War of 1812, driven by issues such as British impressment of American seamen and trade restrictions. Madison's presidency saw the repeal of the Embargo Act and the eventual declaration of war against Britain on June 1, 1812."], 7)
    results = divconq.run()
    print(results)
