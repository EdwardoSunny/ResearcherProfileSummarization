import multiprocessing
from openai import OpenAI
import time

client = OpenAI()

class DivConqSummary:
    def __init__(self, domain_prompt, documents_list, max_abstracts_per_group):
        self.max_abstracts_per_group = max_abstracts_per_group
        self.domain_prompt = domain_prompt
        self.documents_list = documents_list
        self.NUM_PROCESSES = len(self.documents_list)

    # maybe implement either reach word count or abstract count first? 
    def reorg_abstract_list(self, documents_list): 
        curr_abstract_count = 0
        result = []
        curr_abstract = ""

        for i in range(0, len(documents_list)):
            curr_abstract_count += 1
            curr_abstract += documents_list[i] + "\n"
            if curr_abstract_count == self.max_abstracts_per_group or i == (len(documents_list) - 1):
                result.append(curr_abstract)
                curr_abstract_count = 0       
                curr_abstract = ""
        return result
        
    def get_GPT_4_response(self, prompt):
        try:
            system_prompt = f"You are a summarization assistant. Your sole task is to provide detailed and accurate summaries of the input text provided by the user that preserves all the ideas in the text. You must not miss or leave out any details in the original text in your summary, all ideas must be present in the summary. The following are one or more articles on various topics. Here are some additional specific context on what the summarizes are: {self.domain_prompt}"

            messages = [
                {"role": "system", "content": system_prompt},
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
        input_prompt = "Write a summary of the following: \n\n" + document
        return self.get_GPT_4_response(input_prompt)

    def run(self):
        # Create a Pool of workers
        it = 0
        st = time.time()
        while (len(self.documents_list) > 1):
            # reorganize reduces document list length
            self.documents_list = self.reorg_abstract_list(self.documents_list)
            self.NUM_PROCESSES = len(self.documents_list)
            with multiprocessing.Pool(processes=self.NUM_PROCESSES) as pool:
                                # Start time
                # Map the function to the document list groups
                self.results = pool.map(self.process_chunk, self.documents_list)
                # End time
                print(self.NUM_PROCESSES)
                print(f"it {it}: document list length is " + str(len(self.documents_list)))
                it += 1
                # update document list to the list of results from each processes' summarization
                self.documents_list = self.results

        et = time.time()
        print("Multiprocessing time:", et-st)
        return self.documents_list 


if __name__ == "__main__":
    divconq = DivConqSummary("You are summarizing some random articles from wikipedia.", ["James Edward Harden Jr. (born August 26, 1989) is an American professional basketball player for the Los Angeles Clippers of the National Basketball Association (NBA). He is widely regarded as one of the greatest scorers and shooting guards in NBA history.", "The Pac-12 Conference Men's Basketball Player of the Year was an award given to the Pac-12 Conference's most outstanding player. The award was first given following the 1975–76 season, when the conference was known as the Pacific-8, and is determined by voting from the Pac-12 media and coaches. There have been two players honored multiple times: David Greenwood of UCLA and Sean Elliott of Arizona. Four freshmen have also won the award: Shareef Abdur-Rahim of California, Kevin Love of UCLA, Deandre Ayton of Arizona and Evan Mobley of USC", "James Madison, America's fourth President (1809-1817), significantly contributed to the Constitution's ratification by co-authoring The Federalist Papers with Alexander Hamilton and John Jay, earning him the title 'Father of the Constitution.' Born in 1751 in Orange County, Virginia, and educated at Princeton, Madison was a prominent figure in the Virginia Assembly, Continental Congress, and the Constitutional Convention. Despite his less charismatic public persona, his wife Dolley was widely admired. Madison played a key role in drafting the Bill of Rights and opposing Hamilton's financial policies, which led to the formation of the Republican Party. As Secretary of State, he confronted France and Britain over their violations of American shipping rights, leading to the Embargo Act of 1807. Elected President in 1808, he faced challenges including the War of 1812, driven by issues such as British impressment of American seamen and trade restrictions. Madison's presidency saw the repeal of the Embargo Act and the eventual declaration of war against Britain on June 1, 1812."], 2)
    results = divconq.run()
    print(results)
