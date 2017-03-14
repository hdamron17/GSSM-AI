#! /usr/bin/env python3
''' 
DavedMD chatbot logic engine parses inputs and responds accordingly
* Stores information about responses to questions about symptoms
* Must have response something if user responds that they've never had any disease
* If the bot gets exhausted of responses, it will "refer you to another professional" aka restart the bot (possibly with a different name)
'''

import random
import re #regex

class Magpy:
    def __init__(self, name="Daved"):
        ''' 
        Constructor for Magpy class which initializes object variables
        '''
        self.name = name

        self.diagnoses = set() #holds all diagnoses
        self.new_diagnoses = set() #holds diagnoses until they are told
        self.symptoms = set() #holds all symptoms the user has said they have

        #disease dictionary with name mapped to symptoms
        self.diseases = {
		"common cold" : {"sneezing", "coughing", "fever", "cold"}, 
		"food poisioning": {"vomiting", "diareah", "upset stomach" },
		"the human condition" : {"loneliness", "sadness", "pain"},''' "broken bones" : {"aches", "pain","impared movement"},'''
		"Writer's Block": {"uncreative", "disorientation", "nausia"}, "Caffine Addiction": {"restless", "anxiety"}, 
		"commiting a social faux pas":{"anxious", "paranoid", "guilty"}, "Oedipus Complex":{"jelousy", "guilty", "violent"},
		"narssicism":{"feelings of superiority", "illusions of grandueur", "talkative", "projection"} 
	}

        #symptoms dictionary with implemented name mapped to other forms of the word
        self.synonyms = {
        	"sneezing" : {"sneez", "achoo"}, "cold" : {"chilly", "freezing"}, "vomitting" : {"puking"}, "pain" : {"suffering"}, 
		"anxiety":{"anxious", "nervous", "nervousness"}, "disatisfaction":{"unhappiness"}, "aches":{"discomfort"}
        }

        self.symptoms_asked = set() #set of symptoms asked about so far
        self.time_scale = "few hours" #time used when asking about your symptoms (increases with time_up())

        self.questioned_symptoms = set() #questions asked about previously

    def greeting(self):
        ''' 
        Gets welcome message (brief introduction then asks about symptoms)

        :return: Returns the bot's first message
        '''
        possibilities = (
            """Hi. I'm %s, M.D. """,
            #TODO add more possibilities
        )

        intro = random.choice(possibilities) % self.name #TODO make intro more complex
        symptoms_sent = self.produce_question()
        return intro + symptoms_sent

    def produce_question(self):
        ''' 
        Gathers a random set of symptoms and asks about the user's experience in the past time_scale

        :return: Return the string asking about symptoms
        '''
        possibilities = (
            """Have you experienced %s in the past %s? """,
            """Have you run into %s in the past %s? """
        )

        rand_symptoms = self.random_symptoms()
        self.questioned_symptoms = rand_symptoms #store questions in case user says "yes"
        symptoms_str = self.combined_string(rand_symptoms, conjunction="or")
        sentence = random.choice(possibilities)

        return sentence % (symptoms_str, self.time_scale)

    def exhaused(self):
        ''' 
        Determines if the bot is exhausted of responses (diagnoses or symptoms)
        '''
        return len(self.symptoms) == len(self.synonyms.keys()) \
                or len(self.diagnoses) == len(self.diseases.keys())

    def random_symptoms(self):
        num_symptoms = random.randint(1, 4)
        symptoms_left = set(self.synonyms.keys()) - self.symptoms
        rand_symptoms = random.sample(symptoms_left, num_symptoms) if len(symptoms_left) > num_symptoms else symptoms_left
        return rand_symptoms

    def time_up(self):
        ''' 
        Increases the time scale if possible and returns whether or not it was updated

        :return: return True if the time_scale was edited else False
        '''
        time_scales = ("few hours", "day", "3 days", "week", "fortnight", "month",
            "3 months", "6 months", "year", "Mayan B'ak'tun", "2 years", "5 years", "decade", "lifetime", "century",
             "modern era", "millennium", "Mayan Piktun", "Mayan Kalabtun", "Megaannus", "Mayan K'inchiltun",
            "Mayan Alautun", "epoch", "eon", "forever")
        new_index = time_scales.index(self.time_scale) + 1
        if new_index >= len(time_scales):
            return False
        else:
            self.time_scale = time_scales[new_index]
            return True

    def parse_response(self, user_input):
        ''' 
        Finds keywords (or doesn't) and produces an appropriate response

        :param user_input: user input string
        :return: Returns the bot's response
        '''
        request_restart = not self.time_up() #increase time scale to prevent issues of asking the same thing twice #TODO if we're at forever, then this will fail

        added_string = "" #Used to add any comments at the beginning

        self.extract_symptoms(user_input) #extracts symptoms from the user respons
        self.extract_new_diagnoses() #see if there are any new diagnoses

        new_diagnoses = self.get_new_diagnoses()
        diagnoses_sent = "" #sentence to inform about diagnoses (empty at first)
        if len(new_diagnoses) != 0:
            diagnoses_sent = self.diagnoses_sentence(new_diagnoses)

        if request_restart or self.exhaused():
            self.__init__(name=random.choice(("Steve", "David", "William")))
            added_string = "Due to scheduling conflicts, I will have to refer you to another doctor. Have a nice day.\n\n" \
                + self.greeting()

        symptoms_sent = self.produce_question()

        return diagnoses_sent + added_string + symptoms_sent

    def extract_symptoms(self, user_input):
        ''' 
        Finds all symptoms found in the input string and adds them to the object list

        :param user_input: user input to be parsed
        '''
        #self.symptoms.update(symptom_list) #adds symptoms to the list
        found = False #keeps track of if anything is found
        for word, values in self.synonyms.items():
            #loop through to find each word and its synonyms
            pattern = "(%s|%s)" % (word, "|".join(values))
            regex_search = re.search(pattern, user_input.lower())
            if regex_search:
                self.symptoms.add(word)
                found = True
        if not found and re.search("(yes|all)", user_input.lower()):
            #user responds yes (as in they have all of the above)
            self.symptoms.update(self.questioned_symptoms)

    def combined_string(self, diagnoses, conjunction="and"):
        ''' 
        Creates a string which lists diagnoses in correct way

        :param diagnoses: collection of strings with diagnoses
        :param conjunction: word to use between the final terms (usually 'and'/'or')
        :return: returns string of list (i.e. 'a' or 'a and b' or 'a, b, and c')
        '''
        num = len(diagnoses) #number of diagnoses
        list_diagnoses = list(diagnoses)
        if num == 0: return "" #hopefully this won't happen but handles case
        if num == 1: return list_diagnoses[0]
        if num == 2: return list_diagnoses[0] + " " + conjunction + " " + list_diagnoses[1]
        return ", ".join(list_diagnoses[:-1]) + ", " + conjunction + " " + list_diagnoses[-1]

    def diagnoses_sentence(self, diagnoses):
        ''' 
        Produces humanoid phrase to bear the bad news

        :param diagnoses: collection of strings representing diagnoses
        :return: returns single string telling about diagnoses
        '''
        possibilities = (
         """I'm sorry, you've been diagnosed with %s. """,
         """Unfortunately you have %s. """,
	 """Sounds like you have %s. """ 
        )
        sentence = random.choice(possibilities)
        diagnoses_str = self.combined_string(diagnoses)
        return sentence % diagnoses_str

    def extract_new_diagnoses(self, threshold=0.5):
        ''' 
        Finds all diagnoses not yet assigned based on current symptoms

        :param threshold: Percent [0,1) of symptoms required for diagnosis
        '''
        for possible_diagnosis in set(self.diseases.keys()) - self.diagnoses:
            #iterates over all diseases in the diseases keys but not diagnoses list
            total_symptoms = self.diseases[possible_diagnosis]
            total_symptoms_count = len(total_symptoms)
            current_symptoms_count = len(total_symptoms & self.symptoms)

            if current_symptoms_count / total_symptoms_count > threshold:
                self.add_diagnosis(possible_diagnosis)

    def add_diagnosis(self, diagnosis):
        ''' 
        Adds the diagnosis to the object's diagnosis list (and new_diagnoses)

        :param diagnosis: diagnosis string to be added
        '''
        self.diagnoses.add(diagnosis)
        self.new_diagnoses.add(diagnosis)

    def get_new_diagnoses(self):
        ''' 
        Attempts to produce diagnosis

        Variable self.new_diagnoses is reset to [] afterward
        :return: Returns a set of diagnoses (strings) which have not yet told
        '''
        ret = self.new_diagnoses
        self.new_diagnoses = set()
        return ret

    def get_diagnoses(self):
        ''' 
        Gets all diagnoses generated so far

        :return: Returns a set of all diagnoses (strings)
        '''
        return self.diagnoses

