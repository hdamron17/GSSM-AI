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
    def __init__(self):
        ''' 
        Constructor for Magpy class which initializes object variables
        '''
        self.diagnoses = set() #holds all diagnoses
        self.new_diagnoses = set() #holds diagnoses until they are told
        self.symptoms = set() #holds all symptoms the user has said they have

        #disease dictionary with name mapped to symptoms
        self.diseases = {
            "common cold" : {"sneezing", "coughing"}
        }

        #symptoms dictionary with implemented name mapped to other forms of the word
        self.synonyms = {
            "sneezing" : {"sneez", "achoo"}
        }

        self.symptoms_asked = set() #set of symptoms asked about so far
        self.time_scale = "few hours" #time used when asking about your symptoms (increases with time_up())

    def greeting(self):
        ''' 
        Gets welcome message (brief introduction then asks about symptoms)

        :return: Returns the bot's first message
        '''
        return "Hi"

    def produce_question(self):
        ''' 
        Gathers a random set of symptoms and asks about the user's experience in the past time_scale

        :return: Return the string asking about symptoms
        '''
        possible_symptoms = set(self.synonyms.keys()) - self.symptoms #symptoms not yet acknowledged
        if len(possible_symptoms) > 0:
            ask_count = len(possible_symptoms) if len(possible_symptoms < 4 else 4 #ask 4 symptoms or as many as possible
        else:
            pass #TODO if there are no answers left, this might be an issue

    def time_up(self):
        ''' 
        Increases the time scale if possible and returns whether or not it was updated

        :return: return True if the time_scale was edited else False
        '''
        time_scales = ("few hours", "day", "3 days", "week", "fortnight", "month",
            "3 months", "6 months", "year", "Mayan B'ak'tun", "2 years", "5 years", "decade", "lifetime", "century",
            "millennium", "Mayan Piktun", "Mayan Kalabtun", "Megaannus", "Mayan K'inchiltun",
            "Mayan Alautun", "epoch", "eon", "forever")
        new_index = time_scales.index(self.time_scale) + 1
        if new_index >= len(time_scales):
            return False
        else:
            self.time_scale = time_scales[new_index]
            return True

    def parse_response(self, input):
        ''' 
        Finds keywords (or doesn't) and produces an appropriate response

        :param input: user input string
        :return: Returns the bot's response
        '''
        self.extract_symptoms(input) #extracts symptoms from the user respons
        request_restart = not self.time_up() #increase time scale to prevent issues of asking the same thing twice #TODO if we're at forever, then this will fail
        self.extract_diagnoses() #see if there are any new diagnoses
        #TODO put all the functions together

    def produce_response(self):
        ''' 
        Produces a bot response based on current bot state

        Uses the get_new_diagnoses() to form a response telling of those diagnoses
        Uses all symptoms - acknowledged symptoms to ask about other symptoms
        Uses time_scale to ask about the user's history in that time
        * i.e. "You have a cold. Have you also had sneezing, coughing, or happiness in the past century?"
        '''
        pass #TODO

    def extract_symptoms(self, symptom_list):
        ''' 
        Finds all symptoms found in the input string and adds them to the object list

        :param input: user input to be parsed
        '''
        #self.symptoms.update(symptom_list) #adds symptoms to the list
        for word, values in self.synonyms.items():
            regex_search = re.search(r"(%s|%s)" % (word, "|".join(values)), input)
            if regex_search:
                self.symptoms.add(word)

    def extract_new_diagnoses(self, threshold=0.5):
        ''' 
        Finds all diagnoses not yet assigned based on current symptoms

        :param threshold: Percent [0,1) of symptoms required for diagnosis
        '''
        for possible_diagnosis in set(self.diseases.keys()) - self.diagnoses:
            #iterates over all diseases in the diseases keys but not diagnoses list
            total_symptoms = self.diseases[possible_diagnoses]
            total_symptoms_count = len(total_symptoms)
            current_symptoms_count = len(total_symptoms - self.symptoms)
            if current_symptoms_count / total_symptoms_count > threshold:
                add_diagnosis(possible_diagnosis)

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
