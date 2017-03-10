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
            "common cold" : ("sneezing", "coughing")
        }

        #symptoms dictionary with implemented name mapped to other forms of the word
        self.synonyms = {
            "sneezing" : ("sneez", "achoo")
        }

        self.symptoms_asked = set() #set of symptoms asked about so far
        self.time_scale = "few hours" #time used when asking about your symptoms (increases with time_up())

    def greeting(self):
        ''' 
        Gets welcome message (brief introduction then asks about symptoms)

        :return: Returns the bot's first message
        '''
        pass #TODO

    def time_up(self):
        time_scales = ("few hours", "day", "3 days", "week", "fortnight", "month",
            "3 months", "6 months", "year", "Mayan B'ak'tun", "2 years", "5 years", "decade", "lifetime", "century",
            "millennium", "Mayan Piktun", "Mayan Kalabtun", "Megaannus", "Mayan K'inchiltun",
            "Mayan Alautun", "epoch", "eon", "forever")
        self.time_scale = time_scales[time_scales.index(self.time_scale) + 1]

    def parse_response(self, input):
        ''' 
        Finds keywords (or doesn't) and produces an appropriate response

        :param input: user input string
        :return: Returns the bot's response
        '''
        pass #TODO

    def answer_symptoms(self, input):
        ''' 
        Used for when user responds to questions about their symptoms

        :param input: user input string
        :return: Returns response (either diagnosis or question about more symptoms
        '''
        pass #TODO

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
