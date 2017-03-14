#! /usr/bin/env python
''' 
Runner module for DavedMD diagnostic chatbot
* creates Mag.py object
* loops exchanging inputs to the chatbot and displaying chatbot respones
* uses keywords to determine quit conditions
'''
from Mag import Magpy


if __name__=="__main__":
    bot = Magpy()
    done = False #End condition

    print(bot.greeting())
    while not done:
        user_response = input(">>> ")
        if user_response.lower() == "i quit":
            done = True #TODO make more complex end condition
        else:
            bot_response = bot.parse_response(user_response)
            print(bot_response)
    print("Good luck finding a cure without me.")
