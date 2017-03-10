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
	print(bot.greeting())
	while True:
		user_response = input()
		bot_response = bot.parse_response(user_response)
		print(bot_response)
		#TODO End condition

	 

