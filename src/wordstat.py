from io import StringIO
import re

class Extension():
    '''
    Extension base class which analyzes some aspect in an extending class
    '''
    def __init__(self, debug=False):
        '''
        Constructor for basic extension
        :param debug: if true, extensions will keep track of a visualization of counting
        '''
        self.reset()

    def view(self):
        return self.visual

    def reset(self):
        '''
        Outer reset does any extension-independent initialization and calls reset()
        '''
        self.total = 0
        self.init()

    def init(self):
        '''
        Any additional initialization will go here in extended class
        '''
        self.key = "" #Overridden by Extension

        #Overriden by extension
    def count(self, character):
        '''
        Updates the count by the letter passed

        :param character: Single character to be analyzed
        '''
        pass

    def get(self):
        '''
        Gets the current count of whatever is being counted
        '''
        return { self.key : self.total }

class ExtensionCharCount(Extension):
    def init(self):
        self.key = "chars"

    def count(self, character):
        if character != '':
            self.total += 1
            return True
        return False

class ExtensionDelimitedCount(Extension):
    def init(self):
        self.key = ""
        self.delimiter = ''#matches line-ending punctuation and whitespace characters
        self.misc_punctuation = '' #matches puntuation marks which could possibly be placed conveniently beside a delimiter but not signify a new word
        self.punctuating = False #set to true when previous character is some sort of puntuation (so that something like "what." doesn't match two words '"what' and '"')

    def count(self, character):
        if character == '':
            if not self.punctuating:
                # if the previous character was not punctuating, this is the end of a word
                self.total += 1
                return True
        elif self.punctuating:
            #the previous character was a punctuation character
            if character not in self.delimiter + self.misc_punctuation:
                #the character is not another piece of punctuation
                self.punctuating = False
            # else do nothing
        else:
            #previously just traditional ol' characters
            if character in self.delimiter:
                self.total += 1
                self.punctuating = True
                return True
            # else do nothing
        return False

class ExtensionWordCount(ExtensionDelimitedCount):
    def init(self):
        super().init()
        self.key = "words"
        self.delimiter = '.!? \t\n\r\f\v'#matches line-ending punctuation and whitespace characters
        self.misc_punctuation = '(){}<>[]$\'\"' #matches puntuation marks which could possibly be placed conveniently beside a delimiter but not signify a new word
        self.punctuating = False #set to true when previous character is some sort of puntuation (so that something like "what." doesn't match two words '"what' and '"')

class ExtensionSentenceCount(ExtensionDelimitedCount):
    def init(self):
        super().init()
        self.key = "sentences"
        self.delimiter = '.!?'#matches line-ending punctuation and whitespace characters
        self.misc_punctuation = '(){}<>[]$\'\" \t\n\r\f\v' #matches puntuation marks which could possibly be placed conveniently beside a delimiter but not signify a new word
        self.punctuating = False #set to true when previous character is some sort of puntuation (so that something like "what." doesn't match two words '"what' and '"')

class ExtensionSyllableCount(Extension):
    def init(self):
        self.key = "syllables"

    def count(self, character):
        pass #TODO count syllables by character

class WordStat():
    def __init__(self, *extensions):
        '''
        Constructs a WordStat object to start counting stats of a text
        :param extensions: multiple Extension classes (not initialized)
        '''
        self.extensions = [ext() for ext in extensions]
        self.reset()

    def reset(self):
        for ext in self.extensions:
            ext.reset()

    def analyze(self, document):
        '''
        Analyzes a text document and calculates its statistics
        :param document: file object to read through
        :return: returns a dictionary with analysis data
            May have keys "words", "sentences", "syllables", etc.
        '''
        character = document.read(1)
        while character != '':
            for ext in self.extensions:
                ext.count(character.lower()) #count for each couting extension
            character = document.read(1)
        for ext in self.extensions:
            ext.count(character) #count one last time with the empty character to finish words, etc.
        return merge_dicts([single.get() for single in self.extensions])

def merge_dicts(dicts):
    '''
    Merges multiple dicts into a single dict

    :param dicts: collection of dicts to be combined
    :return: returns a single dictionary with all aspects included
    Note: if keys overlap the latter value will be incorporated
    '''
    result = {}
    for single in dicts:
        result.update(single)
    return result

if __name__ == '__main__':
    counter = WordStat(ExtensionCharCount, ExtensionWordCount, ExtensionSentenceCount, ExtensionSyllableCount)
    text = StringIO("I would walk 500 miles. And I\nwould walk 500 more.\nHe said, \"Hi.\" And I said, \"I don't wanna talk to you no more!\"")
    output = counter.analyze(text)
    print(output)
