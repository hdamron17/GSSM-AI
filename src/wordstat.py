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
        self.key = "words"
        self.delimiter = '.!? \t\n\r\f\v'#matches line-ending punctuation and whitespace characters
        self.misc_punctuation = '(){}<>[]$\'\"' #matches puntuation marks which could possibly be placed conveniently beside a delimiter but not signify a new word
        self.punctuating = False #set to true when previous character is some sort of puntuation (so that something like "what." doesn't match two words '"what' and '"')

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
    text = StringIO("I would walk 500 miles. And I would walk 500 more.")
    output = counter.analyze(text)
    print(output)
