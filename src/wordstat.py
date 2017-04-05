#! /usr/bin/env python

'''
Analyzes texts and calculates their reading levels

Put relative paths to files in texts/index.txt
'''

from io import StringIO
import re
import os, sys
from os.path import join as pathjoin, dirname, abspath


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
        self.misc_punctuation = '(){}<>[]$:;,/\\\'\"' #matches puntuation marks which could possibly be placed conveniently beside a delimiter but not signify a new word
        self.punctuating = False #set to true when previous character is some sort of puntuation (so that something like "what." doesn't match two words '"what' and '"')

class ExtensionSentenceCount(ExtensionDelimitedCount):
    def init(self):
        super().init()
        self.key = "sentences"
        self.delimiter = '.!?'#matches line-ending punctuation and whitespace characters
        self.misc_punctuation = '(){}<>[]$:;,/\\\'\" \t\n\r\f\v' #matches puntuation marks which could possibly be placed conveniently beside a delimiter but not signify a new word
        self.punctuating = False #set to true when previous character is some sort of puntuation (so that something like "what." doesn't match two words '"what' and '"')

class ExtensionSyllableCount(Extension):
    def init(self):
        self.key = "syllables"
        self.prev = '' #previous letter
        self.vowels = "aeiouy" #all vowels to watch for
        self.word_count = ExtensionWordCount() #word counter because word ending signals end of syllable
        self.tion_pos = -1 #keeps track of position in "tion" - common exception (starts -1 because 0 is first index)
        self.tion = "tion" #full tion string to index
        self.vowel_y = False #becomes true when y is preceded by a vowel so if next is vowel, it is another syllable
        self.lone_e = False #becomes true when e follows a non-vowel so silent e can be detected
        self.vowel_count = 0 #counts number of non-y vowels in each word to allow specification of silent e
        self.misc_punctuation = '(){}<>[]$:;,/\\\'\"\t\n\r\f\v' #matches puntuation marks which could possibly be placed conveniently beside a delimiter but not signify a new word

        ### Syllable rules
        # A consecutive series of vowels is a syllable unless
        #   o follows any vowel
        #   a follows u or i
        #   there are vowels on both sides of y
        # e on the end of a word is silent
        # "tion" is a single syllable

    def count(self, character):
        # print(character, end='') #TODO remove

        if character in self.misc_punctuation and character != '':
            # ignore any character that is punctuation
            return

        new_word = self.word_count.count(character) #pass the character on

        if new_word:
            if self.vowel_count > 1 and self.lone_e:
                #subtract one because this is the end and the previous e was actually silent
                self.total -= 1
                # print('{*}', end='') #TODO remove
            self.vowel_count = 0
        elif character == 'o' and self.prev in self.vowels and self.prev not in 'o':
            # o follows non-o vowel so syllable
            self.total += 1
            # print('#', end='') #TODO remove
        elif character == "a" and self.prev in "ui":
            # a follows u or i so syllable
            self.total += 1
            # print('|', end='') #TODO remove
        elif self.vowel_y and character in self.vowels:
            # vowel following y is syllable
            self.total += 1
            # print('|', end='') #TODO remove
        elif character in self.vowels and (self.prev not in self.vowels or self.prev == '') and character != 'y':
            # vowel not preceded by a vowel
            self.total += 1
            # print('&', end='') #TODO remove

        # Statements which affect
        if character == 'y' and self.prev in self.vowels and self.prev != '':
            # this is a y following a value so the next may be another syllable even though it's multiple vowels in a row
            self.vowel_y = True
        else:
            # reset vowel_y counter
            self.vowel_y = False

        if character == 'e' and self.prev not in self.vowels:
            # e not after a vowel is could be silent
            self.lone_e = True
        else:
            # reset lone_e counter
            self.lone_e = False

        if character in self.vowels and character != 'y':
            # word has another vowel (to be reset at start of word)
            self.vowel_count += 1

        if character == self.tion[self.tion_pos+1] and (self.prev == self.tion[self.tion_pos] if self.tion_pos > 0 else True):
            # this is the next letter in the tion sequence
            self.tion_pos += 1
            #print('*', end='')
            if self.tion_pos >= 3:
                #the tion sequence is complete and a syllable was over counted
                self.total -= 1
                self.tion_pos = -1
                # print('{^}', end='') #TODO remove
        else:
            self.tion_pos = -1

        self.prev = character if not new_word else '' #update the previous character or cast any new word to empty

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

    def analyze(self, fstream):
        '''
        Analyzes a text document and calculates its statistics
        :param fstream: stream (likely File or StringIO) object to read through
        :return: returns a dictionary with analysis data
            May have keys "words", "sentences", "syllables", etc.
        '''
        start_pos = fstream.tell() #current position to rewind to
        character = fstream.read(1)

        while character != '':
            for ext in self.extensions:
                ext.count(character.lower()) #count for each couting extension
            character = fstream.read(1)

        for ext in self.extensions:
            ext.count(character) #count one last time with the empty character to finish words, etc.

        ret = merge_dicts([single.get() for single in self.extensions])
        self.reset() #reset for next iteration
        fstream.seek(start_pos) #restart reader

        return ret

def flesch_kincaid_level(fstream):
    '''
    Analyzes a text document and calculates Flesch Kincaid reading level
    :param fstream: stream (likely File or StringIO) object to read through
    :return: returns the decimal number grade level of the appropriate reader
    '''
    counter = WordStat(ExtensionCharCount, ExtensionWordCount, ExtensionSentenceCount, ExtensionSyllableCount)
    stats = counter.analyze(fstream)

    words = stats['words']
    sentences = stats['sentences']
    syllables = stats['syllables']

    words_per_sentence = words / sentences
    syllables_per_word = syllables / words

    # Flesch-Kincaid algorithm from http://www.readabilityformulas.com/flesch-grade-level-readability-formula.php
    result = (0.39 * words_per_sentence) + (11.8 * syllables_per_word) - 15.59
    return result

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
    textspath = pathjoin(dirname(abspath(sys.argv[0])), "..", "texts")
    with open(pathjoin(textspath, "index.txt")) as index:
        for line in index.readlines():
            line = line.partition("#")[0].strip()
            if len(line) > 0:
                try:
                    with open(pathjoin(textspath, line)) as text:
                        print("Analysis of " + line)
                        print("  Flesch-Kincaid reading level: %.2f" % flesch_kincaid_level(text))
                except:
                    print("Failed to read " + line)
