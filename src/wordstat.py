from io import StringIO

class Extension():
    '''
    Extension base class which analyzes some aspect in an extending class
    '''
    def __init__(self):
        '''
        Constructor for basic extension
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
        self.total += 1

class ExtensionWordCount(Extension):
    def init(self):
        self.key = "words"

    def count(self, character):
        pass #TODO count words letter by character

class ExtensionSentenceCount(Extension):
    def init(self):
        self.key = "sentences"

    def count(self, character):
        pass #TODO count sentences by character

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
                ext.count(character)
            character = document.read(1)
        else:
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
    text = StringIO("I would walk 500 miles. And I would walk 500 more.")
    output = counter.analyze(text)
    print(output)
