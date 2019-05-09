Hunter Damron’s Implementation of the Flesch-Kincaid Algorithm by Flesch and Kincaid

The Flesch-Kincaid algorithm is used for grade-level readings to assess appropriate reading level. It is based on the average number of words per sentence and the average number of syllables per word and calculates readability as a number roughly in the range 0-12. Outside of that range, it begins to lose validity. This project also attempted to use SMOG and Power-Sumner-Kearl algorithms for verification of the Flesch-Kincaid result, but was never successful in those attempts. The calculation of reading statistics (i.e. word count, syllable count, sentence count, etc.) was completed not with regex, in the interest of speed. Instead, counters were created as extensions to a single counter such that the text only had to be iterated once. Syllable counting was the most interesting – a base counter recorded syllables according to traditional rules, but a list of exceptions was applied when necessary to correct the base count. Words and sentences just used a straight delimiter for differentiation.

Algorithm pros/cons: the Flesch-Kincaid algorithm was simple to implement but because of that simplicity, it left out some important aspects of the count. First, it counts only the number of syllables without caring about where they are, so proper nouns (which often have many syllables) are not included. The Flesch-Kincaid algorithm, along with most others, also neglect to factor in more complex sources of difficulty like poorly worded sentences or the intricacy of sentence structure. These factors would be difficult to detect, but they would be more reliable reasons for difficulty of a reading.