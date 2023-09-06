#Method 2: Part of Speech tagging using bigrams

import collections, math

class pos_tagger:

	#Training
	def __init__(self, trainData):
		#Initialize data structures
		tagBigramCounts = dict()
		tagBigramCounts["<s>"] = collections.Counter()
		tagWordCounts = dict()
		vocab = set()
		tagVocab = set()
		tagBigramProbs = dict()
		tagWordProbs = dict()

		#Work through the training data and count tag bigrams and word-tag pairs
		numWords = 0
		for line in trainData:
			prevTag = "<s>"
			words = line.split()[1:]
			age = words[0]
			words = words[1:]

			for wordTagPair in words:
				numWords += 1
				wordTagPair = wordTagPair.split('/')
		
				word = wordTagPair[0]
				tag = wordTagPair[1]

				#Initialize the counter if the tag hasn't been seen before
				if tag not in tagWordCounts:
					tagWordCounts[tag] = collections.Counter()
				if tag not in tagBigramCounts: #Prep for the next tag we'll see
					tagBigramCounts[tag] = collections.Counter()
					tagVocab.add(tag)
				if word not in vocab:
					vocab.add(word)
		
				#Add to tagBigramCounts for prevTag-tag bigram
				tagBigramCounts[prevTag][tag] += 1

				#Add to tagWordCounts
				tagWordCounts[tag][word] += 1

				#Set current tag as new prevTag
				prevTag = tag
	
			#Account for end of line symbol </s>
			tagBigramCounts[prevTag]["</s>"] += 1

		#Get vocabulary size
		vocabSize = len(vocab)
		tagVocabSize = len(tagVocab)
		tagVocabWithEnd = set(tagVocab)
		tagVocabWithEnd.add("</s>") #We only need to check the end tag sometimes

		#Set smoothing parameter
		eta = 0.005

		#Get probabilities of each tag bigram
		for sourceTag in tagBigramCounts:
			thisTagTotal = sum(tagBigramCounts[sourceTag].values())
			tagBigramProbs[sourceTag] = dict()
			#for destinationTag in tagBigramCounts[sourceTag]:
			for destinationTag in tagVocabWithEnd:
				if destinationTag in tagBigramCounts[sourceTag]:
					tagBigramProbs[sourceTag][destinationTag] = float(tagBigramCounts[sourceTag][destinationTag] + eta)/(thisTagTotal + tagVocabSize)
				else:
					tagBigramProbs[sourceTag][destinationTag] = eta/(thisTagTotal + tagVocabSize)

		#Get p(w|t) probabilities
		for tag in tagWordCounts:
			thisTagTotal = sum(tagWordCounts[tag].values())
			tagWordProbs[tag] = dict()
			for word in tagWordCounts[tag]:
				tagWordProbs[tag][word] = float(tagWordCounts[tag][word] + eta)/(thisTagTotal + vocabSize)


		#Save relevant data structures
		self.tagBigramProbs = tagBigramProbs
		self.tagWordProbs = tagWordProbs
		self.tagVocab = tagVocab
		self.vocabSize = vocabSize

	#Testing
	def decode(self, words):
		numWords = len(words)

		#Get relevant data structures
		tagBigramProbs = self.tagBigramProbs
		tagWordProbs = self.tagWordProbs
		tagVocab = self.tagVocab
		vocabSize = self.vocabSize

		#Create array of tag names; these will become state names
		tags = []
		for tag in tagVocab:
			tags.append(tag)

		#Create list of states- this is a two-dimensional array that represents the states of the HMM. Traverse each row to get topological order
		states = []
		states.append(["<s>"])
		for i in range(1, numWords+1):
			states.append(tags)
		states.append(["</s>"])

		#Initialize viterbi data structures. viterbi and pointers are arrays of dictionaries, where each row represents a word in the word sequence 
		#and the dictionary keys are the different states (possible tags). For viterbi, the values are the log weights; for pointers, the values 
		#are the previous states
		viterbi = [] 
		pointers = []
		viterbi.append({"<s>": 0})
		pointers.append({"<s>": 0})
		for i in range(1, numWords+1):
			viterbi.append(dict())
			pointers.append(dict())
			for tag in tags:
				viterbi[i][tag] = 0
				pointers[i][tag] = 0
		viterbi.append({"</s>": 0})
		pointers.append({"</s>": 0})
	
		#Work through the Viterbi algorithm
		for i in range(1, numWords+1): #Work through each word in the word sequence
			wordTagPair = words[i-1].split('/')
			word = wordTagPair[0]
			actualTag = wordTagPair[1]

			for destinationState in states[i]: #For each possible tag
				for sourceState in states[i-1]: #Loop through each transition to this tag
					if word in tagWordProbs[destinationState]:
						#The log probability is log( p(t'|t) * p(w|t) )
						logProb = math.log10(tagBigramProbs[sourceState][destinationState]*tagWordProbs[destinationState][word])
					else:
						logProb = math.log10(tagBigramProbs[sourceState][destinationState]*0.001/float(vocabSize)) #Give default value- unknown word
					newWeight = viterbi[i-1][sourceState] + logProb #Add to current running log weight
					if newWeight > viterbi[i][destinationState] or viterbi[i][destinationState] == 0: #If we find a new best weight or it hasn't been set yet
						viterbi[i][destinationState] = newWeight
						pointers[i][destinationState] = sourceState

		#Do final state: 
		destinationState = "</s>"
		for sourceState in states[numWords]:
			logProb = math.log10(tagBigramProbs[sourceState][destinationState])
			newWeight = viterbi[numWords][sourceState] + logProb
			if newWeight > viterbi[numWords+1][destinationState] or viterbi[numWords+1][destinationState] == 0:
				viterbi[numWords+1][destinationState] = newWeight
				pointers[numWords+1][destinationState] = sourceState

		#Reconstruct tag sequence from pointers
		tagSequence = [0 for i in range(0, numWords + 1)]
		i = numWords
		tagSequence[numWords] = "</s>"
		while i > 0:
			tagSequence[i-1] = pointers[i+1][tagSequence[i]]
			i = i-1

		return tagSequence


