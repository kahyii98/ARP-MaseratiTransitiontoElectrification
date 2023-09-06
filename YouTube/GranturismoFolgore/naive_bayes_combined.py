#Combined bag of words: word unigrams and bigrams plus POS n-grams, all in a naive bayes classifier

import math

class naive_bayes_combined:
	def __init__(self, trainData = None):
		if trainData != None: #Train if the training data has been provided
			self.train(trainData)

	def train(self, trainingData):
		#Set categories
		categories = ['10', '20', '30', '40']

		#Create data structures for word counts and category counts
		wordCounts = dict()
		categoryDocumentCounts = dict()
		categoryWordCounts = dict()
		wordProbs = dict()
		for category in categories:
			wordCounts[category] = dict()
			categoryDocumentCounts[category] = 0
			categoryWordCounts[category] = 0
			wordProbs[category] = dict()
		numDocuments = 0
		vocabSet = set()

		#Read through each line; count and divide
		for line in trainingData:
			words = line.rstrip().split()
	
			#Separate first element (category) from the rest of the words
			category = words[0]
			words = words[1:]

			#Process the category
			if category not in categories:
				continue
			categoryDocumentCounts[category] += 1
			numDocuments += 1

			#Count words
			for word in words:
				parts = word.split("/")

				#Parse to get the actual word
				if len(parts) != 3:
					continue
				word = parts[0]
				pos = parts[1]
				sentencePart = parts[2]
		
				if word not in wordCounts[category]:
					wordCounts[category][word] = 0

				wordCounts[category][word] += 50
				categoryWordCounts[category] += 50
				vocabSet.add(word)

			#Count bigrams
			prevWord = words[0].split("/")[0]
			for word in words[1:]:
				parts = word.split("/")
	
				#Parse to get the actual word
				if len(parts) != 3:
					continue
				word = parts[0]
				pos = parts[1]
				sentencePart = parts[2]

				bigram = prevWord + " " + word

				if bigram not in wordCounts[category]:
					wordCounts[category][bigram] = 0

				wordCounts[category][bigram] += 50
				categoryWordCounts[category] += 50
				vocabSet.add(bigram)

				prevWord = word

			#Construct tag sequence
			tagSequence = ["<s>"]
			for word in words:
				parts = word.split("/")
				pos = parts[1]
				tagSequence.append(pos)
			tagSequence.append("</s>")

			#Construct all of the tag n-grams that can be used, for n=3,4
			tagNGrams = []

			#Add trigrams
			prevPrevTag = tagSequence[0]
			prevTag = tagSequence[1]
			for tag in tagSequence[2:]:
				tagTrigram = prevPrevTag + "/" + prevTag + "/" + tag
				tagNGrams.append(tagTrigram)	
				prevPrevTag = prevTag
				prevTag = tag	

			#Add 4-grams (only if the tag sequence has at least 4 tags, meaning it's at least two words plus <s> and </s>
			if len(tagSequence) >= 4:
				prevPrevPrevTag = tagSequence[0]
				prevPrevTag = tagSequence[1]	
				prevTag = tagSequence[2]
				for tag in tagSequence[3:]:
					tag4Gram = prevPrevPrevTag + "/" + prevPrevTag + "/" + prevTag + "/" + tag
					tagNGrams.append(tag4Gram)
					prevPrevPrevTag = prevPrevTag
					prevPrevTag = prevTag
					prevTag = tag	

			#Add these tag trigrams and 4-grams to the bag of words
			for tagNGram in tagNGrams:
				if tagNGram not in wordCounts[category]:
					wordCounts[category][tagNGram] = 0

				wordCounts[category][tagNGram] += 1
				categoryWordCounts[category] += 1
				vocabSet.add(tagNGram)

		#Set up smoothing
		delta = 100
		vocabSize = len(vocabSet)

		#Calculate probabilities for each category
		categoryProbs = dict()
		for category in categories:
			categoryProbs[category] = float(categoryDocumentCounts[category]) / numDocuments
	
		#Calculate probabilities for each word
		for category in categories:
			for word in wordCounts[category]:
				wordProbs[category][word] = ( float(wordCounts[category][word]) + delta ) / ( categoryWordCounts[category] + vocabSize*delta )

		#Calculate probabilities for unknown words
		unknownWordProb = dict()
		for category in categories:
			unknownWordProb[category] = float(delta) / ( categoryWordCounts[category] + vocabSize*delta )

		#Save relevant data structures
		self.wordProbs = wordProbs
		self.unknownWordProb = unknownWordProb
		self.categories = categories

	def decode(self, words, tagSequence):
		#Get relevant data structures
		wordProbs = self.wordProbs
		unknownWordProb = self.unknownWordProb
		categories = self.categories

		#Construct all of the tag n-grams that can be used, for n=3,4
		tagNGrams = []

		#Add trigrams
		prevPrevTag = tagSequence[0]
		prevTag = tagSequence[1]
		for tag in tagSequence[2:]:
			tagTrigram = prevPrevTag + "/" + prevTag + "/" + tag
			tagNGrams.append(tagTrigram)	
			prevPrevTag = prevTag
			prevTag = tag	

		#Add 4-grams (only if the tag sequence has at least 4 tags, meaning it's at least two words plus <s> and </s>
		if len(tagSequence) >= 4:
			prevPrevPrevTag = tagSequence[0]
			prevPrevTag = tagSequence[1]	
			prevTag = tagSequence[2]
			for tag in tagSequence[3:]:
				tag4Gram = prevPrevPrevTag + "/" + prevPrevTag + "/" + prevTag + "/" + tag
				tagNGrams.append(tag4Gram)
				prevPrevPrevTag = prevPrevTag
				prevPrevTag = prevTag
				prevTag = tag	

		#Add up log probabilities for each word and n-gram, for each category:
		thisLineLogProb = dict()
		for category in categories:
			thisLineLogProb[category] = 0	

			#Count word unigrams
			for word in words:
				parts = word.split("/")

				#Parse to get the actual word
				if len(parts) != 3:
					continue
				word = parts[0]
				pos = parts[1]
				sentencePart = parts[2]

				#Increment probability
				if word in wordProbs[category]:
					thisWordProb = wordProbs[category][word]
				else:
					thisWordProb = unknownWordProb[category]
				thisLineLogProb[category] += math.log10(thisWordProb)

			#Count bigrams
			prevWord = words[0].split("/")[0]
			for word in words[1:]:
				parts = word.split("/")
	
				#Parse to get the actual word
				if len(parts) != 3:
					continue
				word = parts[0]
				pos = parts[1]
				sentencePart = parts[2]

				bigram = prevWord + " " + word

				#Increment probability
				if bigram in wordProbs[category]:
					thisWordProb = wordProbs[category][bigram]
				else:
					thisWordProb = unknownWordProb[category]
				thisLineLogProb[category] += math.log10(thisWordProb)

			#Count tag n-grams
			for tagNGram in tagNGrams:
				#Increment probability
				if tagNGram in wordProbs[category]:
					thisWordProb = wordProbs[category][tagNGram]
				else:
					thisWordProb = unknownWordProb[category]
				thisLineLogProb[category] += math.log10(thisWordProb)

		#For this line, choose the category with the biggest log prob
		winningCategory = max(thisLineLogProb, key=thisLineLogProb.get)
		
		return winningCategory











