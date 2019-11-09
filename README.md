# chatbot

	Selected the most frequent 8000 words in Twitter Chat-log with 100k query-response pair to build vocabulary, expelled long and short sentences, and pretrained GloVe word-vectors to initialize embedding

	Built 3-layer bidirectional GRUs encoder, and 3-layer unidirectional GRUs decoder using attention and dropout.

	Applied masked negative log likelihood loss and trained the model through random Teacher Forcing. 

	Utilized beam search to find the sentence with the highest probability and get BLEU Score of 0.31.
