# import these modules
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print("rocks :", lemmatizer.lemmatize("rocks"))
print("has  :", lemmatizer.lemmatize("has"))  # is this a joke?

# a denotes adjective in "pos"
print("better :", lemmatizer.lemmatize("better", pos="a"))
