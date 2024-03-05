# import these modules
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print("rocks :", lemmatizer.lemmatize("rocks"))
print("trees  :", lemmatizer.lemmatize("trees"))  # is this a joke?

# a denotes adjective in "pos"
print("better :", lemmatizer.lemmatize("better", pos="a"))
