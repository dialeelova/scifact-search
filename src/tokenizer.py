import re
from functools import lru_cache

from nltk.stem import PorterStemmer

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been
before being below between both but by can did do does doing down during each
few for from further had has have having he her here hers herself him himself
his how i if in into is it its itself just me more most my myself no nor not
now of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when
where which while who whom why will with you your yours yourself yourselves
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_stemmer = PorterStemmer()

@lru_cache(maxsize=None)
def _stem(word):
    return _stemmer.stem(word)

def tokenize(text):
    words = _TOKEN_RE.findall(text.lower())
    return [_stem(w) for w in words if w not in STOPWORDS]

if __name__ == "__main__":
    print(tokenize("Running cats"))
    print(tokenize("The 1,000 genomes project enables mapping of rare variants."))
