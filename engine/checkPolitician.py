from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cosine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer

from engineutils import config, resroot
from googleutils import parseGoogle
from nltk.corpus import stopwords
import string, math, os, sys
import pandas as pd
import yaml, pickle
import numpy as np
import math

modelfile  = resroot+"NLP_politician_model.pkl"
mapfile    = resroot+"NLP_politician_wordmap.pkl"
nlpoutfile = resroot+"NLP_simple_out.pkl"

trained_model    = pickle.load(open(modelfile))
trained_word_map = pickle.load(open(mapfile))

### Splits words, simplifies them (lower case, remove stopwords, lemmatize)
def prepareNLPData(data) :

    punct = list(string.punctuation)    ## Punctuation to remove
    stopw = stopwords.words('english')  ## Stopwords to remove

    ## Divide words, and remove punctuation and stopwords
    tokens = word_tokenize(data)
    filtered_words = [w.lower() for w in tokens if w.lower() not in stopw]
    filtered_words = [w for w in filtered_words if w not in punct]

    ## Lemmatise words to limit phasespace: e.g dogs --> dog, doing --> do 
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatised = [wordnet_lemmatizer.lemmatize(t) for t in filtered_words]

    return lemmatised


### Creates a dictionary mapping words to indices
def getWordMap(toks,word_map = {}) :

    ### In case we need to add to a previous dictionary
    cur_index = len(word_map)

    ### Build vectors of words
    for tok in toks :
        if tok not in word_map :
            word_map[tok] = cur_index
            cur_index+=1

    return word_map


### Given lemmatised tokens and a wordmap created a vector
def tokensToVector(tokens, word_map, label = None) :
    
    dim = len(word_map)
    if label is not None : dim += 1

    x = np.zeros(dim)
    keys = word_map.keys()
    for t in tokens :
        if t in keys : 
            x[word_map[t]] += 1

    ## Normalise to get word frequency
    norm = x.sum()
    if norm > 0 : x = x / float(norm)

    if label is not None : x[-1] = label
    return x


### Given a list texts applies lemmatisation and creates the map
def makeDataSet(people,word_map={}) :

    ## Build map from words to vector index
    toks = []
    for pol in people :
        lemms = prepareNLPData(pol)
        word_map = getWordMap(lemms,word_map)
        toks.append(lemms)
    
    return word_map, toks


### Function to train a NLP model for classifying politicians 
def trainNLPModel(politicians,normals) :

    print "Tockenising and vectorising"
    word_map, pol_toks = makeDataSet(politicians,word_map_simple)
    word_map, norm_toks = makeDataSet(normals,word_map)
    
    N = len(pol_toks) + len(norm_toks)
    data = np.zeros((N,len(word_map)+1))
    i = 0
    for toks in pol_toks :
        data[i,:] = tokensToVector(toks, word_map, label = 1)
        i +=1
    for toks in norm_toks :
        data[i,:] = tokensToVector(toks, word_map, label = 0)
        i +=1

    ### Shuffle to avoid using only politicians or only normals for training
    np.random.shuffle(data)

    features = data[:,:-1]
    labels   = data[:,-1]

    ntest = int(-0.2 * len(data))
    features_train = features[:ntest,]
    labels_train = labels[:ntest,]
    features_test  = features[ntest:,]
    labels_test  = labels[ntest:,]

    print "Fitting Logistic Regression model"
    model = LogisticRegression()
    model.fit(features_train,labels_train)

    ## Save model for future use
    pickle.dump(model,open(modelfile,"w"))
    pickle.dump(word_map,open(mapfile,"w"))
    print "Classification rate", model.score(features_test,labels_test)

    return model


### Given a text return if it is a politician 
### N.B.: Uses a pretrained model
### N.B.: Threshold can be changed in cfg.yml
def isPolitician(person) :

    word_map, alltoks = makeDataSet([person],trained_word_map)
    data = np.zeros((len(toks),len(word_map)))
    i = 0
    for toks in alltoks :
        data[i,:] = tokensToVector(toks, word_map, label = None)
        i +=1
    
    #Code to check the weight given to each word
    #for w,i in word_map.iteritems() :
    #    weight = model.coef_[0][i]
    #    if(abs(weight)>thr) :
    #        print word, weight
    return int(trained_model.predict(data) > config['nlp_thr'])


### Calculates cosigne between to vectors as a similarity measure (could use np.dot)
def cosvec(v1,v2) :
    
    if len(v1) != len(v2) :
        return "len(v1) = "+str(len(v1))+" len(v2) = "+str(len(v2))+" They must be equal!"
        return 0.

    xy, xx, yy = 0., 0., 0.
    for i in range(len(v1)) :
        xx += v1[i]*v1[i]
        yy += v2[i]*v2[i]
        xy += v1[i]*v2[i]
    
    if math.sqrt(xx)*math.sqrt(yy) < 1e-6 :
        #print "One of the two vectors is empty!!"
        return 0.

    return float(xy) / (math.sqrt(xx)*math.sqrt(yy))



### Get the simplified map for these few political words to test

politics_words = ["politics","election","decision","minister","senator","president",
                        "parliament","congress","vote","nation","party","leader","idea",
                        "democratic","war","welfare","internaitonal","constitution",
                        "institution","state","head","military","governament",
                        "tyranny","federal","global","corruption","power","referendum"]

def getTestMap() :

    lemms             = prepareNLPData(' '.join(politics_words))
    word_map_simple   = getWordMap(lemms)
    #print lemms
    #print word_map_simple

    return lemms, word_map_simple

### Make the test objects globally available
lemms_simple, word_map_simple = getTestMap()


### Returns average score on a list of texts
def scoreSimpleNLP(people) :

    #word_map, alltoks = makeDataSet(people,word_map_simple)
    #testvector = tokensToVector(lemms_simple, word_map, label = None) 
    word_map, alltoks = makeDataSet(people)
    testvector = tokensToVector(lemms_simple, word_map_simple, label = None) 

    vecs   = [ tokensToVector(toks, word_map_simple, label = None) for toks in alltoks ]
    scores = [ cosvec(testvector,vec) for vec in vecs ]

    return np.mean(scores)


### Returns true if the score of the simple model passes a threshold
### N.B.: Threshold was pretrained and can be changed in cfg.yml
def isPoliticianSimple(person) :
    return scoreSimpleNLP([person]) > config['simple_nlp_thr']


### Function to train the simplified model
def trainSimpleNLPModel(politicians,normals) :

    #print "Vectorising"
    #vectorizer = CountVectorizer()
    #texts = [' '.join(politics_words)]
    #texts.extend(politicians)
    #texts.extend(normals)
    #vectorizer.fit(texts)
    #print "Vectorised"
    #testvec  = vectorizer.transform([' '.join(politics_words)])
    #print testvec[0]
    #print testvec[0][:30]
    #pol_vecs = vectorizer.transform(politicians)
    #norm_vecs = vectorizer.transform(normals)

    word_map, pol_toks = makeDataSet(politicians)
    word_map, norm_toks = makeDataSet(normals)
    
    testvector = tokensToVector(lemms_simple, word_map_simple, label = None)
    pol_vecs   = [ tokensToVector(toks, word_map_simple, label = None) for toks in pol_toks ]    
    norm_vecs  = [ tokensToVector(toks, word_map_simple, label = None) for toks in norm_toks ]

    ntestpol  = int(-0.2 * len(pol_vecs))
    ntestnorm = int(-0.2 * len(norm_vecs))
    pol_scores  = [ cosvec(testvector,vec) for vec in pol_vecs[:ntestpol] ]
    norm_scores = [ cosvec(testvector,vec) for vec in norm_vecs[:ntestnorm] ]

    ## The treshold will be the middle point between the two averages
    thr = (np.mean(pol_scores) + np.mean(norm_scores))/2.
    print "Mean Politics = ", np.mean(pol_scores)
    print "Mean Normal   = ", np.mean(norm_scores)
    print "Threshold     = ", thr

    outdata = [ {'isPol':1, 'simpleNLPScore':score} for score in pol_scores ]
    outdata.extend([ {'isPol':0, 'simpleNLPScore':score} for score in norm_scores])
    df = pd.DataFrame(outdata)
    pickle.dump(df,open(nlpoutfile,"w"))


    if thr is not None :
        pol_scores  = [ cosvec(testvector,vec) for vec in pol_vecs[ntestpol:] ]
        norm_scores = [ cosvec(testvector,vec) for vec in norm_vecs[ntestnorm:] ]
        print "Pos politics ", sum([1 for x in pol_scores if x > thr ]) / float(len(pol_scores))
        print "Pos normals ", sum([1 for x in norm_scores if x < thr ]) / float(len(norm_scores))
        print "Neg politics ", sum([1 for x in pol_scores if x < thr ]) / float(len(pol_scores))
        print "Neg normals ", sum([1 for x in norm_scores if x > thr ]) / float(len(norm_scores))

    return thr



if __name__ == "__main__" :

    print "This scripts assumes you ran python engine/googleutils.py before"
    print "It will use data from ", resroot+"GoogleDF.pkl"
    if not os.path.exists(resroot+"GoogleDF.pkl") :
        print "Please run 'python engine/googleutils.py' to get data for training"
        sys.exit()

    from googleutils import parseGoogle
    import pickle
    data = pickle.load(open(resroot+"GoogleDF.pkl"))
    
    politicians = data.loc[data['isPol']==1,'googletext'].tolist()
    normals     = data.loc[data['isPol']==0,'googletext'].tolist()

    print "Training Simple model"
    trainSimpleNLPModel(politicians,normals)
    print "Training Logistic model"
    trainNLPModel(politicians,normals)

