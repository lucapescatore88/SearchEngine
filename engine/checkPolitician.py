from engineutils import config, resroot, saveDataWithPrediction
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cosine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from googleutils import parseGoogle
from nltk.corpus import stopwords
import string, math, os, sys
import pandas as pd
import yaml, pickle
import numpy as np
import math

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("NLPVectorisation").getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

modelfile  = resroot+"NLP_politician_model.pkl"
mapfile    = resroot+"NLP_politician_wordmap.pkl"
nlpoutfile = resroot+"NLP_simple_out.pkl"
fullnlpoutfile = resroot+"NLP_out.pkl"

#Function to check the weight given to each word
def checkWeights(word_map, model):

    recs = []
    for w,i in word_map.iteritems() :
        recs.append( {"word": w, "coeff" : model.coef_[0][i] })
    df = pd.DataFrame(recs)
    print "Most politics related words"
    print df.sort_values('coeff', ascending=False).head(30)
    print "Least politics related words"
    print df.sort_values('coeff').head(30)


trained_model    = pickle.load(open(modelfile))
trained_word_map = pickle.load(open(mapfile))
checkWeights(trained_word_map,trained_model)

### Splits words, simplifies them (lower case, remove stopwords, lemmatize)
def prepareNLPData(data) :

    punct = list(string.punctuation)    ## Punctuation to remove
    stopw = stopwords.words('english')  ## Stopwords to remove

    ## Divide words, and remove punctuation and stopwords
    tokens = word_tokenize(data)
    filtered_words = [w.lower() for w in tokens if w.lower() not in stopw]
    filtered_words = [w for w in filtered_words if w not in punct]

    ## Lemmatise words to limit phasespace: e.g dogs --> dog, doing --> do 
    lemmatizer = WordNetLemmatizer()
    lemmatised = [lemmatizer.lemmatize(w) for w in filtered_words]

    return lemmatised


def makeDataSetSpark(people,word_map={},fillWM=True) :

    punct = list(string.punctuation)    ## Punctuation to remove
    stopw = stopwords.words('english')  ## Stopwords to remove
    lemmatizer = WordNetLemmatizer()

    spark.sparkContext.broadcast(punct)
    spark.sparkContext.broadcast(stopw)
    spark.sparkContext.broadcast(lemmatizer)

    # Load up our data and convert it to the format MLLib expects.
    toks = []
    for person in people :
        inputs     = spark.sparkContext.parallelize([person])
        words      = inputs.flatMap(lambda x: word_tokenize(x) )
        nolink     = words.filter(lambda w : "//" not in w and "www" not in w and "http" not in w )
        nodigit    = nolink.filter(lambda w : not any(char.isdigit() for char in w) )
        nodash     = nodigit.flatMap(lambda x: x.split("-") )
        lowords    = nodash.map(lambda w: w.lower())
        nostop     = lowords.filter(lambda w : w not in stopw and w not in punct)
        nounicode  = nostop.filter(lambda w : all(ord(char) < 128 for char in w) )
        lemmatised = nounicode.map(lambda w: lemmatizer.lemmatize(w))

        toks.append(lemmatised.collect())
        if fillWM : fillWordMap(toks[-1],word_map)
    
    return word_map, toks


### Creates a dictionary mapping words to indices
def fillWordMap(toks,word_map = {}) :

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
    for t in tokens :
        if t in word_map.keys() : 
            x[word_map[t]] += 1

    ## Normalise to get word frequency
    norm = x.sum()
    if norm > 0 : x = x / float(norm)

    if label is not None : x[-1] = label
    return x


### Given a list texts applies lemmatisation and creates the map
def makeDataSet(people,word_map={},fillWM=True) :

    ## Build map from words to vector index
    toks = []
    for pol in people :
        lemms = prepareNLPData(pol)
        if fillWM : word_map = fillWordMap(lemms,word_map)
        toks.append(lemms)
    
    return word_map, toks


### Function to train a NLP model for classifying politicians 
def trainNLPModel(politicians,normals) :

    poltexts = politicians.loc[:,'googletext'].tolist()
    normtexts = politicians.loc[:,'googletext'].tolist()
    if config['usespark'] :
        word_map, pol_toks = makeDataSetSpark(poltexts,word_map_simple)
        word_map, norm_toks = makeDataSetSpark(normtexts,word_map)
    else :
        word_map, pol_toks = makeDataSet(poltexts,word_map_simple)
        word_map, norm_toks = makeDataSet(normtexts,word_map)

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
    #with open(modelfile,"w") as of :
    #    pickle.dump(model,of)
    #with open(mapfile,"w") as of :
    #    pickle.dump(word_map,of)

    print "Classification rate", model.score(features_test,labels_test)

    dataPol = np.zeros((len(pol_toks),len(word_map)))
    i = 0
    for toks in pol_toks :
        dataPol[i,:] = tokensToVector(toks, word_map)
        i +=1
    politicians['scorePol'] = model.predict(dataPol)
    dataNoPol = np.zeros((len(norm_toks),len(word_map)))
    i = 0
    for toks in norm_toks :
        dataNoPol[i,:] = tokensToVector(toks, word_map)
        i +=1
    normals['scorePol'] = model.predict(dataNoPol)
    
    with open(fullnlpoutfile,"w") as of :
        pickle.dump(politicians.append(normals),of)

    return model


### Given a text return if it is a politician 
### N.B.: Uses a pretrained model
### N.B.: Threshold can be changed in cfg.yml
def isPolitician(person) :

    if config['usespark'] :
        word_map, alltoks = makeDataSetSpark([person],trained_word_map,False)
    else :
        word_map, alltoks = makeDataSet([person],trained_word_map,False)
    
    data = np.zeros((1,len(word_map)))
    data[0,:] = tokensToVector(alltoks, word_map)

    predicted = trained_model.predict(data)
    return predicted[0] > config['nlp_thr']



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

lemms_simple      = prepareNLPData(' '.join(politics_words))
word_map_simple   = fillWordMap(lemms_simple)
testvector        = tokensToVector(lemms_simple, word_map_simple)

### Returns true if the score of the simple model passes a threshold
### N.B.: Threshold was pretrained and can be changed in cfg.yml
def isPoliticianSimple(person) :

    if config['usespark'] :
        word_map, alltoks = makeDataSetSpark([person],word_map_simple,False)
    else :
        word_map, alltoks = makeDataSet([person],word_map_simplem,False)

    vec        = tokensToVector(alltoks[0], word_map_simple)
    score      = cosvec(testvector,vec)

    return score > config['simple_nlp_thr']


### Function to train the simplified model
def trainSimpleNLPModel(politicians,normals) :

    poltexts = politicians.loc[:,'googletext'].tolist()
    normtexts = politicians.loc[:,'googletext'].tolist()
    if config['usespark'] :
        word_map, pol_toks = makeDataSetSpark(poltexts)
        word_map, norm_toks = makeDataSetSpark(normtexts)
    else :
        word_map, pol_toks = makeDataSet(poltexts)
        word_map, norm_toks = makeDataSet(normtexts)

    pol_vecs   = [ tokensToVector(toks, word_map_simple) for toks in pol_toks ]    
    norm_vecs  = [ tokensToVector(toks, word_map_simple) for toks in norm_toks ]

    ntestpol  = int(-0.2 * len(pol_vecs))
    ntestnorm = int(-0.2 * len(norm_vecs))
    pol_scores  = [ cosvec(testvector,vec) for vec in pol_vecs[:ntestpol] ]
    norm_scores = [ cosvec(testvector,vec) for vec in norm_vecs[:ntestnorm] ]

    ## The treshold will be the middle point between the two averages
    thr = (np.mean(pol_scores) + np.mean(norm_scores))/2.
    print "Mean Politics = ", np.mean(pol_scores)
    print "Mean Normal   = ", np.mean(norm_scores)
    print "Threshold     = ", thr

    politicians['scorePolSimple'] = pd.Series(pol_scores)
    normals['scorePolSimple'] = pd.Series(norm_scores)

    with open(nlpoutfile,"w") as of :
        pickle.dump(politicians.append(normals),of)

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

    import pickle
    data = pickle.load(open(resroot+"GoogleDF.pkl"))
    
    politicians = data.loc[data['isPol']==1]
    normals     = data.loc[data['isPol']==0]

    print "Training Simple model"
    trainSimpleNLPModel(politicians,normals)
    print "Training Logistic model"
    trainNLPModel(politicians,normals)

