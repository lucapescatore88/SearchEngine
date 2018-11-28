from engineutils import config, resroot, saveDataWithPrediction
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from scipy.spatial.distance import cosine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
from googleutils import parseGoogle
from nltk.corpus import stopwords
import string, math, os, sys
import pandas as pd
import yaml, pickle
import numpy as np

spark = None
if config['usespark'] :
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("NLPVectorisation").getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

pcamodelfile   = resroot+"PCA_politician_model.pkl"
modelfile      = resroot+"NLP_politician_model.pkl"
mapfile        = resroot+"NLP_politician_wordmap.pkl"
nlpoutfile     = resroot+"NLP_simple_out.pkl"
fullnlpoutfile = resroot+"NLP_out.pkl"

trained_model    = pickle.load(open(modelfile))
trained_word_map = pickle.load(open(mapfile))
trained_pca      = pickle.load(open(pcamodelfile))

def keepword(w) :
    if len(w) < 4 : return False
    if "/" in w or "www" in w or "http" in w: return False
    if "''" in w or "`" in w or '\t' in w or '~' in w : return False
    if not all(ord(char) < 128 for char in w) : return False
    if any(char.isdigit() for char in w) : return False
    return True

### Splits words, simplifies them (lower case, remove stopwords, lemmatize)
def prepareNLPData(data) :

    punct = list(string.punctuation)    ## Punctuation to remove
    stopw = stopwords.words('english')  ## Stopwords to remove

    ## Divide words, and remove punctuation and stopwords
    tokens = word_tokenize(data)
    newtokens, newtokens2 = [], []
    for tok in tokens : newtokens.extend(tok.split("-"))
    for tok in newtokens : newtokens2.extend(tok.split("'"))

    filtered_words = [w.lower() for w in newtokens2 if w.lower() not in stopw]
    filtered_words = [w for w in filtered_words if w not in punct]
    filtered_words = [w for w in filtered_words if keepword(w)]
    
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
        nodash     = words.flatMap(lambda x: x.split("-") )
        noapostr   = nodash.flatMap(lambda x: x.split("'") )
        filtered   = noapostr.filter(lambda w : keepword(w) )
        lowords    = filtered.map(lambda w: w.lower())
        nostop     = lowords.filter(lambda w : w not in stopw and w not in punct)
        lemmatised = nostop.map(lambda w: lemmatizer.lemmatize(w))

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

    vec = np.zeros(dim)
    vocabulary = word_map.keys()
    for t in tokens :
        if t in vocabulary : 
            vec[word_map[t]] += 1

    ## Normalise to get word frequency
    norm = vec.sum()
    if norm > 1e-6 : vec = vec / float(norm)
    else : vec = np.zeros(dim)

    if label is not None : vec[-1] = label
    return vec


### Given a list texts applies lemmatisation and creates the map
def makeDataSet(people,word_map={},fillWM=True) :

    ## Build map from words to vector index
    toks = []
    for pol in people :
        lemms = prepareNLPData(pol)
        if fillWM : word_map = fillWordMap(lemms,word_map)
        toks.append(lemms)
    
    return word_map, toks


def getVectorDF(alltoks,word_map,label=None) :
    recs = []
    vocaulary = word_map.keys()
    indices   = word_map.values()
    for toks in alltoks:
        d = { vocaulary[indices.index(i)] : x for i,x in enumerate(tokensToVector(toks, word_map)) }
        if label is not None : d['Label'] = 1
        recs.append(d)
    return pd.DataFrame(recs)

### Function to train a NLP model for classifying politicians 
def trainNLPModel(politicians,normals) :

    ### Tockenise and clean data and fill the word map
    poltexts = politicians.loc[:,'googletext'].tolist()
    normtexts = normals.loc[:,'googletext'].tolist()
    word_map = None
    if config['usespark'] :
        word_map, pol_toks = makeDataSetSpark(poltexts,word_map_simple)
        word_map, norm_toks = makeDataSetSpark(normtexts,word_map)
    else :    
        word_map, pol_toks = makeDataSet(poltexts,word_map_simple)
        word_map, norm_toks = makeDataSet(normtexts,word_map)

    #print "I studied, now I know %i words" % len(word_map)
    ### Since it's a long computation time save intermediate steps for testing
    #word_map  = pickle.load(open("mywordmap.pkl"))
    #pol_toks  = pickle.load(open("poltoks.pkl"))
    #norm_toks = pickle.load(open("normtoks.pkl"))

    with open("mywordmap.pkl","w") as of :
        pickle.dump(word_map,of)
    with open("poltoks.pkl","w") as of :
        pickle.dump(pol_toks,of)
    with open("normtoks.pkl","w") as of :
        pickle.dump(norm_toks,of)

    ### Vectorise data
    dataPol  = getVectorDF(pol_toks,word_map,label=1)
    dataNorm = getVectorDF(norm_toks,word_map,label=1)
    with open("dataNorm.pkl","w") as of :
        pickle.dump(dataNorm,of)
    with open("dataPol.pkl","w") as of :
       pickle.dump(dataPol,of)

    ### Since it's a long computation time save intermediate steps for testing
    #dataNorm = pickle.load(open("dataNorm.pkl"))
    #dataPol = pickle.load(open("dataPol.pkl"))
    
    ### Train the model
    dat = dataPol.append(dataNorm)
    feats = dat.drop('Label',axis=1)
    labs  = dat['Label']

    feats_train, feats_test, labs_train, labs_test \
        = train_test_split(feats, labs, test_size=0.2)

    #print "Doing PCA to make life easier for the model"
    pca = PCA(n_components=10)
    pca_train = pca.fit_transform(feats_train)

    #print "Fitting Logistic Regression model"
    model = LogisticRegression()
    model.fit(pca_train,labs_train)
    #model.fit(feats_train,labs_train)

    ## Save model for future use
    with open(modelfile,"w") as of :
        pickle.dump(model,of)
    with open(pcamodelfile,"w") as of :
        pickle.dump(pca,of)

    pca_test = pca.transform(feats_test)
    #print "Classification rate", model.score(feats_test,labs_test)
    print "Classification rate", model.score(pca_test,labs_test)

    ## Save scored data for plotting
    allpol_pca  = pca.transform(dataPol.drop('Label',axis=1))
    allnpol_pca = pca.transform(dataNorm.drop('Label',axis=1))
    politicians['scorePol'] = model.predict_proba(allpol_pca)[:,0]
    normals['scorePol'] = model.predict_proba(allnpol_pca)[:,0]

    #politicians['scorePol'] = model.predict_proba(dataPol.drop('Label',axis=1))[:,1]
    #normals['scorePol'] = model.predict_proba(dataNorm.drop('Label',axis=1))[:,1]
    eff, thr = optimiseThr(politicians,normals,'scorePol')
    print "Best thrshold", thr

    with open(fullnlpoutfile,"w") as of :
        pickle.dump(politicians.append(normals),of)

    return model

def scorePolitician(person) :

    if config['usespark'] :
        word_map, alltoks = makeDataSetSpark([person],trained_word_map,False)
    else :
        word_map, alltoks = makeDataSet([person],trained_word_map,False)
    
    data     = getVectorDF(alltoks,word_map)
    features = trained_pca.transform(data)

    ### N.B.: Value is rescaled to be between 0 and 1 (obtained in plotNLPout.py)
    classindex = trained_model.classes_.tolist().index(1)
    score = (trained_model.predict_proba(features)[:,classindex][0]-0.40300)/0.047346
    print "Politician score is", score
    return score

### Given a text return if it is a politician 
### N.B.: Uses a pretrained model
### N.B.: Threshold can be changed in cfg.yml
def isPolitician(person) :

    score = scorePolitician(person)
    print "Politician score is", score
    return predicted > config['isPolitician_prob_threshold']



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
        word_map, alltoks = makeDataSet([person],word_map_simple,False)

    vec        = tokensToVector(alltoks[0], word_map_simple)
    score      = cosvec(testvector,vec)

    return score > config['simple_nlp_thr']


### Function to train the simplified model
def trainSimpleNLPModel(politicians,normals) :

    poltexts = politicians.loc[:,'googletext'].tolist()
    normtexts = normals.loc[:,'googletext'].tolist()
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

    politicians['scorePolSimple'] = pd.Series(pol_scores)
    normals['scorePolSimple'] = pd.Series(norm_scores)
    eff, thr = optimiseThr(politicians,normals,'scorePolSimple')

    print "Mean Politics = ", np.mean(pol_scores)
    print "Mean Normal   = ", np.mean(norm_scores)
    print "Threshold     = ", thr

    with open(nlpoutfile,"w") as of :
        pickle.dump(politicians.append(normals),of)

    return thr


def optimiseThr(data1,data2,var) :

    data = data1.append(data2)
    cuts = np.linspace(data[[var]].min(),data[[var]].max(),100)
    tot1 = float(len( data1.values ))
    tot2 = float(len( data2.values ))
    
    eff, rej = [], []
    mindist = 100
    bestcut, besteff = -1, -1
    for c in cuts :
        eff.append( len( data1.loc[data1[var]>c].values ) / tot1 )
        rej.append( len( data2.loc[data2[var]<c].values ) / tot2 )
        dist = (1 -eff[-1])**2 +(1-rej[-1])**2
        if dist < mindist : 
            mindist = dist
            bestcut = c
            besteff = eff[-1]

    return besteff, bestcut


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

