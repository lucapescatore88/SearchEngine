from engineutils import res
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import pickle
import sys

data = pickle.load(open(res.RESOURCES+"XGScored.pkl"))
corr = data[['TweetCounts','TweetFollows','money','isFamous']].corr()

#print data.head()

sb.set()
sb.heatmap(corr,annot=True,cmap="Blues")
plt.savefig(res.PLOTS+"VarsCorrelation.pdf")
plt.clf()


dataFam   = data.loc[data['isFam']==1,:]
dataNoFam = data.loc[data['isFam']==0,:]

#print dataNoFam.loc[dataNoFam['TweetCounts'] > 0, ['TweetCounts','isFamous','isFam']].head(30)

sb.jointplot(x="TweetCounts", y="TweetFollows", data=data)
plt.savefig(res.PLOTS+"Counts_vs_Follow_hexagons.pdf")
plt.clf()

sb.relplot(x="TweetCounts", y="TweetFollows", hue="isFam", data=data)
plt.savefig(res.PLOTS+"Counts_vs_Follow.pdf")
plt.clf()


for var in ['isFamous','TweetFollows','TweetCounts'] :
    sb.distplot(dataNoFam[var],kde=False)
    sb.distplot(dataFam[var],kde=False)
    plt.legend(['Famous (mean = %.2f)' % dataFam[var].mean(),
        'Normals (mean = %.2f)' % dataNoFam[var].mean()], loc='best')
    plt.yscale('log')
    plt.savefig(res.PLOTS+var+".pdf")
    plt.clf()


eff = []
rej = []
cuts = np.linspace(dataFam['isFamous'].min(),dataFam['isFamous'].max(),30)
totFam = float(len( dataFam.values ))
totNoFam = float(len( dataNoFam.values ))
mindist = 100
bestcut = -1
besteff = -1
for c in cuts :
    eff.append( len( dataFam.loc[dataFam['isFamous']>c].values ) / totFam )
    rej.append( len( dataNoFam.loc[dataNoFam['isFamous']<c].values ) / totNoFam )
    dist = (1 -eff[-1])**2 +(1-rej[-1])**2
    if dist < mindist : 
        mindist = dist
        bestcut = c
        besteff = eff[-1]

print "The best cut is:", bestcut, "with efficiency", besteff
plt.plot(eff,rej)
plt.xlabel('Efficiency')
plt.ylabel('Rejection')
plt.savefig(res.PLOTS+"XG_ROC.pdf")
plt.clf()





