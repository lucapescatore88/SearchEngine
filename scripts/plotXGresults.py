from engineutils import res
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import pickle
import sys

data = pickle.load(open(res.RESOURCES+"XGScored.pkl"))
corr = data[['TweetCounts','TweetFollow','money','isFamous']].corr()

#print data.head()

sb.set()
sb.heatmap(corr,annot=True,cmap="Blues")
plt.savefig(plots+"VarsCorrelation.pdf")
plt.clf()

#sys.exit()
dataFam   = data.loc[data['isFam']==1,:]
dataNoFam = data.loc[data['isFam']==0,:]

sb.distplot(dataFam['isFamous'],kde=False)
sb.distplot(dataNoFam['isFamous'],kde=False)
plt.legend(['Famous (mean = %.2f)' % dataFam['isFamous'].mean(),'Normals (mean = %.2f)' % dataNoFam['isFamous'].mean()], ncol=2, loc='best');
plt.savefig(res.PLOTS+"XGout_seaborn.pdf")
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





