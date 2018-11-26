from engineutils import resroot, root
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import pickle
import sys

plots = root+"/plots/"

data = pickle.load(open(resroot+"XGScored.pkl"))
corr = data[['TweetCounts','TweetFollow','money']].corr()

sb.color_palette("Blues")
sb.heatmap(corr)
plt.savefig(plots+"VarsCorrelation.pdf")
plt.clf()

sys.exit()
dataFam   = data.loc[data['isFam']==1,'famousScore']
dataNoFam = data.loc[data['isFam']==0,'famousScore']

sb.distplot(dataFam,kde=False)
sb.distplot(dataNoFam,kde=False)
plt.legend(['Famous (mean = %.2f)' % dataFam.mean(),'Normals (mean = %.2f)' % dataNoFam.mean()], ncol=2, loc='best');
plt.savefig(plots+"XGout_seaborn.pdf")
plt.clf()

eff = []
rej = []
cuts = np.linspace(data[['famousScore']].min(),data[['famousScore']].max(),30)
totPol = float(len( dataFam.values ))
totNoPol = float(len( dataNoFam.values ))
mindist = 100
bestcut = -1
besteff = -1
for c in cuts :
    eff.append( len( dataPol.loc[data['famousScore']>c].values ) / totFam )
    rej.append( len( dataPol.loc[data['famousScore']<c].values ) / totNoFam )
    dist = (1 -eff[-1])**2 +(1-rej[-1])**2
    if dist < mindist : 
        mindist = dist
        bestcut = c
        besteff = eff[-1]

print "The best cut is:", bestcut, "with efficiency", besteff
plt.plot(eff,rej)
plt.xlabel('Efficiency')
plt.ylabel('Rejection')
plt.savefig(plots+"XG_ROC.pdf")
plt.clf()





