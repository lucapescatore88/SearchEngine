from engineutils import root, resroot
import matplotlib.pyplot as plt
import seaborn as sb
import pickle

plots = root+"/plots/"

data = pickle.load(open(resroot+"TwitterDF.pkl"))

print data.describe()
total   = len(data.loc[:,['isFam']].values)
counts0 = len(data.loc[data['TweetCounts']==0,['isFam']].values)

print "Mean count for famous: ", data.loc[data['isFam']==True,['TweetCounts']].mean()
print "Mean count for non-famous: ", data.loc[data['isFam']==False,['TweetCounts']].mean()
print "Max count for famous: ", data.loc[data['isFam']==True,['TweetCounts']].max()
print "Max count for non-famous: ", data.loc[data['isFam']==False,['TweetCounts']].max()

print "Mean follow for famous: ", data.loc[data['isFam']==True,['TweetFollow']].mean()
print "Mean follow for non-famous: ", data.loc[data['isFam']==False,['TweetFollow']].mean()
print "Max follow for famous: ", data.loc[data['isFam']==True,['TweetFollow']].max()
print "Max follow for non-famous: ", data.loc[data['isFam']==False,['TweetFollow']].max()

sb.relplot(x="TweetCounts", y="TweetFollow", hue = "isFam", data=data);
plt.savefig(plots+"Twitter_hits_vs_follows.pdf")
plt.clf()

#teetcounts = data['TweetCounts'].toarray()
sb.distplot(data.loc[data['isFam']==True,['TweetCounts']],kde=False)
sb.distplot(data.loc[data['isFam']==False,['TweetCounts']],kde=False)
#plt.yscale('symlog', linthreshy=0.05)
plt.savefig(plots+"Twitter_hits.pdf")
plt.clf()

sb.distplot(data.loc[data['isFam']==False,['TweetFollow']],kde=False)
sb.distplot(data.loc[data['isFam']==True,['TweetFollow']],kde=False)
#plt.yscale('symlog', linthreshy=0.05)
plt.savefig(plots+"Twitter_followers.pdf")
plt.clf()

#sb.pairplot(data[["TweetCounts", "TweetFollow","isFam"]], hue="isFam")#, size=5)
sb.pairplot(data[["TweetCounts", "TweetFollow"]])
#sb.pairplot(data[["TweetCounts","isFam"]], hue="isFam")
plt.savefig(plots+"Twitter_pairplot.pdf")
plt.clf()

#g = sb.FacetGrid(data[data['isFam']==0], hue = 'isFam')
g = sb.FacetGrid(data, hue = 'isFam')
g = (g.map(plt.hist, "TweetCounts")).add_legend()
plt.savefig(plots+"Twitter_Facet.pdf")
plt.clf()


