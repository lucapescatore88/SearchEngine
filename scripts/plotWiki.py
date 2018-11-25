from engineutils import root, resroot
import matplotlib.pyplot as plt
import seaborn as sb
import pickle
import sys

plots = root+"/plots/"

data = pickle.load(open(resroot+"WikiDF.pkl"))

print "Some famous people"
print data[data['isFam']==1].head(5)
print "Some non-famous people"
print data[data['isFam']==0].head(5)
print data.describe()

sys.exit()
sb.relplot(x="TweetCounts", y="TweetFollow", hue = "isFam", data=data);
plt.savefig(plots+"Twitter_hits_vs_follows.pdf")
plt.clf()

#teetcounts = data['TweetCounts'].toarray()
sb.distplot(data.TweetCounts.dropna(),kde=False)
plt.savefig(plots+"Twitter_hits.pdf")
plt.clf()

ax = sb.distplot(data.TweetFollow.dropna(),kde=False)
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