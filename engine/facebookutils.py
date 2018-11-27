import facebook
import requests

token = "EAAKQ57b1k00BADrb1QPDPZC4fekpdX0tk0bQq40e0llnA87Vjb9olbgZCcfRTM2yFZBifgnrwsY9wN4uvxTJ4GCIKhtkkN2re9Sg53O6QvAcBEIOBszFcVWJEGGdi3JoSC0iwsZBosLxAmDYKxQf2zjMa5joZBDG1yJCnPgyIgMBXztY5b4wjM6cGLa8XmZAdMVOTEG3xb6QZDZD"
#token="722274834813773|sVWMk3m9C_XHeWQwUzYui-BBEPM"
graph = facebook.GraphAPI(access_token=token, version="3.1")
#people = graph.search(q='Luca Pescatore',type='post')

#link="https://www.facebook.com/v3.2/dialog/oauth?client_id=722274834813773&redirect_uri=https://127.0.0.1:5000/&state=OK"
#res = requests.get(link)
query = "Luca pescatore"

events = graph.request("/search?q=Poetry&type=event&limit=10000")
print events
data = graph.request('/search?q=someusernane&type=user')
data = graph.request('/search?q=EPFL&type=event&limit=10000')
#data = graph.request('/me?fields=id,name')
print data

#friends = graph.get_connections(id='me', connection_name='friends')
#permissions = graph.get_permissions(user_id=12345)
#print('public_profile' in permissions)