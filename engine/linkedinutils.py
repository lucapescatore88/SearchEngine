from linkedin import linkedin

#KEY = 'wFNJekVpDCJtRPFX812pQsJee-gt0zO4X5XmG6wcfSOSlLocxodAXNMbl0_hw3Vl'
#SECRET = 'daJDa6_8UcnGMw1yuq9TjoO_PMKukXMo8vEMo7Qv5J-G3SPgrAV0FqFCd0TNjQyG'
KEY = "77w8el5fztlakk"
SECRET = "yd7F6IqTIkgoPjZ7"
RETURN_URL = "http://localhost:8080"
auth = linkedin.LinkedInAuthentication(KEY, SECRET, 
    RETURN_URL, linkedin.PERMISSIONS.enums.values())
#app = linkedin.LinkedInApplication(token=auth.get_access_token())
print auth.authorization_url


g = app.get_profile()
print g

prof = app.search_profile(selectors=[{'people': ['Luca', 'Pescatore']}])
print prof

#, params={'keywords': 'apple microsoft'})

