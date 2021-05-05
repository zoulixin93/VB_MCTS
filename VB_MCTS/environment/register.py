import requests
import json



def Register(email,institution,names,baseuri):
    registerUrl = '%s/api/users/v0/register'%baseuri;
    registerInfo = json.dumps({"email": email, "institution": institution, "names": names});
    response = None
    try:
        response = requests.post(registerUrl, data = registerInfo, headers = {'Content-Type': 'application/json', 'Accept': 'application/json'});
        data = response.json()
        userID = data['userId']
    except Exception as e:
        if response is not None:
            print(response)
        raise e;

    return userID
