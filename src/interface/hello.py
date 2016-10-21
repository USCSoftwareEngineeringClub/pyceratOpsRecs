import httplib, urllib, base64
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/search/", methods=['POST', 'GET'])
def callAPI():
	error = None
	_url = 'https://api.projectoxford.ai/vision/v1/ocr' #https://api.projectoxford.ai/vision/v1.0/ocr[?language][&detectOrientation ]
	_key = 'f8968ffd96d2475cb7ec347c51f24e3e' #Here you have to paste your primary key it is a header
	_maxNumRetries = 10

	headers = {
	    # Request headers
	    'Content-Type': 'application/json',
	    'Ocp-Apim-Subscription-Key': _key,
	}

	params = urllib.urlencode({
	    # Request parameters
	    'detectOrientation ': 'true',
	})

	try:
	    conn = httplib.HTTPSConnection('api.projectoxford.ai')
	    conn.request("POST", "/vision/v1.0/ocr?%s" % params, "{body}", headers)
	    response = conn.getresponse()
	    data = response.read()
	    print(data)
	    conn.close()
	except Exception as e:
	    print("[Errno {0}] {1}".format(e.errno, e.strerror))



if __name__ == "__main__":
    app.run()
