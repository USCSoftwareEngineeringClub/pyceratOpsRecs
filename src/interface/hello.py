import requests, urllib, httplib, base64
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/search", methods=['POST', 'GET'])
def callAPI():
	error = None
	_url = 'https://api.projectoxford.ai/vision/v1.0/ocr' #https://api.projectoxford.ai/vision/v1.0/ocr[?language][&detectOrientation ]
	_key = 'f8968ffd96d2475cb7ec347c51f24e3e' #Here you have to paste your primary key it is a header
	_maxNumRetries = 10

	bodyURL = request.args.get('uri','')
	print(bodyURL)

	headersIn = {
	    'Content-Type': 'application/json',
	    'Host': 'api.projectoxford.ai',
	    'Ocp-Apim-Subscription-Key': _key,
	}

	paramsIn = urllib.urlencode({
		'language': 'en',
	    'detectOrientation': 'false'
	})

	try:
		conn = httplib.HTTPSConnection('api.projectoxford.ai')
		conn.request("POST", "/vision/v1.0/ocr?%s" % paramsIn, "{body}", headersIn)
		response = conn.getresponse()
		data = response.read()
		print(data)
		conn.close()
		#
		# r = requests.post(_url, data={"url":"https://csgsarchitects.files.wordpress.com/2011/12/111_new-blog.jpg"},\
	 #    			 params=paramsIn, headers=headersIn)
		# print r.content
		# print r.json()
		# return r.text
	    #
		
		#
	    # print 'hello'
	    # conn.request("POST", "/vision/v1.0/ocr?%s" % params, {"url":"http://example.com/images/test.jpg"}, headers)
	    # response = conn.getresponse()
	    # data = response.read()
	    # print(data)
	    # conn.close()
	except Exception as e:
	    print(e)
	    return e



if __name__ == "__main__":
    app.run()
