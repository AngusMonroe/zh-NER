from flask import Flask
from evaluate import get_ne
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/<query>')
def show_user_profile(query):
    # show the user profile for that user
    return 'Entity %s' % get_ne(query)

if __name__ == '__main__':
    app.debug=True
    app.run(host='0.0.0.0')
