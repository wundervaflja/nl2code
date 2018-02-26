from flask import Flask
import code_gen 

app = Flask(__name__)

app['model'], app['train'] = code_gen.start()


@app.route('/text2code')
def index():
    pass

if __name__ == '__main__':
 	app.run(debug=True)