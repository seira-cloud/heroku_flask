'''
Controller file for the web appilaction

The central file of the application
'''

from flask import Flask
from flask import render_template
from flask import request
import pandas as pd
from movie_rec_app.recommender import model_recommender, user_recommendation



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', title='Awesome Movie Recommender')

@app.route('/recommender')
def recommender():
    html_form_data = dict(request.args)
    print(html_form_data)

    df = pd.read_csv('movie_rec_app/user_item_matrix.csv')
    recs= user_recommendation(html_form_data, model_recommender(df))
    
    #recs = get_recommendations()

    return render_template('recommendations.html',
                            movies = recs)

if __name__ == "__main__": 
    app.run(debug=True, port=5500) 

