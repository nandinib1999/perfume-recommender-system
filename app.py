from flask import Flask, render_template, request, redirect, jsonify
import os
import PerfumeRecommender as pir
import json

app = Flask(__name__)

@app.route('/display', methods=['GET', 'POST'])
def get_recommendations():
	if request.method == "POST":
		query =  request.form['search_query'];
		num = request.form['range'];
		pir.load_models('final_perfume_data.csv', 'models')
		recommended_perfumes = pir.find_similar_perfumes(query, num)
		print(recommended_perfumes)
		perfume_details = pir.details_of_recommendations(recommended_perfumes)
		return json.dumps({'status':'OK', 'data':perfume_details});

@app.route('/search', methods=['GET', 'POST'])
def search_page():
	return render_template('search.html', recommendations=None)

@app.route('/home')
def homepage():
   return render_template('home.html')

@app.route('/')
def redirect_to_homepage():
	return redirect('/home')


if __name__ == '__main__':
    app.run(debug=True)