from flask import Flask, render_template, request, flash, redirect, url_for, session
import os
import PerfumeRecommender as pir

app = Flask(__name__)

@app.route('/search', methods=['GET', 'POST'])
def search_page():
	if request.method == 'POST':
		query_string = request.form['search_query']
		num_rec = request.form['range']
		print(query_string)
		print(num_rec)
		pir.load_models('final_perfume_data.csv', 'models')
		recommended_perfumes = pir.find_similar_perfumes(query_string, num_rec)
		print(recommended_perfumes)
		perfume_details = pir.details_of_recommendations(recommended_perfumes)
	return render_template('search.html')

@app.route('/home')
def homepage():
   return render_template('home.html')

@app.route('/')
def redirect_to_homepage():
	return redirect('/home')


if __name__ == '__main__':
    app.run(debug=True)