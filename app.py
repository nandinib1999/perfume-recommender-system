from flask import Flask, render_template, request, flash, redirect, url_for, session
import os

app = Flask(__name__)

@app.route('/search')
def search_page():
	return render_template('search.html')

@app.route('/home')
def homepage():
   return render_template('home.html')

@app.route('/')
def redirect_to_homepage():
	return redirect('/home')


if __name__ == '__main__':
    app.run(debug=True)