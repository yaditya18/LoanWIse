from __future__ import division
from flask import Flask, render_template, url_for, request, redirect, session, flash, g
from predictor import call_model


app=Flask(__name__)

app.secret_key = '2'

@app.route('/', methods = ['GET'])
def home_page():
        return render_template('home_page.html', message='hello')


@app.route('/data_input', methods=['GET', 'POST'])
def data_input():
       if request.method=='GET':
            return render_template('data_input_page.html')
       elif request.method=='POST':
            amount=request.form['amount']
            min_balance=request.form['min_balance']
            five_k=request.form['five_k']
            data=[float(amount), float(min_balance), float(five_k)]
            value=call_model(data)
            if value[0]==0:
                  message='You are not applicable for the loan.'
            elif value[0]==1:
                  message='You are applicable for the loan.'
            else:
                  message='Not rendering'
            # message='Please type something'
            return render_template('predicton_page.html', message=message)


if __name__ == '__main__':
    app.run()