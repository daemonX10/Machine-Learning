from flask import Flask, request , render_template,jsonify

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route('/math', methods=['POST'])
def math_ops():
    import math
    if (request.method=="POST"):
        ops = request.form['operation']
        num1=int(request.form['num1'])
        num2=int(request.form['num2'])

        if ops=='add':
            r=num1+num2
            result="The sum of " +  str(num1) +  ' add ' + str(num2) + " is " + str(r)
            return render_template('results.html', result=result)
        
        if ops=='subtract':
            r=num1-num2
            result="The sum of " +  str(num1) +  ' add ' + str(num2) + " is " + str(r)
            return render_template('results.html', result=result)
        
        if ops=='multiply':
            r=num1*num2
            result="The sum of " +  str(num1) +  ' add ' + str(num2) + " is " + str(r)
            return render_template('results.html', result=result)
        
        if ops=='divide':
            r=num1/num2
            result="The sum of " +  str(num1) +  ' add ' + str(num2) + " is " + str(r)
            return render_template('results.html', result=result)
    
        if ops=='log':
            r= math.log(num1)
            r2=math.log(num2)
            result="The log of " +  str(num1) +  " is " + str(r) +' and log of ' + str(num2) + " is " + str(r2)
            return render_template('results.html', result=result)
    
        

if __name__ == "__main__":
    app.run(host="0.0.0.0")