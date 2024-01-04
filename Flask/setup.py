from flask import Flask


app = Flask(__name__)


def hello_world():
    return "hello, World"

@app.route("/")
def test():
    a=5+5
    return "this is my test function{}".format(a)

def request_input():
    data = request.args.get("x")
    


# @app.route("/<name>")
# def name(name):
#     return f"Hello, {name}"

# @app.route("/admin")
# def admin():
#     return "Hello, Admin"

if __name__=="__main__":
    app.run(host="0.0.0.0")