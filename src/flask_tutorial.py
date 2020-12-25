import sys
sys.path.append('..')

from flask import Flask, render_template, url_for
from .forms import RegistrationForm, LoginForm

app = Flask(__name__)  # just the name of the module
app.config['SECRET_KEY'] = 'f4ec120c2b53787b212683b6b81e6f84'

posts = [
    {
        'author': 'Corey Schafer',
        'title': 'Blog Post 1',
        'content': 'First post content',
        'date_posted': 'April 20, 2018'
    },
    {
        'author': 'Simon Dagenais',
        'title': 'Blog Post 2',
        'content': 'Second post content',
        'date_posted': 'Dec 25, 2020'
    }

]


@app.route('/')  # both paths lead to home page
@app.route('/home')
def home():
    return render_template("home.html", posts=posts)


@app.route('/about')
@app.route('/About')
def about():
    return render_template("about.html", title='About')


@app.route('/register')
def register():
    form = RegistrationForm()
    return render_template('register.html', title='Register', form=form)

@app.route('/login')
def register():
    form = LoginForm()
    return render_template('login.html', title='Login', form=form)


if __name__ == '__main__':
    app.run(debug=True)
