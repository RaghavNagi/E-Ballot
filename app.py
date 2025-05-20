from flask import Flask, render_template, session
from routes.register import register_bp
from routes.vote import vote_bp
import os


app = Flask(__name__)

# Set the secret key for session management
app.secret_key = 'my_super_secret_key'  # Generates a random secret key

# Register blueprints
app.register_blueprint(register_bp)
app.register_blueprint(vote_bp)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)