import os
import pusher

basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    SECRET_KEY = 'dfa34jah8fj483nas#gvaSG$Ga2d3df3g'
    UPLOAD_FOLDER = './app/static/uploads/'

    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False