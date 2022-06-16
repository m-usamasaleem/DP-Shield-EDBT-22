from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager

from config import Config

from app.update import Updater
from app.methods import ImageGenerator
from app.clustering import ClusterGenerator


app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login = LoginManager(app)
login.login_view = 'signin'

image_generator = ImageGenerator(app.config['UPLOAD_FOLDER'])
clusterVis_generator = ClusterGenerator(app.config['UPLOAD_FOLDER'])

from app import routes, dbmodels
