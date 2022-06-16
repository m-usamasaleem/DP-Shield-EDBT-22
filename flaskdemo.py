from app import app, db
from app.dbmodels import User, Update

@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'User': User, 'Update': Update}

