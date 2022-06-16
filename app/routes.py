import os
from flask import render_template, flash, request, redirect, url_for, session, send_from_directory
from flask_login import current_user, login_user, logout_user, login_required
from werkzeug.utils import secure_filename
from werkzeug.urls import url_parse
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import func
from app import app
from app.forms import LoginForm
from app.dbmodels import User

# from app.update import Updater
from app import image_generator, clusterVis_generator


@app.route('/')
@app.route('/home')
def home():
    if "user" in session:
        user = session["user"]

        return render_template('home.html', title='Mainpage', username=user)
    else:
        return render_template('home.html', title='Mainpage')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if current_user.is_authenticated:
        return redirect(url_for('account'))

    if request.method == "POST":
        # insecure method of handling user data. For this application maximum security does not seem necessary
        # but if it becomes necessary, use something like Flask-WTForums
        username = request.form['username'].lower()

        # user = User.query.filter_by(username=username).first()
        user = User.query.filter(func.lower(User.username) == func.lower(username)).first()
        if user is None or not user.check_password(request.form['password']):
            # return render_template('signin.html')
            return render_template('signin.html', error="Invalid Login")
        else:
            login_user(user, remember=True)
        
            # Code to redirect back to the page where a login was required
            next_page = request.args.get('next')
            if not next_page or url_parse(next_page).netloc != '':
                next_page = url_for('signin')
            return redirect(next_page)

        return redirect(url_for("home"))
    else:
        return render_template('signin.html')

@app.route('/logout')
def logout():
    logout_user()

    return redirect(url_for("home"))

@app.route('/account')
@login_required
def account():
    return render_template('account.html')

@app.route('/methods', methods=['GET', 'POST'])
@app.route('/methods')
def methods():
    if request.method == 'POST':
        params = request.json

        image_generator.clean_upload_folder()

        if params['method'] == 'dpsamp':
            save_time = image_generator.gen_dpsamp_image(params)
        elif params['method'] == 'dppix':
            save_time = image_generator.gen_dppix_image(params)
        elif params['method'] == 'dpsvd':
            save_time = image_generator.gen_dpsvd_image(params)
        elif params['method'] == 'snow':
            save_time = image_generator.gen_snow_image(params)

        return str(save_time)

    return render_template('methods.html')

@app.route('/clustering', methods=['GET', 'POST'])
@app.route('/clustering')
def clustering():
    if request.method == 'POST':
        params = request.json

        clusterVis_generator.clean_upload_folder()

        clusterVis_generator.generate_private_images(params)
        save_time = clusterVis_generator.generate_clustering_vis2(params)

        return str(save_time)

    return render_template('clustering.html')

@app.route('/evaluations')
def evaluations():
    return render_template('evaluations.html')

@app.route('/qualitative')
def qualitative():
    return render_template('qualitative.html')

# @app.route('/update', methods=['GET', 'POST'])
# @login_required
# def update(): 
#     if request.method == 'POST':
#         a_file = request.files['awdupload']
#         p_file = request.files['propupload']

#         if a_file.filename == '' or p_file.filename == '':
#             return 'Please upload both files. One or more files missing.'

#         # create the file paths
#         awd_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(a_file.filename))
#         prop_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(p_file.filename))

#         # save the files
#         a_file.save(awd_path)
#         p_file.save(prop_path)

#         # perform csv checks here
#         main_updater.set_paths(awd_path, prop_path)
#         results = main_updater.check_awd_and_prop()
            
#         return results[1]

#     return render_template('update.html')

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

@app.route('/display/<filename>')
def display_image(filename):
	print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

# @app.route('/findnewgrants', methods=['POST'])
# def findnewgrants():
#     if request.method == 'POST':
#         ret = main_updater.find_new_grants() # DEBUG
#         try:
#             # find_new_grants returns a tuple in the format:    (# new awards found, # new proposals found)
#             # ret = main_updater.find_new_grants()
#             print(ret) # DEBUG

#             retStr = f'Searching finished... Found {ret[0]} new awards and {ret[1]} new proposals in the uploaded sheets'

#             return retStr
#         except Exception as e:
#             print(str(e))
#             return str(e)
