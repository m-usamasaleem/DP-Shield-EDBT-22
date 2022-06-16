# Image privacy demo

## Step 1: Create conda environment
1. Navigate to the cloned repository
2. Create a conda environment with "conda create --name ENV_NAME --python=3.6 --file requirements.txt"

## Step 2: Add folders & files that were excluded from repository (due to large size)
1. Create folder ./app/static/uploads
2. Classification models
    - ./app/static/classification_models/casia_webface.pkl
    - ./app/static/classification_models/vggface2.pkl
    - ./app/static/classification_models/lda_casia.pkl
    - ./app/static/classification_models/lda_vggface2.pkl
3. Dataset images (to display during privacy attack)
    - Example path for CASIA: ./app/static/dataset_faces/casia/1...
    - Example path for vggface2: ./app/static/dataset_faces/vggface2/1...
    - Example path for vggface2 LDA: ./app/static/dataset_faces/visual2_casia-webfaceLDA/clean/74/..
    - Example path for vggface2 LDA: ./app/static/dataset_faces/visual2_vggface2LDA/clean/17/..

## Step 3: Run Flask server
* (optional) set flask into development mode. This way changes to the code will update the site automatically
  * Command on Windows: "set FLASK_ENV=development"
  * Command on Linux: "export FLASK_ENV=development"
* Run command "flask run"
* Navigate to the link printed in the console
