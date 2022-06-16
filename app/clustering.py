from sqlite3 import Timestamp
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from app.privacy_methods import samplingdp, pixelation, svd, snow
import cv2
import os
import re
from scipy.spatial import distance
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from facenet_pytorch import MTCNN, InceptionResnetV1, training, fixed_image_standardization
from facenet_pytorch import InceptionResnetV1
import glob
import time
import tqdm 
import numpy as np
import pickle
import matplotlib.lines as mlines

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class ClusterGenerator():
    def __init__(self, upload_folder):
        self.save_path = upload_folder
        self.im_folder = './app/static/img'
        self.dataset_faces_folder = '.app/static/dataset_faces'

        self.flatui = ["red", "gold", "cyan", "chartreuse", "magenta"]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.standard_transform = transforms.Compose([
            np.float32, 
            transforms.ToTensor(),
            fixed_image_standardization
        ])

        print('Initializing models for clustering visualization...')
        print('\t-LDA VGG Faces2')
        self.lda_vggface2 = LinearDiscriminantAnalysis(n_components=2)
        model_path = './app/static/classification_models/lda_vggface2.pkl'
        with open(model_path, 'rb') as f:
            self.lda_vggface2 = pickle.load(f)

        print('\t-LDA CASIA Webface')
        self.lda_casia = LinearDiscriminantAnalysis(n_components=2)
        model_path = './app/static/classification_models/lda_casia.pkl'
        with open(model_path, 'rb') as f:
            self.lda_casia = pickle.load(f)

        print('\t-SVC VGG Faces2')
        self.clf_vggfaces2 = SVC(kernel='rbf', probability=True)
        model_path = './app/static/classification_models/vggface2.pkl'
        with open(model_path, 'rb') as f:
            self.clf_vggfaces2 = pickle.load(f)

        print('\t-SVC CASIA Webface')
        self.clf_casiaweb = SVC(kernel='rbf', probability=True)
        model_path = './app/static/classification_models/casia-webface.pkl'
        with open(model_path, 'rb') as f:
            self.clf_casiaweb = pickle.load(f)

    def clean_upload_folder(self, time_in_seconds=3):
        deleted_files = 0
        cur_time = time.time()

        for fname in os.listdir(self.save_path):
            x = int(re.findall("\d+", fname)[0])

            if cur_time-x >= 3:
                os.remove(f'{self.save_path}/{fname}')
                deleted_files += 1
                # print(f'{self.save_path}/{fname}')

            print(f'Deleted {deleted_files} files in upload folder cleanup')

    def get_files(self, path='./', ext=('.png', '.jpeg', '.jpg')):
        """ Get all image files """
        files = []

        for e in ext:
            files.extend(glob.glob(f'{path}/**/*{e}'))

        files.sort(key=lambda p: (os.path.dirname(p), int(os.path.basename(p).split('.')[0])))

        return np.array(files)

    def generate_private_images(self, params):
        print('Generating images...')
        dataset_path = 'visual2_vggface2LDA' if params['dataset_name']=='vggface' else 'visual2_casia-webfaceLDA'
        images = self.get_files(f'./app/static/dataset_faces/{dataset_path}/clean')

        for im_path in images:
            if params['method'] == 'dpsamp':
                epsilon = float(params['epsilon'])
                kval = int(params['cluster_sz'])
                mval = int(params['mval'])

                _, sampled_im, private_im = samplingdp.run_samplingdp_on_image(im_path, k=kval, epsilon=epsilon, m_param=mval, verbose=False)

            elif params['method'] == 'dppix':
                epsilon = float(params['epsilon'])
                bval = int(params['block_sz'])
                mval = int(params['mval'])

                input_image = cv2.imread(im_path)
                private_im, median, blur = pixelation.pixelation(im_path, is_rgb=False, m=mval, epsilon=epsilon, b = bval, delta_p = 255)

                private_im = cv2.resize(private_im, (input_image.shape[1], input_image.shape[0]), interpolation=cv2.INTER_NEAREST)

            elif params['method'] == 'dpsvd':
                epsilon = float(params['epsilon'])
                ival = int(params['eigen_sz'])

                private_im, priv_im_median, no_priv = svd.svd(im_path, ival, epsilon)

            elif params['method'] == 'snow':
                delta = float(params['delta'])
                pval = 1-delta

                private_im = snow.apply_snow(im_path, p=pval)

            save_im_path = im_path.replace('clean', 'noisy')

            cv2.imwrite(save_im_path, private_im)
    
    def getEmbeds(self, model, n, loader):
        model.eval()
        X_test = []
        embeds, labels = [], []
        for i, (x, y) in enumerate(loader, 1): 
                embed = model(x.to(self.device))
                embed = embed.data.cpu().numpy()
                X_test.append(embed[0])
        return X_test

    def evaluation(self, path, modelpath, lda, subdir=""):
        ALIGNED_TEST_DIR = os.path.join(path, subdir)

        testF = self.get_files(ALIGNED_TEST_DIR)[:]

        testD = datasets.ImageFolder(ALIGNED_TEST_DIR, transform=self.standard_transform)
        testL = DataLoader(testD, batch_size=1, num_workers=2)

        model = InceptionResnetV1(pretrained=modelpath, classify=False, dropout_prob=0.6, device=self.device).eval()

        X_test=self.getEmbeds(model, 1, testL)
        Y_test=[]

        for y in testF:
            if not subdir:
                Y_test.append(np.int64(y.split("\\")[1])) # NOTE: May need to change the path depending on the system OS
            else:
                Y_test.append(np.int64(y.split("\\")[2])) # NOTE: May need to change the path depending on the system OS
        
        X_test=np.array(X_test)
        Y_test=np.array(Y_test)

        # loading SVC classifier
        if modelpath=="vggface2":
            clf = self.clf_vggfaces2
        elif modelpath=="casia-webface":
            clf = self.clf_casiaweb  
        else:
            print("Error in loading ")

        pred = clf.predict(X_test)
        
        return X_test, Y_test, pred

    def visualization2(self, vis2_clean_xtest,vis2_clean_ytest,vis2_clean_Labels,vis2_noisy_xtest,vis2_noisy_ytest,vis2_noisy_Labels,dpmethod,lda_model,total_entities,dataset):
        # transforming embeddings to 2d
        vis2_clean = lda_model.transform(vis2_clean_xtest)
        vis2_noisy = lda_model.transform(vis2_noisy_xtest)
        
        fig, ax = plt.subplots(1,2,figsize=(20, 8))
        
        legends=[]
        for i in range(total_entities):
            # extracting same entity 2d embeds
            xaxis,yaxis=vis2_clean[(i*5):(i +1)*5, 0],vis2_clean[(i*5):(i +1)*5, 1]
            
            # plotting same entities
            img = ax[0].scatter(xaxis, yaxis, c=self.flatui[i%5], alpha=1,  marker='o',s=80 )
            
            # adding legend
            legends.append(mlines.Line2D([], [], color=self.flatui[i%5], marker='o', ls='', markersize =15,label=vis2_clean_ytest[(i*5):(i +1)*5][0]))
        # title
        # ax[0].set_title('Orignal in 2d space (Dim Reduction using LDA)' , fontsize=16)
        ax[0].set_xlabel('x', fontsize=25)
        ax[0].set_ylabel('y', fontsize=25)
        ax[0].tick_params(axis='both', labelsize=18)
        
        # caption
        # textstr = '\n'.join(( dataset+ ' clean samples',"Method : "+dpmethod,'o : Orignal Face', "x : Private Face"))
        # ax[0].text(0.35, 1.1, textstr, transform=ax[0].transAxes, fontsize=17,bbox=dict(boxstyle='round', facecolor='w', alpha=0.9))

        #plotting legends
        leg = ax[0].legend(handles=legends,fontsize=20,labelcolor=self.flatui)
        
        # plot1
        legends=[]
        for i in range(total_entities):
            # extracting same entity 2d embeds
            xaxis,yaxis=vis2_noisy[(i*5):(i +1)*5, 0],vis2_noisy[(i*5):(i +1)*5, 1]
            # plotting same entities
            img = ax[1].scatter(xaxis, yaxis, c=self.flatui[i%5], alpha=1,  marker='x',s=80 )
            
            # adding legend
            legends.append(mlines.Line2D([], [], color=self.flatui[i%5], marker='x', ls='', markersize =15,label=vis2_noisy_ytest[(i*5):(i +1)*5][0]))
        
        # title
        # ax[1].set_title('Orignal in 2d space (Dim Reduction using LDA)' , fontsize=16)
        ax[1].set_xlabel('x', fontsize=25)
        ax[1].set_ylabel('y', fontsize=25)
        ax[1].tick_params(axis='both', labelsize=18)
        
        # caption
        # textstr = '\n'.join(( dataset+ ' noisy samples',"Method : "+dpmethod,'o : Orignal Face', "x : Private Face"))
        # ax[1].text(0.35, 1.1, textstr, transform=ax[1].transAxes, fontsize=17, bbox=dict(boxstyle='round', facecolor='w', alpha=0.9))
        
        #plotting legends
        leg = ax[1].legend(handles=legends,fontsize=20,labelcolor=self.flatui)

        timestamp = int(time.time())
        
        fig.savefig(f'./app/static/uploads/{timestamp}vis2.png', bbox_inches='tight')

        return timestamp

    def generate_clustering_vis2(self, params):
        dataset_name = 'vggface2' if params['dataset_name']=='vggface' else 'casia-webface'
        dataset_id = 'visual2_vggface2LDA' if params['dataset_name']=='vggface' else 'visual2_casia-webfaceLDA'
        dataset_path = f'./app/static/dataset_faces/{dataset_id}'

        vis2_clean_xtest,vis2_clean_ytest,vis2_clean_Labels=self.evaluation(dataset_path, dataset_name, self.lda_vggface2, "clean")
        vis2_noisy_xtest,vis2_noisy_ytest,vis2_noisy_Labels=self.evaluation(dataset_path, dataset_name, self.lda_vggface2, "noisy")

        total_entities = 5
        dpmethod = params['method']
        dataset = 'VGG Faces2' if params['dataset_name']=='vggface' else 'CASIA Faces'

        timestamp = self.visualization2(vis2_clean_xtest, vis2_clean_ytest, vis2_clean_Labels, vis2_noisy_xtest, 
            vis2_noisy_ytest, vis2_noisy_Labels, dpmethod, self.lda_vggface2, total_entities, dataset
        )

        return timestamp