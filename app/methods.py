from numpy.lib.function_base import delete
from app.privacy_methods import samplingdp, pixelation, svd, snow
from skimage.metrics import structural_similarity as compute_ssim
import torch
import glob
import time
import cv2
import os
import re

# For privacy attacks
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from sklearn.svm import SVC
import numpy as np
import torchvision
import pickle
import json

class ImageGenerator():
    def __init__(self, upload_folder):
        self.save_path = upload_folder
        self.im_folder = './app/static/img'
        self.dataset_faces_folder = '.app/static/dataset_faces'

        print('Initializing classification models for privacy attacks...')
        print('\t-CASIA faces...')
        self.casia_cnn = InceptionResnetV1(pretrained='casia-webface', classify=False).eval()
        print('\t-VGGFace2...')
        self.vggface_cnn = InceptionResnetV1(pretrained='vggface2', classify=False).eval()
        print('\t-SVC Models...')
        self.casia_clf = SVC(kernel='rbf', probability=True)
        self.vgg_clf = SVC(kernel='rbf', probability=True)

        model_path = './app/static/classification_models/casia-webface.pkl'
        with open(model_path, 'rb') as f:
            self.casia_clf = pickle.load(f)

        model_path = './app/static/classification_models/vggface2.pkl'
        with open(model_path, 'rb') as f:
            self.vgg_clf = pickle.load(f)
        print(f'Finished loading classification models for privacy attacks')

        print(f'Uploading to {self.save_path}')

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
    
    def calculate_mse_rmse(self, im1, im2):
        assert im1.shape == im2.shape, 'Image shapes are not equal. Cannot compute MSE'
        n_pixels = im1.shape[0]*im1.shape[1]

        sse = np.sum(((im1.astype(np.uint16)-im2)**2))
        mse = sse/n_pixels
        rmse = np.sqrt(mse)

        return (mse, rmse)

    def gen_classifier_results(self, im_path, dataset, input_im_path):
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = np.moveaxis(im, -1, 0)
        im = torch.tensor(im, dtype=torch.float32)
        im = fixed_image_standardization(im)
        im = im.unsqueeze(0)

        inim = cv2.imread(input_im_path)
        inim = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB)
        inim = np.moveaxis(inim, -1, 0)
        inim = torch.tensor(inim, dtype=torch.float32)
        inim = fixed_image_standardization(inim)
        inim = inim.unsqueeze(0)

        if dataset=='casia':
            embedding = self.casia_cnn(im).detach().numpy()
            embedding2 = self.casia_cnn(inim).detach().numpy()
            clf = self.casia_clf
        else:
            embedding = self.vggface_cnn(im).detach().numpy()
            embedding2 = self.vggface_cnn(inim).detach().numpy()
            clf = self.vgg_clf

        # Private Image
        probs = clf.predict_proba(embedding)
        top_5_idxs = np.argsort(probs, axis=1, )[:, -5:][0]
        top_5_ids = np.flip(clf.classes_[top_5_idxs])
        top_5_proba = np.flip(probs[0][top_5_idxs])
        # Make the ids/probabilities serializable by JSON
        top_5_ids = [int(e) for e in top_5_ids]
        top_5_proba = [float(e) for e in top_5_proba]

        # Input Image
        probs = clf.predict_proba(embedding2)
        top_5_idxs_input = np.argsort(probs, axis=1, )[:, -5:][0]
        top_5_ids_input = np.flip(clf.classes_[top_5_idxs_input])
        top_5_proba_input = np.flip(probs[0][top_5_idxs_input])
        # Make the ids/probabilities serializable by JSON
        top_5_ids_input = [int(e) for e in top_5_ids_input]
        top_5_proba_input = [float(e) for e in top_5_proba_input]


        return [top_5_ids, top_5_proba, top_5_ids_input, top_5_proba_input]

    def get_id_image(self, id, dataset, gt_id):
        if id==gt_id:
            return f'../static/img/{dataset}_id_{id}.png'

        path = glob.glob(f'./app/static/dataset_faces/{dataset}/{id}/*.png')[0]
        path = os.path.normpath(path)
        split_path = path.split(os.sep)
        path = f'{split_path[-4]}/{split_path[-3]}/{split_path[-2]}/{split_path[-1]}'

        return path

    def gen_dpsamp_image(self, params):
        imname = os.path.split(params['input_image_path'])[-1]
        dataset = imname[:-4].split('_')[0]
        gt_id = imname[:-4].split('_')[-1]
        image_path = f'{self.im_folder}/{imname}'
        assert os.path.exists(image_path), 'Image path does not exist!'

        epsilon = float(params['epsilon'])
        kval = int(params['cluster_sz'])
        mval = int(params['mval'])

        # Load input image
        input_image = cv2.imread(image_path)
        if len(input_image.shape) == 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        _, sampled_im, private_im = samplingdp.run_samplingdp_on_image(image_path, k=kval, epsilon=epsilon, m_param=mval, verbose=False)

        mse, rmse = self.calculate_mse_rmse(input_image, private_im)
        ssim = compute_ssim(input_image, private_im)

        save_time = int(time.time())
        cv2.imwrite(f'{self.save_path}/samp_sampledim_{save_time}.jpg', sampled_im)
        cv2.imwrite(f'{self.save_path}/samp_private_{save_time}.jpg', private_im)
        
        ids, probs, inp_ids, inp_probs = self.gen_classifier_results(f'{self.save_path}/samp_private_{save_time}.jpg', dataset, image_path)

        id_im_paths = [self.get_id_image(e, dataset, gt_id) for e in ids]
        inp_id_im_paths = [self.get_id_image(e, dataset, gt_id) for e in inp_ids]

        return_dict = {
            'save_time': save_time,
            'groundtruth_id': gt_id,
            'dataset': dataset,
            'privMSE': mse,
            'privSSIM': ssim,
            'top_5_ids': ids,
            'top_5_probabilities': probs,
            'top_5_impaths': id_im_paths,
            'top_5_probabilities_input': inp_probs,
            'top_5_impaths_input': inp_id_im_paths,
            'top_5_ids_input': inp_ids
        }
        return json.dumps(return_dict)


    def gen_dppix_image(self, params):
        imname = os.path.split(params['input_image_path'])[-1]
        dataset = imname[:-4].split('_')[0]
        gt_id = imname[:-4].split('_')[-1]
        image_path = f'{self.im_folder}/{imname}'
        assert os.path.exists(image_path), 'Image path does not exist!'

        # Load input image
        input_image = cv2.imread(image_path)
        if len(input_image.shape) == 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        epsilon = float(params['epsilon'])
        bval = int(params['block_sz'])
        mval = int(params['mval'])

        blur_dp, median, blur = pixelation.pixelation(image_path, is_rgb=False, m=mval, epsilon=epsilon, b = bval, delta_p = 255)

        blur_dp = cv2.resize(blur_dp, (input_image.shape[1], input_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        median = cv2.resize(median, (input_image.shape[1], input_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        blur = cv2.resize(blur, (input_image.shape[1], input_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        mse, rmse = self.calculate_mse_rmse(input_image, blur_dp)
        ssim = compute_ssim(input_image, blur_dp)

        save_time = int(time.time())
        cv2.imwrite(f'{self.save_path}/pix_blur_dp_{save_time}.jpg', blur_dp)
        cv2.imwrite(f'{self.save_path}/pix_median_blur_dp_{save_time}.jpg', median)
        cv2.imwrite(f'{self.save_path}/pix_blur_nodp_{save_time}.jpg', blur)

        ids, probs, inp_ids, inp_probs = self.gen_classifier_results(f'{self.save_path}/pix_blur_dp_{save_time}.jpg', dataset, image_path)

        id_im_paths = [self.get_id_image(e, dataset, gt_id) for e in ids]
        inp_id_im_paths = [self.get_id_image(e, dataset, gt_id) for e in inp_ids]

        return_dict = {
            'save_time': save_time,
            'groundtruth_id': gt_id,
            'dataset': dataset,
            'privMSE': mse,
            'privSSIM': ssim,
            'top_5_ids': ids,
            'top_5_probabilities': probs,
            'top_5_impaths': id_im_paths,
            'top_5_probabilities_input': inp_probs,
            'top_5_impaths_input': inp_id_im_paths,
            'top_5_ids_input': inp_ids
        }
        return json.dumps(return_dict)

    def gen_dpsvd_image(self, params):
        imname = os.path.split(params['input_image_path'])[-1]
        dataset = imname[:-4].split('_')[0]
        gt_id = imname[:-4].split('_')[-1]
        image_path = f'{self.im_folder}/{imname}'
        assert os.path.exists(image_path), 'Image path does not exist!'

        # Load input image
        input_image = cv2.imread(image_path)
        if len(input_image.shape) == 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        epsilon = float(params['epsilon'])
        ival = int(params['eigen_sz'])

        priv_im, priv_im_median, no_priv = svd.svd(image_path, ival, epsilon)

        save_time = int(time.time())
        cv2.imwrite(f'{self.save_path}/svd_priv_im_{save_time}.jpg', priv_im)
        cv2.imwrite(f'{self.save_path}/svd_priv_im_median_{save_time}.jpg', priv_im_median)
        cv2.imwrite(f'{self.save_path}/svd_nopriv_{save_time}.jpg', no_priv)

        mse, rmse = self.calculate_mse_rmse(input_image, priv_im)
        ssim = compute_ssim(input_image, priv_im)

        ids, probs, inp_ids, inp_probs = self.gen_classifier_results(f'{self.save_path}/svd_priv_im_{save_time}.jpg', dataset, image_path)

        id_im_paths = [self.get_id_image(e, dataset, gt_id) for e in ids]
        inp_id_im_paths = [self.get_id_image(e, dataset, gt_id) for e in inp_ids]

        return_dict = {
            'save_time': save_time,
            'groundtruth_id': gt_id,
            'dataset': dataset,
            'privMSE': mse,
            'privSSIM': ssim,
            'top_5_ids': ids,
            'top_5_probabilities': probs,
            'top_5_impaths': id_im_paths,
            'top_5_probabilities_input': inp_probs,
            'top_5_impaths_input': inp_id_im_paths,
            'top_5_ids_input': inp_ids
        }
        return json.dumps(return_dict)


    def gen_snow_image(self, params):
        imname = os.path.split(params['input_image_path'])[-1]
        dataset = imname[:-4].split('_')[0]
        gt_id = imname[:-4].split('_')[-1]
        image_path = f'{self.im_folder}/{imname}'
        assert os.path.exists(image_path), 'Image path does not exist!'

        # Load input image
        input_image = cv2.imread(image_path)
        if len(input_image.shape) == 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        delta = float(params['delta'])
        pval = 1-delta

        priv_im = snow.apply_snow(image_path, p=pval)

        mse, rmse = self.calculate_mse_rmse(input_image, priv_im)
        ssim = compute_ssim(input_image, priv_im)

        save_time = int(time.time())
        cv2.imwrite(f'{self.save_path}/snow_priv_{save_time}.jpg', priv_im)

        ids, probs, inp_ids, inp_probs = self.gen_classifier_results(f'{self.save_path}/snow_priv_{save_time}.jpg', dataset, image_path)

        id_im_paths = [self.get_id_image(e, dataset, gt_id) for e in ids]
        inp_id_im_paths = [self.get_id_image(e, dataset, gt_id) for e in inp_ids]

        return_dict = {
            'save_time': save_time,
            'groundtruth_id': gt_id,
            'dataset': dataset,
            'privMSE': mse,
            'privSSIM': ssim,
            'top_5_ids': ids,
            'top_5_probabilities': probs,
            'top_5_impaths': id_im_paths,
            'top_5_probabilities_input': inp_probs,
            'top_5_impaths_input': inp_id_im_paths,
            'top_5_ids_input': inp_ids
        }
        return json.dumps(return_dict)