# Self-Supervised Viewpoint Learning from Image Collections

This repository contains code for our work on Self-Supervised Viewpoint Learning from Image Collections (SSV) accepted at CVPR 2020. 
SSV provides a framework to learn viewpoint estimation of objects just using images of objects without the need for groundtruth viewpoint annotations.

![ssv](utils/ssv_small.gif)

## Links
* [PDF](https://research.nvidia.com/sites/default/files/pubs/2020-03_Self-Supervised-Viewpoint-Learning/SSV-CVPR2020.pdf) 
* [Arxiv](http://arxiv.org/abs/2004.01793) 
* [NVIDIA Project Page](https://research.nvidia.com/publication/2020-03_Self-Supervised-Viewpoint-Learning)  


## Prerequisites
We used Pytorch 1.0 with CUDA 10 and CuDNN 7.4.1 in Ubuntu 16.04.
All the dependencies are provided in requirements.txt
A similar environment can be created using:  
`conda create --name ssv --file requirements.txt`

Please download MTCNN-Pytorch from [here](https://github.com/TropComplique/mtcnn-pytorch) and install it in 'data_preprocessing' folder. This is required for preprocessing the datasets.

## Datasets
### 300W-LP
300W-LP dataset can be downloaded from [here](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm). To preprocess it run:  
`python preprocess_data.py --src-dir <path_to_300wlp_dataset> --dst-dir <path_to_processed_300wlp> --datset 300WLP`

Create an lmdb of the preprocessed 300W-LP data by running:  
`python  prepare_lmdb.py <path_to_processed_300wlp> --out <path_to_300wlp_lmdb>`


### BIWI Head Pose
BIWI headpose estimation dataset can by downloaded by writing to the authors of 'Random Forests for Real Time 3D Face Analysis', Fanelli et al, IJCV 13. To preprocess it run:  
`python preprocess_data.py --src-dir <path_to_biwi_dataset> --dst-dir <path_to_processed_biwi> --datset BIWI`


## Test
To test SSV download the pretrained models from [here](link.to.models).
Run the following:   
`python test_vpnet.py --data_dir <path_to_processed_biwi> --model_path <path_to_pretrained_model> `

## Training
To train SSV from scratch, run the following:  
`python3 train.py --exp_name SSV --data_path <path_to_300wlp_lmdb> --num_workers 4 --exp_root  <path_to_experiments_dir>  --save_interval 5000 --sample_interval 500 --batch_size 2 --lr 0.0005 --code_size 64 --z_recn_weight 1.0 --vp_recn_weight 1.0 --img_recn_weight 0.6 --flip_cons_weight 0.6 --flipc_recn_weight_G 0.5 --az_range 1.4 --el_range 1.2 --ct_range 0.75`

## Citation

Please cite our paper if you find this code useful for your research.

```
@inproceedings{mustikovelaCVPR20,
	title = {Self-Supervised Viewpoint Learning From Image Collections},
	author = {Mustikovela, Siva Karthik and Jampani, Varun and De Mello, Shalini and Liu, Sifei and Iqbal, Umar and Rother, Carsten and Kautz, Jan},
	booktitle = {IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
	month = june,
	year = {2020}
}
```
