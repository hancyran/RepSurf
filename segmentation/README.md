# RepSurf for Segmentation <br>

By *[Haoxi Ran\*](https://hancyran.github.io/) , Jun Liu, Chengjie Wang* ( * : corresponding contact)

### [PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Ran_Surface_Representation_for_Point_Clouds_CVPR_2022_paper.pdf) | [arXiv](http://arxiv.org/abs/2205.05740)


## Preparation

### Environment

We tested under the environment:

* python 3.7
* pytorch 1.6.0 / 1.8.0
* cuda 10.1 / 11.1
* gcc 7.2.0
* h5py
* sharedarray
* tensorboardx

For anaconda user, initialize the conda environment **repsurf-seg** by:

```
sh init.sh
```

## Experiments

### S3DIS Area-5 (Data & Logs: [Google Drive](https://drive.google.com/drive/folders/1jIZuy4RPFJ4YHAE8ScVQgwtBwNGgfKnv?usp=sharing))

* Performance using the same settings:

<table style="width:100%">
  <thead>
    <tr>
      <th>Model</th>
      <th>mIoU</th>
      <th>mAcc</th>
      <th>OA</th>
      <th>#Params</th>
      <th>Training Time</th>
      <th>Code</th>
      <th>Training Log</th>
      <th>Test Log</th>
      <th>Checkpoint</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Point Transformer <br> (our settings)</td>
      <td align="center">70.37 (official: 70.4)</td>
      <td align="center">77.02 (official: 76.5)</td>
      <td align="center">90.80 (official: 90.8)</td>
      <td align="center">7.767M</td>
      <td align="center">19.91h</td>
      <td align="center"><a href="./models/pointtransformer/pointtransformer.py">pointtransformer.py</a></td>
      <td align="center"><a href="https://drive.google.com/file/d/1cLQetUso-fVzlfcJODXlfV-7MXa3vl-Y/view?usp=sharing">google drive</a></td>
      <td align="center"><a href="https://drive.google.com/file/d/1umrMvmwLsexKUZytcMdE12ek8xIk8E3_/view?usp=sharing">google drive</a></td>
      <td align="center"><a href="https://drive.google.com/file/d/1XnbRR2Yi6MFWVl5LVtBxLOTBN9qhuxlV/view?usp=sharing">google drive <br> (30 MB)</a></td>
    </tr>
    <tr>
      <td align="center">PointNet++ SSG (our settings)</td>
      <td align="center">64.05</td>
      <td align="center">71.52</td>
      <td align="center">87.92</td>
      <td align="center">0.968M</td>
      <td align="center">9.08h</td>
      <td align="center"><a href="./models/pointnet2/pointnet2_ssg.py">pointnet2_ssg.py</a></td>
      <td align="center"><a href="https://drive.google.com/file/d/1xUkUB0iT-WYzzzR5yiWhZkSYPjjarKlC/view?usp=sharing">google drive</a></td>
      <td align="center"><a href="https://drive.google.com/file/d/1floQ53zgTxSs_nDn_MosIUWz4Rt7eHQx/view?usp=sharing">google drive</a></td>
      <td align="center"><a href="https://drive.google.com/file/d/1hdj7G8dplCouHYor16pChd7pB8M4rodu/view?usp=sharing">google drive <br> (4 MB)</a></td>
    </tr>
    <tr>
      <td align="center">PointNet++ SSG <b>w/ Umbrella RepSurf</b> (ours)</td>
      <td align="center"><b>68.86</b></td>
      <td align="center"><b>76.54</b></td>
      <td align="center"><b>90.22</b></td>
      <td align="center"><b>0.976M</b></td>
      <td align="center">9.18h</td>
      <td align="center"><a href="./models/repsurf/repsurf_umb_ssg.py">repsurf_umb_ssg.py</a></td>
      <td align="center"><a href="https://drive.google.com/file/d/1C1mG7XFsJAiQYHMNuA8bVitEuY4TGXKY/view?usp=sharing">google drive</a></td>
      <td align="center"><a href="https://drive.google.com/file/d/1mNgmWhYcp2njwJybkGjLVModERCR9fr8/view?usp=sharing">google drive</a></td>
      <td align="center"><a href="https://drive.google.com/file/d/1pmXBt4wHKpC5llmD6pMNo2NmZZKNIQaq/view?usp=sharing">google drive <br> (4 MB)</a></td>
    </tr>
  </tbody>
</table>
<br>

**Note**: 
1. The performance (mIoU/mAcc/OA) are from the final predictions on the whole scenes of S3DIS Area-5, while the results during training is on sub-sampled scenes for fast validation. 
2. The training time of all above implementations is estimated on four NVIDIA RTX 3090. The time in the logs contains both training and validating time.
3. To speed up the training process, we apply Sectorized FPS (in the first stage) for all above methods. It can save 30～40% training time and does not affect the performance.   
4. To lessen the instability from grid sampling during inference, we apply median filtering to all the above implementations. Besides, it can slightly improve the results (~0.4 mIoU).

* To (firstly install gdown by **pip install gdown** and) download dataset:

```
cd ./data/S3DIS
gdown https://drive.google.com/u/1/uc?id=1UDM-bjrtqoIR9FWoIRyqLUJGyKEs22fP
tar zxf s3dis.tar.gz && rm s3dis.tar.gz && cd -
```

* To train one model (**Umbrella RepSurf, Point Transformer, PointNet2**) for S3DIS Area-5:

```
sh scripts/s3dis/train_[MODEL].sh  # MODEL: repsurf_umb, pointnet2, pointtransformer
```

* To test one model (**Our Umbrella RepSurf, Point Transformer, PointNet2**) for S3DIS Area-5 on whole scenes:

```
sh scripts/s3dis/test_[MODEL].sh  # MODEL: repsurf_umb, pointnet2, pointtransformer
```

## Acknowledgment

We thank the [Point Transformer Implementation](https://github.com/POSTECH-CVLab/point-transformer) for the library pointops.

## License

RepSurf is under the Apache-2.0 license. Please contact the primary author **Haoxi Ran (ranhaoxi@gmail.com)** for
commercial use.
