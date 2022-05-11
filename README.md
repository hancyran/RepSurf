# RepSurf - Surface Representation for Point Clouds

The pytorch official implementation for "Surface Representation for Point Clouds"


<div align="center">
  <img src="assets/teaser.png" width="600px">
</div>

## Preparation

### Environment

We tested under the environment:

* python 3.6
* pytorch 1.5.0
* cuda 10.1
* gcc 7.3.1

### Compile

Compile cuda-based point operators:

```
sh compile_pointops.sh
```

**Note**: if the package of pointops cannot be found globally, maybe you need to move the folder from 
**./modules/pointops** to **/usr/local/lib64/python3.6/site-packages** manually.

## Classification

### ScanObjectNN

* Performance:
<table>
  <thead>
    <tr style="text-align: center;">
      <th>Model</th>
      <th>Accuracy</th>
      <th>#Params</th>
      <th>Augment</th>
      <th>Code</th>
      <th>Log</th>
      <th>Checkpoint</th>
    </tr>
  </thead>
  <tbody>
    <tr style="text-align: center;">
      <td><a href="https://github.com/ajhamdi/MVTN">MVTN</a></td>
      <td>82.8</td>
      <td>4.24M</td>
      <td>None</td>
      <td><a href="https://github.com/ajhamdi/MVTN/blob/master/models/mvtn.py">link</a></td>
      <td>N/A</td>
      <td><a href="https://github.com/ajhamdi/MVTN/blob/master/results/checkpoints/scanobjectnn/model-00029.pth">link</a></td>
    </tr>
    <tr style="text-align: center;">
      <td><a href="https://github.com/ma-xu/pointMLP-pytorch">PointMLP</a></td>
      <td>85.7</td>
      <td>12.6M</td>
      <td>Scale, Shift</td>
      <td><a href="https://github.com/ma-xu/pointMLP-pytorch/blob/main/classification_ScanObjectNN/models/pointmlp.py">link</a></td>
      <td><a href="https://web.northeastern.edu/smilelab/xuma/pointMLP/checkpoints/fixstd/scanobjectnn/pointMLP-20220204021453/">link</a></td>
      <td><a href="https://web.northeastern.edu/smilelab/xuma/pointMLP/checkpoints/fixstd/scanobjectnn/pointMLP-20220204021453/">link</a></td>
    </tr>
    <tr style="text-align: center;">
      <td>PointNet++ SSG</td>
      <td>77.9</td>
      <td>1.475M</td>
      <td>Rotate, Jitter</td>
      <td><a href="https://github.com/hkust-vgd/scanobjectnn/blob/master/pointnet2/models/pointnet2_cls_ssg.py">link</a></td>
      <td>N/A</td>
      <td>N/A</td>
    </tr>
    <tr style="text-align: center;">
      <td><b>Umbrella RepSurf</b> (PointNet++ SSG)</td>
      <td><b>84.87</b></td>
      <td>1.483M</td>
      <td>None</td>
      <td><a href="models/repsurf/scanobjectnn/repsurf_ssg_umb.py">repsurf_ssg_umb</a></td>
      <td><a href="https://drive.google.com/file/d/1qJK8T3dhF6177Xla227aXPEeNtyNssLF/view?usp=sharing">google drive</a></td>
      <td><a href="https://drive.google.com/file/d/17UDArfvNVjrJBTjr_HdxcOQipn0DWMMf/view?usp=sharing">google drive (6MB)</a></td>
    </tr>
    <tr style="text-align: center;">
      <td><b>Umbrella RepSurf</b> (PointNet++ SSG, 2x)</td>
      <td><b>86.05</b></td>
      <td>6.806M</td>
      <td>None</td>
      <td><a href="models/repsurf/scanobjectnn/repsurf_ssg_umb_2x.py">repsurf_ssg_umb_2x</a></td>
      <td><a href="https://drive.google.com/file/d/15HwmAi1erL68G08dzNQILSipwCIDfNAw/view?usp=sharing">google drive</a></td>
      <td><a href="https://drive.google.com/file/d/1yGPNt1REzxVwn8Guw-PFHFcwxvfueWgf/view?usp=sharing">google drive (27MB)</a></td>
    </tr>
  </tbody>
</table>
<br>

* To download dataset:

```
wget http://download.cs.stanford.edu/orion/scanobjectnn/h5_files.zip
unzip h5_files.zip
```

**Note**: We conduct all experiments on the hardest variant of ScanObjectNN (**PB_T50_RS**).
<br>

* To train Umbrella RepSurf on ScanObjectNN:

```
sh scripts/repsurf/scanobjectnn/repsurf_ssg_umb.sh
```

* To train Umbrella RepSurf (2x setting):

```
sh scripts/repsurf/scanobjectnn/repsurf_ssg_umb_2x.sh
```

## Visualization

We provide several visualization results in the folder **./visualization** for a closer look at the construction of RepSurf.


## TODO

- [ ] Classification on ModelNet40
- [ ] Segmentation on S3DIS / ScanNet
- [ ] Detection on ScanNet / SUN RGB-D based on [GroupFree3D](https://github.com/zeliu98/Group-Free-3D)

## Acknowledgment
We use library [pointops](https://github.com/hszhao/PointWeb/tree/master/lib/pointops) from [PointWeb](https://github.com/hszhao/PointWeb). 


## License
RepSurf is under the Apache-2.0 license. Please contact the primary author **Haoxi Ran (ranhaoxi@gmail.com)** for commercial use.
