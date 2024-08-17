:compass: SimpleLoopDetector
=====
### :beer: :beer: This work is based on [DBoW2](https://github.com/dorian3d/DBoW2).
Thanks for their amazing works!

## Dependences
- OpenCV (Tested with 4.4.0)

## Installation

```shell
mkdir -p build && cd build
cmake .. # tested with cmake 3.22.1 and gcc/cc 11.4.0 on Ubuntu
make # tested with GNU Make 4.3
sudo make install
```

## Usage
### Working Pipeline
1. Load a trained ORB vocabulary used in [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3).
2. Load a sequence of Image and compute their ORB features.
3. Retrieval by BoW.

### Run
```bash
cd build
./retrieval /path/to/ORBvoc.txt /path/to/image/root /path/to/image_name_file
```

## Todo
- [ ] Use yml config for arguement parsing
- [ ] Output the retireval results to file
- [ ] Pybind


## Citing

If you use this software in an academic work, please cite:
```bash
@ARTICLE{GalvezTRO12,
  author={G\'alvez-L\'opez, Dorian and Tard\'os, J. D.},
  journal={IEEE Transactions on Robotics},
  title={Bags of Binary Words for Fast Place Recognition in Image Sequences},
  year={2012},
  month={October},
  volume={28},
  number={5},
  pages={1188--1197},
  doi={10.1109/TRO.2012.2197158},
  ISSN={1552-3098}
}
```


