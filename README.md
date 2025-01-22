# Resiliend Source Seeking with robot swarms - 3D

In this repository, we implement a 3D PD control in SO(3) to align robot swarms with the ascending direction given by our resilient source-seeking algorithm. For a comprehensive understanding of the mathematical theory underlying this code, we recommend reviewing the following works:

    @misc{bautista2024so3attitudecontrollersalignment,
      title={SO(3) attitude controllers and the alignment of robots with non-constant 3D vector fields}, 
      author={Jesus Bautista and Hector Garcia de Marina},
      year={2024},
      url={https://arxiv.org/abs/2406.14998}, 
    }

    @misc{acuaviva2024resilientsourceseekingrobot,
      title={Resilient source seeking with robot swarms}, 
      author={Antonio Acuaviva and Jesus Bautista and Weijia Yao and Juan Jimenez and Hector Garcia de Marina},
      year={2024},
      url={https://arxiv.org/abs/2309.02937}, 
    }

## Features
This project includes:

* A 3D resilient source-seeking algorithm implementation.
* A Proportional-Derivative (PD) feedback controller for 3D heading control, implemented in SO(3) for systems with 3 degrees of freedom (DOF).
* Numerical tools for computing mathematical expressions necessary to operate in SO(3), (e.g., different techniques to compute the exponential and logarithmic maps).

## Quick Install

To install, run:

```bash
python install.py
```

## Usage

We recommend running the Jupyter notebooks in the `notebooks` directory to get an overview of the project's structure and see the code in action.

## Credits

This repository is maintained by [Jesús Bautista Villar](https://sites.google.com/view/jbautista-research). For inquiries or further information, please get in touch with him at <jesbauti20@gmail.com>.
