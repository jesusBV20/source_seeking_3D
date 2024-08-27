# Resiliend Source Seeking in 3D

In this repository, we implement a 3D PD control in SO(3) to align with the ascending direction given by our resilient source-seeking algorithm for robot swarms. To understand the mathematical theory behind this code, we highly recommend to consult our following works:

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
  
Throughout this project, you will find the following features:

* An implementation of our resilient source-seeking algorithm in 3D.
* A Proportional + Derivative feedback controller for the 3D heading. It is implemented in SO(3) and works for systems with 3 DOF (degrees of freedom). 
* Some numerical tools to compute the mathematical expression that we need to work in SO(3) (e.g., different techniques to compute the exp and log map).
    
-> [Jesús Bautista Villar](https://sites.google.com/view/jbautista-research) is the main maintainer of this repository. He welcomes any inquiries or requests for further information and can be reached via email at <jesbauti20@gmail.com>.

## Quick Install

Just run `install.py`.
