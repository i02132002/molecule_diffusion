# Molecule Diffusion Project
<img src="https://github.com/i02132002/molecule_diffusion/blob/master/device_figure.png" width="500" />
In this study, I track molecular diffusion trajectories on graphene and analyze the diffusion paths of the molecules under different gating conditions. I analyzed the orientation of the molecules as they hopped from one site to another and discovered that the diffusion path taken by the molecules depends strongly on whether they are charged or not. This was a surprising discovery which revealed the nature of electrostatically tunable adsorption of charge-carrying molecules on electrodes.


5GB of video data obtained by scanning tunneling microscopy (STM) was processed to obtain this result. An example of the video data analyzed:


<img src="https://github.com/i02132002/molecule_diffusion/blob/master/demo_movie.gif" width="500" />

## Orientation analysis
I used k-means clustering to identify and track molecule orientation changes from image moments calculated from imperfect experimental images.


<img src="https://github.com/i02132002/molecule_diffusion/blob/master/molecule_orientation_figure.png" width="500" />

## Full project
Please see `MotionAnalyzer-demo.ipynb` for detailed analysis!
