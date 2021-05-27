# Compute local reference frames using [FLARE](http://www.vision.deis.unibo.it/research/78-cvlab/82-lrf)

* Requirement:  Install [PCL](https://pointclouds.org/downloads/)

Compute the LRF:
```
mkdir build && cd build
cmake .. && make
./flare_estimation
```

* Reference: A. Petrelli, L. Di Stefano, "A repeatable and efficient canonical reference for surface matching", 3DimPVT, 2012. [[PDF](http://www.vision.deis.unibo.it/LRF/LRF_repeatability_3DimPvt2012.pdf)]
