# cuRecon

this will be a state of the art industrial 3d reconstruction pipeline built with c++/cuda. we will load 3d colored pointclouds from different views and create a 3d model with high resolution mesh. 




the 3d reconstruction pipeline consists of:

1. Data Preprocessing
    - AprilTag detection?
    - Cropping
    - Downsampling
    - Noise Removal
2. Feature Extraction
    - 3D point cloud deep learning model PARENet: https://github.com/yaorz97/PARENet
        - converting model to onnx (might need op registration)
        - convert model to tensorrt format
        - load model from c++ and do inference
3. Feature Matching
    - Kdtree? PointDSC++?
4. Correspondences Selection
    - Maximal Graph Cliques: https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques
5. Transformation Matrix Calculation
    - WeightedSVD
6. Mesh Creation
    - idk we'll see


