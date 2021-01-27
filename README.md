# Pruned-JPEG-docker
Files to build Docker image for pruned JPEG classification  

## Usage

0. Pre-process (only necessary when using new images for experiments)
    
    This is to make sure that the image sizes are only dependent on the number of DCT coefficients, not dependent on the compression parameters when JPEG are generated.    
    
    Put the new images in a folder `test_images_original`, create an empty folder `test_images_fjpeg`, and run `full_jpeg.py` to generate the JPEG images with full (64) DCT coefficients for experiments. Then, replace `test_images` by `test_images_fjpeg` in `Dockerfile`. (The folder names can be changed, but remember to change the corresponding names in `full_jpeg.py`, and that in `Dockerfile`.)  
    
1. Build docker image

    ```
    sudo docker build -t pruned-jpeg .
    ```
    This is to build a docker image named `pruned-jpeg`.
    
2. Run `pruned-jpeg` image
    
    ```
    sudo apt-get install x11-xserver-utils
    ```
    ```
    xhost +
    ```
    These two commands will disable access control, and clients can connect from any host. By doing so, the figures generated in the container can be displayed in the  monitor.
       
    ```
    sudo docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY pruned-jpeg
    ```
    This command will run the `pruned-jpeg` image interactively.
    
    In the terminal, `No.x of total n images is being processed` will indicate which image is being processed. After all `n` images are processed, the results will be shown in three figures.
    
    `Fig. 1` shows the inference accuracy of pruned JPEG images when pruning 1 to 64 DCT coefficients, and that of full JPEG images.
    
    `Fig. 2` shows the image size of pruned JPEG images when pruning 1 to 64 DCT coefficients, and that of full JPEG images.
    
    `Fig. 3` shows the relationship between the inference accuracy and the normalized pruned JPEG image size, and the fitted curve.
