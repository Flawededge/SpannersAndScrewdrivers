# Spanners and Screwdrivers

This assignment is about training a neural network to decide if there is a spanner or a screwdriver in an image
The assignment brief is located inside "brief.pdf"

### What each file does
* download_imagenet
    * download_images.py
        * This downloads images from each line of a text file. There's a 'screwdriver.txt' and a 'spanner.txt' to test this on.
There are a bunch of bad images and broken links so it tests out the full image checking capability

* download_open-images
    * download_images.py
        * This goes through the open-images database and downloads the descriptions and annotations files.
It then grabs all of the images from the given classes with constraints
        * Doesn't get many images in the dataset I want :'(

* Neural stuff
    * Frazers Example
        * Frazer's example from stream
        * TODO: A bit of commenting in these files to understand what's going on
    
    * My Net
        * main.py
            * My attempt at replicating what Frazer's example does to get an idea of what's going on
        
