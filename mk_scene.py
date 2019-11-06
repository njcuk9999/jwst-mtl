import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def mk_disk_images(box_width = 79, rdisk = 7.5, wdisk = 3.0, tilt = 45.0, 
                   sky_angle = 45.0, star_to_disk_ratio = 1.0,doplot = False):

    # create a scene with a central star and a disk. The disk has some 
    # forward scattering an can be tilted in sky plane and line-of-sight angle
    #
    # box_width : size of the box. Needs to be odd; the star will be added on 
    #             the median pixel
    #
    # rdisk : radius of disk in pixels. Can be fractional
    # wdisk : e-width of the disk in pixels.
    #
    # tilt : line-of-sight tilt of the disk. Tilt = 0 would make the disk 
    #         edge-on, tilt = 70 makes it nearly-but-not-quite face-on
    #
    # sky_angle : rotate the position angle of the disk
    #    
    # star_to_disk_ratio : star is brighter than disk by this factor
    #
    # doplot : set to True if you want to see the image of the disk
    #
    
    # create the box
    im = np.zeros([box_width,box_width])
    
    # create indices that define disk position
    x1,y1 = np.indices([box_width,box_width],dtype = float)-box_width//2
    # rotate and tilt coordinates from disk to sky
    x2 = np.cos(sky_angle/(180/np.pi))*x1 + np.sin(sky_angle/(180/np.pi))*y1
    y2 = -np.sin(sky_angle/(180/np.pi))*x1 + np.cos(sky_angle/(180/np.pi))*y1
    x2 = x2/np.sin(tilt/(180/np.pi))
    
    # radius in disk coordinates
    r = np.sqrt( x2**2+y2**2 ) 
    
    # forward scattering function
    scatter = np.sin(np.arctan2(x2,y2))*np.cos(tilt/(180/np.pi))+1
    
    # flux of disk
    disk = np.exp(-.5*(r-rdisk)**2/wdisk*2)*scatter
    
    # we add a bright star in the middle
    im[box_width//2,box_width//2] = np.sum(disk)*star_to_disk_ratio
    
    # adding disk to star
    im = im+disk
    
    disk /= np.sum(im)
    # normalize image to unity
    im /= np.sum(im)
    
    if doplot:
        plt.imshow(im,vmin = np.min(im),vmax = np.max(disk)*1.2)
        plt.show()

    return im


if __name__ == "__main__":
    # get disk image
    im = mk_disk_images(doplot=True)
    # save to file
    fits.writeto('my_scene.fits', im)
    
    
