import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Etienne super ring code')
    parser.add_argument('--box_width', dest='box_width', type=int,
                        default=77,
                        help=('size of the box. NATIVE pixels, the final '
                              'image will be oversampling*box_width in size'))
    parser.add_argument('--rdisk', dest='rdisk', type=float,
                        default=7.5,
                        help=('radius of disk in pixels. Can be fractional. '
                              'This is in NATIVE pixels, NOT oversampled'))
    parser.add_argument('--wdisk', dest='wdisk', type=float,
                        default=3.0,
                        help=('e-width of the disk in pixels. Against, NATIVE'
                              ' and NOT pixels oversampled'))
    parser.add_argument('--tilt', dest='tilt', type=float,
                        default=45.0,
                        help=('line-of-sight tilt of the disk. Tilt = 0 would '
                              'make the disk edge-on, tilt = 70 makes it '
                              'nearly-but-not-quite face-on'))
    parser.add_argument('--sky_angle', dest='sky_angle', type=float,
                        default=45.0,
                        help='rotate the position angle of the disk')
    parser.add_argument('--star_to_disk_ratio', dest='star_to_disk_ratio',
                        nargs=1, type=float, default=1.0,
                        help='star is brighter than disk by this factor')
    parser.add_argument('--oversampling', dest='oversampling',
                        type=float,
                        default=11, help='oversampling factor')
    parser.add_argument('--doplot', dest='doplot', type=bool,
                        default=True,
                        help=('set to True if you want to see the image '
                              'of the disk'))
    args = parser.parse_args()
    return args


def mk_disk_images(box_width=77, rdisk=7.5, wdisk=3.0, tilt=45.0,
                   sky_angle=45.0, star_to_disk_ratio=1.0,
                   oversampling=11, doplot=False):
    # create a scene with a central star and a disk. The disk has some
    # forward scattering an can be tilted in sky plane and line-of-sight angle
    #
    # box_width : size of the box. NATIVE pixels, the final image will be 
    #             oversampling*box_width in size
    #
    # rdisk : radius of disk in pixels. Can be fractional. This is in NATIVE
    #         pixels, NOT oversampled.
    #
    # wdisk : e-width of the disk in pixels. Against, NATIVE and NOT 
    #         pixels oversampled
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
    im = np.zeros([box_width * oversampling, box_width * oversampling])

    # create indices that define disk position
    x1, y1 = np.indices([box_width * oversampling,
                         box_width * oversampling], dtype=float) / oversampling
    x1 -= box_width / 2.0
    y1 -= box_width / 2.0

    # rotate and tilt coordinates from disk to sky
    x2 = np.cos(sky_angle / (180 / np.pi)) * x1 + np.sin(sky_angle / (180 / np.pi)) * y1
    y2 = -np.sin(sky_angle / (180 / np.pi)) * x1 + np.cos(sky_angle / (180 / np.pi)) * y1
    x2 = x2 / np.sin(tilt / (180 / np.pi))

    # radius in disk coordinates
    r = np.sqrt(x2 ** 2 + y2 ** 2)

    # forward scattering function
    scatter = np.sin(np.arctan2(x2, y2)) * np.cos(tilt / (180 / np.pi)) + 1

    # flux of disk
    disk = np.exp(-.5 * (r - rdisk) ** 2 / wdisk * 2) * scatter

    # we add a bright star in the middle
    star_flux = np.sum(disk) * star_to_disk_ratio
    im[(box_width * oversampling) // 2, (box_width * oversampling) // 2] = star_flux

    # adding disk to star
    im = im + disk

    disk /= np.sum(im)
    # normalize image to unity
    im /= np.sum(im)

    if doplot:
        plt.imshow(im, vmin=np.min(im), vmax=np.max(disk) * 1.2)
        plt.show()

    return im


if __name__ == "__main__":
    # get args
    args = parse_args()
    # run function
    im = mk_disk_images(box_width=args.box_width, rdisk=args.rdisk,
                        wdisk=args.wdisk, tilt=args.tilt,
                        sky_angle=args.sky_angle,
                        star_to_disk_ratio=args.star_to_disk_ratio,
                        oversampling=args.oversampling,
                        doplot=args.doplot)
    # save to file
    fits.writeto('my_scene.fits', im, overwrite=True)
