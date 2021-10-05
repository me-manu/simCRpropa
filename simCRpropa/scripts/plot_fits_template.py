import matplotlib.pyplot as plt
from gammapy.maps import Map
from pathlib import PosixPath
from astropy.io import fits
import argparse

if __name__ == '__main__':
    usage = "usage: %(prog)s -f path/to/fits/file.fits"
    description = "plot a fits template"
    parser = argparse.ArgumentParser(usage=usage,description=description)
    parser.add_argument('-f','--file', required=True)
    args = parser.parse_args()

    filepath = PosixPath(args.file)
    
    try:
        m = Map.read(fits_file)
    except:

        fits_file = fits.open(filepath)

        # get meta data, gives problems with JSON encoder
        meta = fits_file[0].header.pop("META", None)

        # read in map
        m = Map.from_hdulist(fits_file)

    # plot the map
    fig, ax, _ =  m.sum_over_axes(['energy_true']).plot(stretch="log", add_cbar=True)
    ax.tick_params(direction="out")
    ax.grid(ls=":", color="0.5", lw=0.5)
    plt.savefig("template_map.png")

