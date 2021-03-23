from mirage.apt import read_apt_xml

import numpy as np

import sys

sys.path.insert(0, '/genesis/jwst/github/jwst-mtl/AMI/')

import mirage_tools.generate_catalog_list as neil

import os.path

def HMS2deg(ra, dec):

    alpha = np.zeros(np.size(ra))
    delta = np.zeros(np.size(ra))

    for n in range(np.size((ra))):
        RA, DEC, rs, ds = '', '', 1, 1

        D, M, S = [float(i) for i in str(dec[n]).split(':')]
        if str(D)[0] == '-':
            ds, D = -1, abs(D)
        deg = D + (M/60) + (S/3600)
        DEC = '{0}'.format(deg*ds)
  
        H, M, S = [float(i) for i in str(ra[n]).split(':')]
        if str(H)[0] == '-':
            rs, H = -1, abs(H)
        deg = (H*15) + (M/4) + (S/240)
        RA = '{0}'.format(deg*rs)

        alpha[n] = RA
        delta[n] = DEC  

    return alpha, delta



def find_unique_indices(thelist):

    indices = range(len(thelist))

    # Fill output and index with unique entries
    output = []
    index = []
    for i, x in enumerate(thelist):
        if x not in output:
            output.append(x)
            index.append(indices[i])

    # Convert to integers and numpy arrays
    index = np.array(index, dtype='int')
    output = np.array(output)

    return index




def apt_field_coordinates(APTxml_filename):

    # Use the XML file reading capabilities in MIRAGE
    xmlstruct = read_apt_xml.ReadAPTXML()
    table = xmlstruct.read_xml(APTxml_filename)

    # Make numpy arrays of all targets and their RA, Dec
    alltargets = np.array(table['TargetID'])
    allRA = np.array(table['TargetRA'])
    allDec = np.array(table['TargetDec'])

    # Identify unique entries based on the target names
    ind = find_unique_indices(alltargets)

    # Convert to alpha, delta (degrees)
    alpha, delta = HMS2deg(ra=allRA[ind], dec=allDec[ind])

    return alltargets[ind], alpha, delta


def apt2cat(xml_filename, obs_date_float):
    # Run the script

    # Read in the xml file and return the fields and positions for each field
    targetname, alpha, delta = apt_field_coordinates(xml_filename)

    # Sculpt the catalogue names
    catname = []
    for i in range(len(alpha)):
        path, base = os.path.split(xml_filename)
        a,b,c = base.partition('.xml')
        catname.append(path+'/'+a+'_'+targetname[i]+'_catalog.list')

    # Make alpha delta input an array
    for i in range(len(alpha)):
        neil.make_catalog(alpha[i], delta[i], 120.0, obs_date_float, catname[i])

    # Output the catalogue name list as well as the field name list
    return catname, targetname


# Run example

# catlist, targetlist = apt2cat('/genesis/jwst/userland-ami/loic_mirage_betapic/betapic_2476_short.xml')
# Add you planet
# neil.add_source() (separation, position angle, dm) you need the V3PA from pysiaf?
# Also return the index of the target
