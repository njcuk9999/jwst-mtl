import os
import pickle
import time

import numpy as np
import requests
from astropy.table import Table
from geopy.geocoders import Nominatim
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point
import itertools

# ============================================================================
# Variables
# ============================================================================
# url to download the aptx file
url = 'https://www.stsci.edu/cgi-bin/get-address-info?id={0}&markupFormat=xml&observatory=JWST'

# set the range of proposals to download
proposal_start = 1000
proposal_end = 3000
# set pickle file
PICKLE_FILE = 'all_storage.pkl'

# set up working directory
WORKING = 'H:\\jwst\jwst-mtl\\Support\\CanadianUsuage\\'

INSTIUTIONS_FILE = WORKING + 'institutions.fits'
# set the geolocator
geolocator = Nominatim(user_agent="geoapiExercises")

with open('api_key.txt', 'r') as f:
    api_key = f.readline()

world_file = WORKING + 'ne_110m_admin_0_countries\\ne_110m_admin_0_countries.shp'

# ============================================================================
# Class definitions
# ============================================================================
class Person:
    def __init__(self, id, name, institution, country='Unknown'):
        self.id = id
        self.name = name
        self.institution = institution
        self.country = country
        self.pi_proposals = []
        self.coi_proposals = []

    def add_pi_proposal(self, proposal_id):
        self.pi_proposals.append(proposal_id)

    def add_coi_proposal(self, proposal_id):
        self.coi_proposals.append(proposal_id)


# ============================================================================
# Functions
# ============================================================================
def get_lat_long(affiliation):
    # location = geolocator.geocode(affiliation)

    # if location:
    #
    #     return location.latitude, location.longitude

    # otherwise try google to find the best
    # Use the Google Maps Geocoding API to get the latitude and longitude of the institution
    affiliation = affiliation.replace(' ', '+')
    response = requests.get(
        f"https://maps.googleapis.com/maps/api/geocode/json?address={affiliation}&key={api_key}")

    if len(response.json()["results"]) == 0:
        return None, None

    result = response.json()["results"][0]
    latitude = result["geometry"]["location"]["lat"]
    longitude = result["geometry"]["location"]["lng"]

    time.sleep(0.25)  # delay to avoid hitting rate limits

    return latitude, longitude


def locate_instutions(all_storage):
    # loop around all propsals and find all unique institutions
    institutions = []
    for proposal_id, proposal in all_storage.items():

        for author in proposal['PI']:
            if author['institution'] not in institutions:
                institutions.append(author['institution'])
        for author in proposal['COI']:
            if author['institution'] not in institutions:
                institutions.append(author['institution'])

    # see if institutions exists
    if os.path.exists(INSTIUTIONS_FILE):
        table = Table.read(INSTIUTIONS_FILE)
        known_institutions = np.array(table['institution'], dtype=str)
        known_lats = np.array(table['latitude'], dtype=float)
        known_longs = np.array(table['longitude'], dtype=float)
    else:
        known_institutions, known_lats, known_longs = [], [], []

    lats, longs = [], []

    final_institutions = []
    # loop around all institutions and get the lat, long
    for institution in tqdm(institutions):
        # deal with none string institution
        if institution is None:
            continue
        # clean institution name
        institution = clean(institution)
        # deal with known institutions
        if institution in known_institutions:
            # find position of insitution in known institutions
            pos = np.where(known_institutions == institution)[0][0]
            # get the lat, long
            latitude = known_lats[pos]
            longitude = known_longs[pos]
        else:
            # get the lat, long
            latitude, longitude = get_lat_long(institution)
        # deal with None
        if latitude is None or longitude is None:
            continue
        # set the lat, long
        final_institutions.append(institution)
        lats.append(latitude)
        longs.append(longitude)

    # clean institution names of all bad characters
    cinstitutions = []
    for institution in final_institutions:
        # remove all non ascii characters
        cinstitutions.append(clean(institution))
    # Join the filtered characters back into a string
    cinstitutions = [''.join(s) for s in cinstitutions]

    table = Table()
    table['institution'] = np.array(cinstitutions).astype(str)
    table['latitude'] = np.array(lats).astype(float)
    table['longitude'] = np.array(longs).astype(float)
    table['count'] = np.zeros(len(table)).astype(int)

    table.write(INSTIUTIONS_FILE, format='fits', overwrite=True)

    return table


def clean(variable):
    # remove all non ascii characters
    variable = filter(lambda x: x.isascii(), variable)
    # join the filtered characters back into a string
    variable = ''.join(variable)
    # return the cleaned instution
    return variable


# Define a function to get the country name from a latitude and longitude coordinate
def get_country_name(latitude, longitude):
    location = geolocator.reverse(f"{latitude}, {longitude}")
    address = location.raw.get('address', {})
    return address.get('country', None)


def plot_institutions(itable):

    # convert to dataframe
    df = itable.to_pandas()


    # convert data frame to geodataframe
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    geometry = np.array(geometry)

    # get world
    world = gpd.read_file(world_file)

    # set up the figure
    plt.close()
    fig, ax = plt.subplots(figsize=(20, 20))
    # plot the world
    world.plot(color='white', edgecolor='black', ax=ax)

    colours = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'grey', 'black']
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

    combs = list(itertools.product(markers, colours))
    combs = np.random.permutation(combs)

    # loop around unique countries
    for it, country in enumerate(np.unique(df['country'])):
        if country in ['0.0', '']:
            continue
        # mask dataframe by country
        mask = df['country'] == country
        # create the geodataframe for country
        gdf = gpd.GeoDataFrame(df[mask], geometry=geometry[mask], crs='EPSG:4326')
        # Plot the GeoDataFrame using the `.plot()` method
        gdf.plot(markersize=50, ax=ax, color=combs[it][1], marker=combs[it][0])
    # show the plot
    plt.show()


def lookup_country_from_institution(institution, itable):
    # find where institution is in itable
    try:
        pos = np.where(itable['institution'] == institution)[0][0]
    except Exception as _:
        return 'Unknown'
    # get the country for known institution
    country = itable['country'][pos]
    # deal with bad country
    if country in ['', '0.0']:
        return 'Unknown'
    # return the country for this institution
    return country


def get_people(all_storage, itable):
    # loop around all proposals and created a list of people
    people = []
    for proposal_id, proposal in all_storage.items():
        # loop around pis and add to people
        for author in proposal['PI']:
            # get the name, id and institution
            if author['middleName'] is not None:
                name = '{firstName} {middleName} {lastName}'.format(**author)
            else:
                name = '{firstName} {lastName}'.format(**author)
            person_id = author['personID']
            institution = author['institution']
            country = lookup_country_from_institution(institution, itable)
            # check if the person already exists
            person_exists = False
            for person in people:
                if person.id == person_id:
                    person_exists = True
                    person.add_pi_proposal(proposal_id)
                    break
            # if person does not exist, create a new person
            if not person_exists:
                person = Person(person_id, name, institution, country)
                person.add_pi_proposal(proposal_id)
                people.append(person)
        # loop around cois and add to people
        for author in proposal['COI']:
            # get the name and institution
            if author['middleName'] is not None:
                name = '{firstName} {middleName} {lastName}'.format(**author)
            else:
                name = '{firstName} {lastName}'.format(**author)
            person_id = author['personID']
            institution = author['institution']
            country = lookup_country_from_institution(institution, itable)
            # check if the person already exists
            person_exists = False
            for person in people:
                if person.id == person_id:
                    person_exists = True
                    person.add_coi_proposal(proposal_id)
                    break
            # if person does not exist, create a new person
            if not person_exists:
                person = Person(person_id, name, institution, country)
                person.add_coi_proposal(proposal_id)
            people.append(person)
    return people


# ============================================================================
# Main Code
# ============================================================================
if __name__ == '__main__':

    # print progress
    print('Getting all proposals from pickle file')
    # check if the pickle file exists
    if os.path.exists(PICKLE_FILE):
        # load pickle file into all_storage
        with open(PICKLE_FILE, 'rb') as f:
            all_storage = pickle.load(f)
    # -------------------------------------------------------------------------
    # get the long and latitude for as many institutions as possible
    print('Locating institutions')
    itable = locate_instutions(all_storage)

    # -------------------------------------------------------------------------
    print('Identifying countries for insitutions')
    itable['country'] = np.zeros(len(itable)).astype(str)
    # get countries for each institution
    for row in tqdm(range(len(itable))):
        # can skip if we already have the country
        if 'country' in itable.colnames:
            if itable['country'][row] not in ['', '0.0']:
                continue
        # otherwise find the country
        if np.isfinite(itable['latitude'][row]):
            country = get_country_name(itable['latitude'][row],
                                       itable['longitude'][row])
            itable['country'][row] = clean(country)
    # re-save table
    itable.write(INSTIUTIONS_FILE, overwrite=True)
    # -------------------------------------------------------------------------
    print('Identifying people')
    # now identify the people (so we don't count individuals twice)
    people = get_people(all_storage, itable)

    # -------------------------------------------------------------------------
    # count statistics
    print('Counting statistics')

    # count the number of pis and cois per country
    country_pis = {}
    country_cois = {}
    for person in people:
        if person.country == 'Unknown':
            continue
        if person.country not in country_pis:
            country_pis[person.country] = 0
            country_cois[person.country] = 0
        if len(person.pi_proposals) > 0:
            country_pis[person.country] += 1
        if len(person.coi_proposals) > 0:
            country_cois[person.country] += 1

    # plot the number of pis and cois per country
    print('Plotting statistics')
    plt.close()
    fig, frames = plt.subplots(1, 2, figsize=(20, 10))
    # plot a bar chart of the pis
    frames[0].bar(country_pis.keys(), country_pis.values())
    frames[0].tick_params(axis='x', rotation=90)
    frames[0].set_title('PIs')
    # plot a bar chat of the cois
    frames[1].bar(country_cois.keys(), country_cois.values())
    frames[1].tick_params(axis='x', rotation=90)
    frames[1].set_title('CoIs')
    # show the plot
    plt.show()
    plt.close()





