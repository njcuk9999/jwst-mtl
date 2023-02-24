import os
import requests
import xmltodict
import pickle

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


# ============================================================================
# Main Code
# ============================================================================
if __name__ == '__main__':

	# storage for the pi institutions and CoI institutions
	pi_institutions = dict()
	coi_institutions = dict()
	# store a list of proposal ids and the institutions
	all_storage = dict()
	# check if the pickle file exists
	if os.path.exists(PICKLE_FILE):
		# load pickle file into all_storage
		with open(PICKLE_FILE, 'rb') as f:
			all_storage = pickle.load(f)

	# try all JWST proposals using the url
	for it in range(proposal_start, proposal_end + 1):
		# print progress
		print('Proposal {0}'.format(it))
		# check if we have already downloaded this proposal
		if it in all_storage:
			print('\tID {0} Already downloaded'.format(it))
			continue
		# try to download the proposal
		try:
			response = requests.get(url.format(it))
			data = xmltodict.parse(response.content)
		except KeyboardInterrupt:
			print('Keyboard Interrupt')
			break
		except:
			print('\tFailed to download proposal id={0}'.format(it))
			continue
		# deal with no investigator report in data
		if 'investigatorReport' not in data:
			print('\tNo investigator report found')
			continue
		# deal with no investigate in data['investigatorReport']
		if 'investigator' not in data['investigatorReport']:
			print('\tNo investigator found')
			continue
		# get the authors
		authors = data['investigatorReport']['investigator']
		# deal with only one author
		if '@type' in authors:
			authors = [authors]
		# print that we have found a proposal
		print('\tFound proposal id={0}'.format(it))
		# add dictionary to all_storage
		all_storage[it] = dict(PI=[], COI=[])
		# loop around authors
		for author in authors:
			# get the institution
			institution = author['institution']
			# deal with pis
			if 'PI' in author['@type']:
				all_storage[it]['PI'].append(author)

				if institution in pi_institutions:
					pi_institutions[institution] += 1
				else:
					pi_institutions[institution] = 1
			else:
				all_storage[it]['COI'].append(author)

				if institution in coi_institutions:
					coi_institutions[institution] += 1
				else:
					coi_institutions[institution] = 1
		# remove any current pickle file
		if os.path.exists(PICKLE_FILE):
			os.remove(PICKLE_FILE)
		# write the storage to a pickle file
		with open(PICKLE_FILE, 'wb') as f:
			pickle.dump(all_storage, f)
