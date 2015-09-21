#!/usr/bin/python

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
from numpy import *

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print "There are", len(enron_data.keys()), "people in the dataset."
print "There are", len(enron_data[enron_data.keys()[1]]), "features per person."

# Number of POIs in the dataset
poi_n = 0

for person in enron_data.keys():
    if enron_data[person]["poi"] == True:
        poi_n += 1

print "There are", poi_n, "POIs in the dataset."

print "James Prentice had", enron_data["PRENTICE JAMES"]["total_stock_value"], \
    "dollars worth of total stock value."

print "Wesley Colwell sent", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"],\
    "emails to POIs."

print "Jeffrey Skilling had", enron_data["SKILLING JEFFREY K"]["exercised_stock_options"], \
    "dollars worth of exercised stock options."

print "Jeffrey Skilling took out", enron_data["SKILLING JEFFREY K"]["total_payments"]/1e6, \
    "million dollars."

print "Kenneth Lay took out", enron_data["LAY KENNETH L"]["total_payments"]/1e6, \
    "million dollars."

print "Andrew Fastow took out", enron_data["FASTOW ANDREW S"]["total_payments"]/1e6, \
    "million dollars."

# Number of people with known salaries and/or e-mails
salary_n = 0
email_n = 0

for person in enron_data.keys():
    if enron_data[person]["salary"] != "NaN":
        salary_n += 1
    if enron_data[person]["email_address"] != "NaN":
        email_n += 1

print "There are", salary_n, "quantified salaries and", email_n, "known email addresses."

# Number of people with known total payments
payout_n = 0

for person in enron_data.keys():
    if enron_data[person]["total_payments"] != "NaN":
        payout_n += 1

payout_n_unknown = len(enron_data.keys()) - payout_n

print "There are", payout_n_unknown, "unquantified salaries, which is", \
    payout_n_unknown*1./len(enron_data.keys())*100., "percent of the total people in the dataset."

payout_poi_n = 0

for person in enron_data.keys():
    if enron_data[person]["poi"] == True:
        if enron_data[person]["total_payments"] != "NaN":
            payout_poi_n += 1

payout_poi_n_unknown = poi_n - payout_poi_n

print "There are", payout_poi_n_unknown, "unquantified salaries, which is", \
    payout_poi_n_unknown*1./poi_n*100., "percent of the total POIs in the dataset."



