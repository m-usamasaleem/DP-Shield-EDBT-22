import pandas as pd
import os
import app
from airtable import Airtable

class Updater():
    def __init__(self):
        self.awd_path = ''
        self.prop_path = ''
        self.both_verified = False
        self.err_msg = ''

        self._award_df = None
        self._proposal_df = None

        self._awards_table = Airtable(app.airtable_table_key, 'NORM_Awards_By_Dept', app.airtable_api_key)
        self._submissions_table = Airtable(app.airtable_table_key, 'NORM_Proposals_By_Dept', app.airtable_api_key)
        self._dept_codes = Airtable(app.airtable_table_key, 'Department Codes', app.airtable_api_key)

    def set_paths(self, award_path, proposal_path):
        self._reset()
        self.awd_path = award_path
        self.prop_path = proposal_path

    def check_awd_and_prop(self):
        award_verified = self._check_awd_file()
        proposal_verified = self._check_proposal_file()

        if award_verified and proposal_verified: # if the csv's are verified
            # we want to update the verified bool and set the paths to the new csv's
            self.both_verified = True
            self.err_msg = 'Proposal and award csv\'s verified. Continue with next step'
        else:
            self.err_msg += 'One of the two csv\'s could not be verified'

        print('done checking')
        return (self.both_verified, self.err_msg)

    def find_new_grants(self):
        # DEBUG: 
        new_awds_found = 5
        self._find_new_grants_awards()
        new_props_found = 5
        self._find_new_grants_proposals()
        print('finished running both')
        return (new_awds_found, new_props_found)

    def run_update(self):
        return True

    def _find_new_grants_awards(self):
        award_records_to_insert = []

        #
        # Find records that do not exist in Airtable, add them to award_records_to_insert
        #
        for idx, grant_raw in enumerate(self._award_df.iterrows()):
            grant = grant_raw[1]

            # DEBUG: Remove when done
            #########################
            if idx != 0 and idx % 100 == 0:
                print(f'{idx} records searched')

            if idx==100:
                break
            #########################

            if grant['FISCAL YEAR'] and int(grant['FISCAL YEAR']) <= 2019: # dont look into grants awarded in or before 2019
                continue

            air_rec = self._awards_table.match('AWARD NO', grant['AWARD NO'])
            contained = bool(air_rec)

            # DEBUG: Change to not contained
            if contained:
                award_records_to_insert.append(grant_raw)

                # some fields are generated within Airtable after updating. I.e. these fields are not updated until we
                # push these new records to Airtable. Below we manually generate the missing fields to show the user
                dept_record = self._dept_codes.search('Name', grant['AWARD ADMIN DEPT NO'])
                grant_college = dept_record[0]['fields']['College Text']
                grant_dept = dept_record[0]['fields']['Dept Name']

                # in some (odd) cases, awards will not contain an "Other Personnel" field. We account for that below
                if not 'OTHER PERSONNEL' in grant.keys():
                    grant['OTHER PERSONNEL'] = "MISSING OTHER PERSONNEL FIELD"
                
                app.pusher_client.trigger('awards-channel', 'award-updates', {
                    'Prop-num': grant['PROPOSAL NO'],
                    'Award-title': grant['AWARD TITLE'],
                    'Award Amount': grant['AWARD AMOUNT'],
                    'PI': grant['AWARD PI'],
                    'Other-personnel': grant['OTHER PERSONNEL'],
                    'Date': '%s/%s' % (grant['REPORT MONTH'], grant['FISCAL YEAR']),
                    'College': grant_college,
                    'Dept': grant_dept
                })

        #
        # Prepare the records to be inserted into Airtable
        #
        proced_records = []

        for grant_raw in award_records_to_insert:
            grant = grant_raw[1]

            if grant['AWARD AMOUNT'] is '' or grant['REPORT MONTH'] is '' or grant['FISCAL YEAR'] is '' or grant['AWARD ADMIN DEPT NO'] is '':
                continue

            # Basic fixes
            grant['AWARD AMOUNT'] = int(grant['AWARD AMOUNT'].replace('$', '').replace(',', '')[:-4])
            grant['REPORT MONTH'] = int(grant['REPORT MONTH'])
            grant['FISCAL YEAR'] = int(grant['FISCAL YEAR'])

            # Convert dept number to department record id
            dept_record = self._dept_codes.search('Name', grant['AWARD ADMIN DEPT NO'])
            grant['AWARD ADMIN DEPT NO'] = [dept_record[0]['id']]

            proced_records.append(grant)

        award_records_to_insert = proced_records

        # return len(award_records_to_insert)

    def _find_new_grants_proposals(self):
        proposal_records_to_insert = []
        #
        # Find records that do not exist in Airtable, add them to proposal_records_to_insert
        #
        print('Inside proposal find new')

        for idx, grant_raw in enumerate(self._proposal_df.iterrows()):
            grant = grant_raw[1]

            # DEBUG: Remove when done
            #########################
            if idx != 0 and idx % 100 == 0:
                print(f'{idx} records searched - proposals')

            if idx==20:
                break
            #########################

            # DEBUG: Uncomment below 2 lines
            # if grant['FISCAL YEAR'] and int(grant['FISCAL YEAR']) <= 2019: # dont look into grants awarded in or before 2019
            #     continue

            air_rec = self._submissions_table.match('Proposal Number', grant['Proposal Number'])
            contained = bool(air_rec)

            # DEBUG: Change to not contained
            if contained:
                proposal_records_to_insert.append(grant_raw)

                # some fields are generated within Airtable after updating. I.e. these fields are not updated until we
                # push these new records to Airtable. Below we manually generate the missing fields to show the user
                dept_record = self._dept_codes.search('Name', grant['Award Admin Dept No'])
                grant_college = dept_record[0]['fields']['College Text']
                grant_dept = dept_record[0]['fields']['Dept Name']

                # in some (odd) cases, awards will not contain an "Other Personnel" field. We account for that below
                if not 'Other Personnel' in grant.keys():
                    grant['Other Personnel'] = "MISSING OTHER PERSONNEL FIELD"
                
                app.pusher_client.trigger('proposals-channel', 'proposal-updates', {
                    'Prop-num': grant['Proposal Number'],
                    'Award-title': grant['Title'],
                    'Total Funds': grant['Total Funds'],
                    'PI': grant['PI'],
                    'Other-personnel': grant['Other Personnel'],
                    'Date': '%s/%s' % (grant['PROPOSAL APPROVED MONTH'], grant['FISCAL YEAR']),
                    'College': grant_college,
                    'Dept': grant_dept,
                    'Status': grant['Status']
                })

        #
        # Prepare records to be inserted into Airtable
        #
        proced_records = []

        for grant_raw in proposal_records_to_insert:
            grant = grant_raw[1]

            if grant['Total Funds'] is '' or grant['PROPOSAL APPROVED MONTH'] is '' or grant['FISCAL YEAR'] is '' or grant['Award Admin Dept No'] is '':
                print('removed grant from awards') # DEBUG: Remove
                continue

            # Basic fixes
            grant['Total Funds'] = int(grant['Total Funds'].replace('$', '').replace(',', '')[:-4])
            grant['Initial Funds'] = int(grant['Initial Funds'].replace('$', '').replace(',', '')[:-4])
            grant['PROPOSAL APPROVED MONTH'] = int(grant['PROPOSAL APPROVED MONTH'])
            grant['FISCAL YEAR'] = int(grant['FISCAL YEAR'])

            # Convert dept number to department record id
            dept_record = self._dept_codes.search('Name', grant['Award Admin Dept No'])
            grant['Award Admin Dept No'] = [dept_record[0]['id']]

            proced_records.append(grant)

        proposal_records_to_insert = proced_records

        # return len(proposal_records_to_insert)

    def _check_awd_file(self):
        awd_name_mapper = {
            "Official Report Date": "OFFICIAL REPORT DATE",
            "MONTH": "REPORT MONTH"
        }

        try: 
            award_df = pd.read_html(self.awd_path, header=0)[0]

            if 'total' in award_df.tail(1).iloc[-1, 0].lower():
                award_df.drop(award_df.tail(1).index, inplace=True)
        except Exception as e:
            self.err_msg += f'An exception occured{e}\n'
            return False

        if 'AWARD NO' in award_df.columns:
            try:
                award_df.rename(columns=awd_name_mapper, inplace=True)

                award_df['FISCAL YEAR'] = award_df['FISCAL YEAR'].astype(int)
                award_df['REPORT MONTH'] = award_df['REPORT MONTH'].astype(int)

                award_df.fillna('', inplace=True)
            except Exception as e:
                self.err_msg += f'An exception occured{e}\n'
                return False
        else:
            self.err_msg += 'Uploaded Award sheet does not look like normal award sheet. Missing \'AWARD NO\' field...'
            return False

        new_path = os.path.join(app.app.config['UPLOAD_FOLDER'], 'processed_csvs/') + 'awards.csv'
        self.awd_path = new_path

        self._award_df = award_df
        award_df.to_csv(new_path, index = None, header=True)
        return True

    def _check_proposal_file(self):
        proposal_name_mapper = {
            "Fiscal Year": "FISCAL YEAR",
            "Prop Approval Month": "PROPOSAL APPROVED MONTH"
        }

        try: 
            proposal_df = pd.read_html(self.prop_path, header=0)[0]

            if 'total' in proposal_df.tail(1).iloc[-1, 0].lower():
                proposal_df.drop(proposal_df.tail(1).index, inplace=True)
        except Exception as e:
            self.err_msg += f'An exception occured{e}\n'
            return False

        if 'AWARD NO' not in proposal_df.columns:
            try:
                proposal_df.rename(columns=proposal_name_mapper, inplace=True)

                proposal_df['FISCAL YEAR'] = proposal_df['FISCAL YEAR'].astype(int)
                proposal_df['PROPOSAL APPROVED MONTH'] = proposal_df['PROPOSAL APPROVED MONTH'].astype(int)

                proposal_df.fillna('', inplace=True)

            except Exception as e:
                self.err_msg += f'An exception occured{e}\n'
                return False
        else:
            self.err_msg += 'Uploaded proposal sheet does not look like normal proposal sheet. It contains \'AWARD NO\' field...'
            return False

        new_path = os.path.join(app.app.config['UPLOAD_FOLDER'], 'processed_csvs/') + 'proposals.csv'
        self.prop_path = new_path

        self._proposal_df = proposal_df
        proposal_df.to_csv(new_path, index = None, header=True)
        return True

    def _reset(self):
        self.__init__()



