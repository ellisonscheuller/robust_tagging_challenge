import sys

output_dir = '/app/output/'
program_dir = '/app/program'
submission_dir = '/app/ingested_program'

sys.path.append(program_dir)
sys.path.append(submission_dir)

# START CUSTOM CODE
## Path to the train/validation data and input test data
## needed for the user submission
## Make sure to share this path with the service account mlchallenges
input_dir = 'REPLACE_ME'  # NRP PVC mount path for train data + public eval files

def main():
    from model import Model
    m = Model(input_dir, output_dir)
    print('Running Training')
    m.fit()
    print('-' * 10)
    print('Running Prediction')
    m.predict()
    print('-' * 10)
    print('Completed Prediction.')
    print('Ingestion Program finished. Moving on to scoring')

# END CUSTOM CODE

if __name__ == '__main__':
    main()
