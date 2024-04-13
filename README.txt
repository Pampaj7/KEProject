HOW TO RUN & EXPECTED TIME TO COMPLETE

pc specs for the tests:
i9 10900k 5Ghz
48GB DDR4 3600Mhz
RTX 4060Ti 8GB
2.5Gb/s internet connection

10GB of free space on the disk required

1. install required packages
2. DO NOT run textSplitter.py -- this file call the API for OPENAI and will cost us money
also the llama etc. models have keys with limited token -- expected times 2-6 hours
3. run mistralExtractor.py to extract the data from the local model. expected time 1-2 hours
4. run main.py to run the tests -- expected time 10-30 minutes

total time expected 3-8 hours