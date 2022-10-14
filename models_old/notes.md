## CLTrial_Transformer

Clinical Transformer model in Brief does the following processes:
1. Row Transformer - Convert a row in the data sample to features
2. CT_Bert - Using the transformed record features model a bert that Convert full records to features

Train the CT_Bert first and save the weights
-> Using the saved model of CT_Bert as Feature Extractor for N number of samples for training a Classifier


Detailed Flow:

