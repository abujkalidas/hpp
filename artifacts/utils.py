import numpy as np
import pickle
class hpp():
    
    def __init__(self, data):
        self.data = data
    
    def load_model(self):
        with open('artifacts/model.pkl', 'rb') as file:
            self.model = pickle.load(file)
            
            
    def predict(self):
        
        self.load_model()
        
        CRIM = self.data['CRIM']
        ZN = self.data['ZN']
        INDUS = self.data['INDUS']
        CHAS = self.data['CHAS']
        NOX = self.data['NOX']
        RM = self.data['RM']
        AGE = self.data['AGE']
        DIS = self.data['DIS']
        RAD = self.data['RAD']
        TAX = self.data['TAX']
        PTRATIO = self.data['PTRATIO']
        B = self.data['B']
        LSTAT = self.data['LSTAT']
        
        array = np.array([CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT], ndmin=2, dtype=float)
        
        result = np.around(self.model.predict(array), 2)[0]
        
        return result
    