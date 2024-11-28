
class Regularizz_reg:
    def __init__(self,epochs,eeta,degree,lambd,reg_type ='none'):
        self.degree = degree
        
        self.epochs = epochs
        self.LR = eeta
        self.L = lambd
        self.reg_type = reg_type

        self.coefficients = np.zeros(degree+1)
        
    def predict(self, X):
            y_pred = np.zeros(X.shape[0])
            for j in range(X.shape[0]):
                prediction = 0
                for i in range(self.degree + 1):
                    prediction += self.coefficients[i] * (X[j] ** i)
                y_pred[j] = prediction
            return y_pred
            
    def training(self,X_train,y_train):
        all_coefficients = []

        for epoch in range(self.epochs):
            del_l_del_b = 0 
            del_l_del_m = 0
            gradients = np.zeros(self.degree + 1)
            
            for k in range(X_train.shape[0]):
                half =0
                for i in range(self.degree + 1):
                    half = half + self.coefficients[i]*(X_train[k]**i)
                full = y_train[k] - half

                for i in range(self.degree + 1):
                    gradients[i] -=  2*(full)*(X_train[k]**i)

            if self.reg_type == 'l1':
                for i in range(self.degree+1):
                    gradients[i] = gradients[i] + self.L*2*self.coefficients[i]
            elif self.reg_type == 'l2':
                for i in range(self.degree+1):
                    gradients[i] = gradients[i] + self.L*2
            
  
            self.coefficients = self.coefficients - (self.LR*gradients)
            all_coefficients.append(self.coefficients.copy())

            
            print(f'Epoch {epoch + 1}, Coefficients: {self.coefficients}')
        return all_coefficients

