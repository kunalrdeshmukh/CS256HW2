import numpy as np
import os
import math
import sys
import utils # imports utils.py file


class SKAlgo(object):
    FILE_FORMAT = ["Q.PNG", "O.PNG", "P.PNG", "S.PNG"]
    CLASS_LETTER = ["O", "P", "W", "Q", "S"]

    # I = [...k]
    # alpha = [1,0,1,0,0,0,0] # Alpha for the first positive image =1
    # and the same for the first negative on
    def __init__(self):
        self.alpha = []
        self.Ip = []  # I+
        self.In = []  # I-
        self.xi1 = np.array([],dtype=float)  # positive input data set
        self.xj1 = np.array([],dtype=float)  # negative input data set
        self.X = []  # Array of all training images.
        self.A, self.B, self.C = np.array([],dtype=float),np.array([],
                                        dtype=float),np.array([],dtype=float)
        self.D, self.E = np.array([],dtype=float),np.array([],dtype=float)
        self.mp, self.mn = np.array([]),np.array([]) # centroids m+ , m-
        self.m = np.array([]) # centroid of all images in train set




    def read(self, class_letter, train_folder_name):
        """Reads images from folder specified."""
        found_data = False  # flag to check if folder has data
        indexCounter = 0 # to count iteration
        first_positive = True # used to set alpha
        first_negative = True # used to set alpha
        names_of_files = []
        for file_name in os.listdir(train_folder_name):
            names_of_files.append(file_name)
        # names_of_files.sort() # sorts file names in alphanumeric order
        for file_name in names_of_files:
            if ("_" + class_letter + ".") in file_name:
                self.Ip.append(indexCounter)
                if first_positive:
                    self.alpha.append(1)
                    first_positive = False
                else :
                    self.alpha.append((0))
                found_data = True

            else:
                self.In.append(indexCounter)
                if first_negative:
                    self.alpha.append(1)
                    first_negative = False
                else:
                    self.alpha.append(0)
                found_data = True
            self.X.append(
                utils.load_image(train_folder_name + "/" + file_name))
            indexCounter += 1
        if not found_data:   # If no data in the folder specified print no data
            print("NO DATA")
            sys.exit()
        return names_of_files




    def calculate_centroid(self):
        """Calculates Centroids m+ , m-"""
        xi_sum = self.X[1]  # initialize
        xip_sum = 0  # initializing wth the first +vs element
        xin_sum = 0  # initializing wth the first -ve element
        for i in xrange(1, len(self.X)):
            xi_sum = np.add(xi_sum, self.X[i])
        self.m = xi_sum / len(self.X)
        # calculating m+
        for i in self.Ip:  # starting from the second postitive element
            xip_sum = np.add(xip_sum,self.X[i])
        self.mp = xip_sum / len(self.Ip)
        # calculating m-
        for i in self.In:  # starting from the second postitive element
            xin_sum = np.add(xin_sum, self.X[i])
        self.mn = xin_sum / len(self.In)



    def kernel(self, x, y, kernel_type):
        """ Calculates kernal value"""
        if kernel_type == 'P':  # Polynomial kernel
            return self.__polynomial_kernel(x, y, 4)  # For this HW, p = 4
        else:
            print 'Kernel type is not supported'
            sys.exit()

    def __polynomial_kernel(self, x, y, p):
        ans = (np.dot(x.transpose(), y) + 1)
        return ans ** p


    def prime(self, x):
        """ Calculates prime of image. """
        # m calculation formula indicates in slide 5 (Sep 27)
        self.calculate_centroid()
        r = np.linalg.norm(self.mp - self.mn)
        xip_norms = []
        xin_norms = []
        for ele in self.X:
            xip_norms = np.linalg.norm(ele - self.mp)
        rp = xip_norms.max()
        for ele in self.X:
            xin_norms = np.linalg.norm(ele - self.mn)
        rn = xin_norms.max()
        lamb_da = (r / (rp + rn)) - (r / (rp + rn)) / 2.0
        return lamb_da * x + (1 - lamb_da) * self.m


    def initialization(self, kernel_type):
        # xi1 is the first positive value (image)
        xi1 = self.X[self.Ip[0]]
        xj1 = self.X[self.In[1]]
        xi1p = self.prime(xi1)
        xj1p = self.prime(xj1)
        self.A = self.kernel(xi1p, xi1p, kernel_type)
        self.B = self.kernel(xj1p, xj1p, kernel_type)
        self.C = self.kernel(xi1p, xj1p, kernel_type)
        self.D = np.zeros(len(self.X),dtype=float)
        self.E = np.zeros(len(self.X),dtype=float)
        for i in range(0, len(self.X)):
            xip = self.prime(self.X[i])
            self.D[i] = self.kernel(xip, xi1p, kernel_type)
            self.E[i] = self.kernel(xip, xj1p, kernel_type)
    def stop(self, eps):
        mi = []
        dinom = math.sqrt(self.A + self.B - 2 * self.C)
        for i in xrange(len(self.X)):
            if i in Ip:
                mi.append((self.D[i] - self.E[i] + self.B - self.C) / dinom)
            else:
                mi.append((self.E[i] - self.D[i] + self.A - self.C) / dinom)
        t = np.argmin(np.array(mi))
        if dinom - mi[t] < eps:
            return True, t
        return False, t


    def adapt(self,i,t):
        ktt = self.kernel(self.X[t],self.X[t],'P')
        if i in self.Ip:
            q = min(1,(self.A-self.D[t]+self.E[t]-self.C)/(self.A + ktt - 2*
                                                        self.D[t]-self.E[t]))
            self.alpha[i] = (1- q)*self.alpha[i] + q * utils.delta(i,t)
            self.A = self.A*(1-q)**2+2*(1-q)*q*self.D[t]+q**2*ktt
            self.C = (1-q)*self.C + q*self.E[t]
            for k in range(len(self.X)):
                self.D[i]=(1-q)*self.D[i]+q *self.kernel(
                    self.prime(self.X[i]),self.prime(self.X[t]),'P')
        else:
            q = min(1,(self.B-self.E[t]+self.D[t]-self.C)/(self.B + ktt - 2*
                                                        self.E[t]-self.D[t]))
            self.alpha[i] = (1- q)*self.alpha[i] + q * utils.delta(i,t)
            self.B = self.B*(1-q)**2+2*(1-q)*q*self.E[t]+q**2*ktt
            self.C = (1-q)*self.C + q*self.D[t]
            for k in range(len(self.X)):
                self.E[i]=(1-q)*self.D[E]+q *self.kernel(
                    self.prime(self.X[i]),self.prime(self.X[t]),'P')
