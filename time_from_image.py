from PIL import Image
import IPython.display

import os
import collections
import numpy as np
from sklearn import datasets, svm, metrics
import sklearn

class time_from_image:
    def __init__(self):

        self.random_state = 42
        self.training_size = 0.8
        
        self.directories = list()
        self.index_ranges = list()
        self.numbers = list()
        self.index_ranges.append((6740, 10**6))
        self.directories.append('D:/Users/Ashafix/Documents/GitHub/NeuroMario/movies_images/Super Mario Kart (USA)-3.bk2/')
        self.numbers.append(list())
        self.numbers[-1].append('01741')
        self.numbers[-1].append('01534')
        self.numbers[-1].append('01575')
        self.numbers[-1].append('02093')
        self.numbers[-1].append('01494')
        self.numbers[-1].append('12437')

        self.index_ranges.append((6559, 10**6))
        self.directories.append('D:/Users/Ashafix/Documents/GitHub/NeuroMario/movies_images/Super Mario Kart (USA)-5.bk2/')
        self.numbers.append(list())
        self.numbers[-1].append('01809')
        self.numbers[-1].append('01660')
        self.numbers[-1].append('01547')
        self.numbers[-1].append('01464')
        self.numbers[-1].append('01587')
        self.numbers[-1].append('12067')


        self.index_ranges.append((5601, 10**6))
        self.directories.append('D:/Users/Ashafix/Documents/GitHub/NeuroMario/movies_images/Super Mario Kart (USA)-6.bk2/')
        self.numbers.append(list())
        self.numbers[-1].append('01653')
        self.numbers[-1].append('01477')
        self.numbers[-1].append('01654')
        self.numbers[-1].append('01522')
        self.numbers[-1].append('01476')
        self.numbers[-1].append('11782')

        self.index_ranges.append((5788, 10**6))
        self.directories.append('D:/Users/Ashafix/Documents/GitHub/NeuroMario/movies_images/Super Mario Kart (USA)-7.bk2/')
        self.numbers.append(list())
        self.numbers[-1].append('01640')
        self.numbers[-1].append('01480')
        self.numbers[-1].append('02072')
        self.numbers[-1].append('01467')
        self.numbers[-1].append('01465')
        self.numbers[-1].append('12124')

        self.index_ranges.append((5746, 10**6))
        self.directories.append('D:/Users/Ashafix/Documents/GitHub/NeuroMario/movies_images/Super Mario Kart (USA)-8.bk2/')
        self.numbers.append(list())
        self.numbers[-1].append('01661')
        self.numbers[-1].append('01634')
        self.numbers[-1].append('01455')
        self.numbers[-1].append('01758')
        self.numbers[-1].append('01488')
        self.numbers[-1].append('11996')

        self.filenames = list()
        
        self.dig_pos = list()
        for i in range(5):
            self.dig_pos.append((0, i * 9, 8, i * 9 + 8))
            self.dig_pos.append((16, i * 9, 24, i * 9 + 8))
            self.dig_pos.append((24, i * 9, 32, i * 9 + 8))
            self.dig_pos.append((40, i * 9, 48, i * 9 + 8))
            self.dig_pos.append((48, i * 9, 56, i * 9 + 8))

        i = 52
        self.dig_pos.append((0, i, 8, i + 8))
        self.dig_pos.append((16, i, 24, i + 8))
        self.dig_pos.append((24, i, 32, i + 8))
        self.dig_pos.append((40, i, 48, i + 8))
        self.dig_pos.append((48, i, 56, i + 8))
        
        
        for movie, direc in enumerate(self.directories):
            self.filenames.append(list())
    
            for a, b, c in os.walk(direc):
        
                for filename in c:
        
                    if filename.endswith('.png'):
                        index = int(filename.split('_frame_')[1].split('.')[0])
                        if index >= (self.index_ranges[movie][0]) and index < self.index_ranges[movie][1]:
                            self.filenames[-1].append(os.path.join(direc, filename))
    def image_arrays_from_file(self, filename, movie, numbers):
        x = 112
        y = 37
        width = 60
        height = 60
        img = Image.open(filename).crop((x, y, x + width, y + height)).convert('LA')
        digits = list()
        digit_hashes = list()
        sorted_images = list()
        for i in range(10):
            digits.append(list())
            digit_hashes.append(list())
            
        for i, pos in enumerate(self.dig_pos):
            digit_array = np.array(img.crop(pos))[:,:,0]
            
            h = hash(digit_array.tostring())
            n = int(numbers[movie][i // 5][i % 5])
            if h not in self.digit_hashes[n]:
                self.digits[n].append(digit_array)
                self.digit_hashes[n].append(h)
                
            
            digits[n].append(digit_array)
            digit_hashes[n].append(h)
            sorted_images.append(digit_array)
                
        return digits, digit_hashes, sorted_images
        
        
    def filenames_to_digits(self, filenames, numbers):

        self.digits = list()
        self.digit_hashes = list()
        
        for i in range(10):
            self.digits.append(list())
            self.digit_hashes.append(list())
        
        
        a = False
        for movie, filename in enumerate(filenames):
            for file in filename:
                self.image_arrays_from_file(file, movie, numbers)
        #return self.digits
    
    def predict_time_from_filenames(self, filenames):
        if not self.classifier:
            self.classify()
        if isinstance(filenames, str):
            filenames = [filenames]
        
        solution = collections.defaultdict(int)
        for filename in filenames:
            x, xx, xxx = aa.image_arrays_from_file(filename, i, aa.numbers)
            solution[''.join([str(s)[0] for s in list(self.classifier.predict(np.array(xxx[-5:]).reshape(5, -1)))])] += 1
        
        return solution
    
    def timestring_to_time(self, timestring):
        
        t = 0
        t += int(timestring[0]) * 60
        t += int(timestring[1]) * 10
        t += int(timestring[2])
        t += int(timestring[3]) * 0.1
        t += int(timestring[4]) * 0.01
        return t
    def test_classifier(self):
        for i, filename in enumerate(self.filenames):
            prediction = aa.predict_time_from_filenames(filename)
            best_score = max([int(i) for i in list(prediction)])
            if best_score != int(aa.numbers[i][5]):
                print('failed')
                print('Expected: {}\nGot:{}'.format(aa.numbers[i][5], prediction))
                print(i)
                return False
        return True
    def classify(self):
        
        
        self.classifier = sklearn.svm.SVC(gamma=0.001)
        self.total_samples = 0
        for i in range(10):
            self.total_samples += len(self.digits[i])
            
        self.all_images = np.zeros([self.total_samples, 8, 8])
        self.all_digits = np.zeros([self.total_samples])
        index = 0
        for i in range(10):
            for d in self.digits[i]:
                self.all_images[index] = d
                self.all_digits[index] = i
                index += 1
        self.data_s, self.digits_s = sklearn.utils.shuffle(self.all_images.reshape(self.total_samples, -1), self.all_digits, random_state=self.random_state)
        self.classifier.fit(self.data_s[0:int(self.total_samples * self.training_size)], np.ravel(self.digits_s[0:int(self.total_samples * self.training_size)]))

        
        

