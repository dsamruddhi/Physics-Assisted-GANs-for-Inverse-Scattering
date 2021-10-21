import os
import numpy as np
from scipy.io import loadmat


class Data:

    @staticmethod
    def check_data_sanctity(input, output):
        assert not np.isnan(input).any()
        assert not np.isnan(output).any()

    @staticmethod
    def get_files(filepath):
        files = os.listdir(filepath)
        files.sort(key=lambda x: int(x.strip(".mat")))
        return files

    @staticmethod
    def get_input_data(filepath):
        real_data = []
        imag_data = []
        files = Data.get_files(filepath)
        for file in files:
            filename = os.path.join(filepath, file)
            guess = loadmat(filename)["guess"]
            real_data.append(guess[0][0][0])
            imag_data.append(guess[0][0][1])
        return real_data, imag_data

    @staticmethod
    def get_output_data(filepath):
        scatterers = []
        files = Data.get_files(filepath)
        for file in files:
            filename = os.path.join(filepath, file)
            scatterer = loadmat(filename)["scatterer"]
            scatterers.append(scatterer)
        return scatterers

    @staticmethod
    def split_data(input, output, test_size):
        test_data_len = int(len(input) * test_size)
        train_data_len = len(input) - test_data_len
        input = np.asarray(input)
        output = np.asarray(output)

        train_input, train_output = input[:train_data_len, :, :, :], output[:train_data_len, :, :]
        test_input, test_output = input[train_data_len:, :, :, :], output[train_data_len:, :, :]
        return train_input, train_output, test_input, test_output

    @staticmethod
    def get_data(input_path, output_path, test_size=0.1):
        X_real, X_imag = Data.get_input_data(input_path)
        X = np.asarray([X_real, X_imag])
        X = np.moveaxis(X, 0, -1)
        y = Data.get_output_data(output_path)
        y = np.asarray(y)
        Data.check_data_sanctity(X, y)

        train_input, train_output, test_input, test_output = Data.split_data(X, y, test_size)
        return train_input, train_output, test_input, test_output
