import numpy as np

from dwave.system.samplers import DWaveSampler

# Numpy arrays should be row-major for best performance


class TrainEnv:
    # require for the data to be passed as NP data arrays
    # -> clear to both user and code what each array is
    # X_train, y_train should be formatted like scikit-learn data arrays are
    # -> X_train is the train data, y_train is the train labels
    """
    Contains the model object (the output of the machine learning).

    TODO:
    """

    def __init__(
        self,
        X_train,
        y_train,
        endpoint_url,
        account_token,
        X_val=None,
        y_val=None,
        fidelity=7,
        dwave_topology="pegasus",
    ):
        """
        Configures the hyperparameters for the model. In essence, this controls how
        the model learns.

        Parameters:
        - X_train           Input Events x Params dataset (given in Scikit-learn's format).
        - y_train           Input Classification dataset (given in Scikit-learn's format, should have
                            one class as -1 and the other as 1).
        - endpoint_url      The url associated with the D-Wave machine desired.
        - account_token     Access token for D-Wave machines associated with your account.
        - X_val             (Optional) Validation Events x Params dataset
                            (given in Scikit-learn's format).
        - y_val             (Optional) Validation Classification dataset
                            (given in Scikit-learn's format).
        - fidelity          (Optional) Number of copies of parameter to make for zooming.
        - dwave_topology    (Optional) Architecture of the desired D-Wave machine.
                            (Possible options defined in D-Wave documentation.)

        Environment Vars:
        - X_train           Dataset of Events x Params used solely for training.
        - y_train           Dataset of Classifcations used solely for training.
        - X_val             Dataset of Events x Params used solely for validation.
        - y_val             Dataset of Classifcations used solely for validation.
        - train_size        Number of events to train on (or per group if dataset is split).
        - fidelity          Number of copies of each parameter to make. The greater the fidelity,
                            the more complex of a decision boundary that could be formed.
        - fidelity_offset   Amount to shift each copy of a param. Should generally not be changed.
        - c_i               Input dataset after the param copies have been created and shifted.
        - C_i               c_i dotted with y_train row-wise.
        - C_ij              c_i dotted with itself row-wise.
        - sampler           Defines the characteristics for the desired D-Wave machine. Can be changed
                            after understanding the Ocean SDK.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        if X_val is None:
            self.create_val_data()

        self.train_size = np.shape(self.X_train)[0]

        self.fidelity = fidelity
        self.fidelity_offset = 0.0225 / fidelity

        self.c_i = None
        self.C_i = None
        self.C_ij = None
        self.train_preprocess()

        self.c_i_val = None
        self.val_preprocess()

        self.dwave_topology = dwave_topology
        self.sampler = DWaveSampler(
            endpoint=endpoint_url,
            token=account_token,
            solver=dict(topology__type=dwave_topology),
            auto_scale=True,
        )  # auto_scale set True by default

    def create_val_data(self):
        """
        Takes a small portion of the training data for validation (useful for
        comparing performance of error-correction schemes).
        """
        dummy_xt, dummy_xv = np.split(
            self.X_train, [int(8 * np.size(self.X_train, 0) / 10)], 0
        )
        dummy_yt, dummy_yv = np.split(
            self.y_train, [int(8 * np.size(self.y_train) / 10)]
        )

        self.X_train, self.X_val = np.array(list(dummy_xt)), np.array(list(dummy_xv))
        self.y_train, self.y_val = np.array(list(dummy_yt)), np.array(list(dummy_yv))

    def train_preprocess(self):
        """
        This duplicates the parameters 'fidelity' times. The purpose is to turn the weak classifiers
        from outputing a single number (-1 or 1) to outputting a binary array ([-1, 1, 1,...]). The
        use of such a change is to trick the math into allowing more nuance between a weak classifier
        that outputs 0.1 from a weak classifier that outputs 0.9 (the weak classifier outputs are continuous)
        -> thereby discretizing the weak classifier's decision into more pieces than binary.

        This then creates a periodic array to shift the outputs of the repeated weak classifier, so that there
        is a meaning to duplicating them. You can think of each successive digit of the resulting weak classifier
        output array as being more specific about what the continuous output was - ie >0, >0.1, >0.2 etc. This
        description is not exactly correct in this case but it is the same idea as what we're doing.
        """
        m_events, n_params = np.shape(
            self.X_train
        )  # [M events (rows) x N parameters (columns)]

        c_i = np.repeat(
            self.X_train, repeats=self.fidelity, axis=1
        )  # [M events (rows) x N*fidelity parameters (columns)]

        offset_array = self.fidelity_offset * (
            np.tile(np.arange(self.fidelity), m_events * n_params) - self.fidelity // 2
        )
        c_i = np.sign(np.ndarray.flatten(c_i, order="C") - offset_array) / (
            n_params * self.fidelity
        )
        c_i = np.reshape(c_i, (m_events, n_params * self.fidelity))
        self.c_i = c_i

        C_i = np.einsum("i, ij", self.y_train, c_i)
        C_ij = np.einsum("ji,jk", c_i, c_i)

        self.C_i, self.C_ij = C_i, C_ij

    def val_preprocess(self):  # added with val
        """
        Same as above but for validation data.
        """
        m_events, n_params = np.shape(
            self.X_val
        )  # [M events (rows) x N parameters (columns)]

        c_i = np.repeat(
            self.X_val, repeats=self.fidelity, axis=1
        )  # [M events (rows) x N*fidelity parameters (columns)]

        offset_array = self.fidelity_offset * (
            np.tile(np.arange(self.fidelity), m_events * n_params) - self.fidelity // 2
        )
        c_i = np.sign(np.ndarray.flatten(c_i, order="C") - offset_array) / (
            n_params * self.fidelity
        )
        c_i = np.reshape(c_i, (m_events, n_params * self.fidelity))

        self.c_i_val = c_i
