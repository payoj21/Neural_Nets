import logging
import pickle
from collections import OrderedDict
from tensor import Tensor

logging.basicConfig()
log = logging.getLogger("MLPNetwork")
log.setLevel(logging.DEBUG)

class MLPNetwork(object):

    def __init__(self):
        self._state = OrderedDict()

    def __setattr__(self, name, value):
        # Only register Tensors. In a model's state dictionary.
        if isinstance(value, Tensor):
            state = self.__dict__.get('_state')
            if state is not None:
                self._state[name] = value
            else:
                raise AttributeError(
                    "Cannot assign parameters before MLPNetowrk.__init__() call")
        object.__setattr__(self, name, value)


    def save_state(self, file_name):
        log.info("Saving model params.")

        blob = {}

        for param in self.__dict__:
            if isinstance(param, Tensor):
                blob[param] = self.__dict__.get(param).value
            else:
                blob[param] = self.__dict__.get(param)
            log.debug('{:s}'.format(param))

        try:
            with open(file_name, 'wb') as wfile:
                pickle.dump(dict(blobs = blob), wfile, pickle.HIGHEST_PROTOCOL)
        except IOError as ioe:
            log.error('I/O error({0}): {1}'.format(ioe.errno, ioe.strerror))

    def load_state(self, params_file):
        skip_list = ['state']
        loaded_params = []
        if not isinstance(params_file, str):
            raise ValueError(
                "File name must be a string."
            )

        log.info("Initializing model params from file: {}".format(params_file))

        blobs = None

        with open(params_file, 'rb') as rfile:
            blobs = pickle.load(rfile)

        if 'blobs' not in blobs:
            raise ValueError()

        log.info("Initializing weights using the file: {}".format(params_file))

        for param in self.__dict__:
            if param not in blobs['blobs'] or param in skip_list:
                log.info("Skipping param: {}".format(param))
            else:
                setattr(self, param, blobs['blobs'][param])
                loaded_params.append(param)
        log.info("Initialization complete.")
        log.info("Params loaded: {}".format(loaded_params))

    def state_dict(self):
        return self._state

    def update_state_dict(self):
        for param, _ in self.__dict__.items():
            if param in self._state: # only updates no additions.
                self._state[param] = self.__dict__.get(param)

    def zero_grad(self, opt):
        opt.zero_grad(self)

    def forward(self, inputs):
        """
            To be overwridden by the inheriting class.
            Any class members declared here will be registered by the
            model's state dictionary - _state.
            DO NOT TAMPER WITH THE _state dictionary.
        """
        raise NotImplementedError()

    def update(self, optimizer):
        raise NotImplementedError()
