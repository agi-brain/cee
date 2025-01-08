import yaml
import os
import torch
from deepmerge import Merger
import GPUtil


class Config(dict):
    def __init__(self, name='default', d=None, add_device=True):
        if d:
            self._load_from_dict(d)
        else:
            path = os.path.join('', f'{name}.yaml')
            self._load_config(path)
        if add_device:
            if torch.cuda.is_available():
                deviceIds = GPUtil.getFirstAvailable(order='memory', maxLoad=0.8, maxMemory=0.8)
                self['device'] = torch.device(f'cuda:{deviceIds[0]}')
            else:
                self['device'] = torch.device('cpu')

    def save_partial(self, path, keys=None, **kwargs):
        if keys:
            dict_file = dict((k, self[k]) for k in keys)
        else:
            dict_file = dict(self)
        dict_file.update(kwargs)

        with open(os.path.join(path), 'w') as f:
            yaml.dump(dict_file, f, default_flow_style=False)

    def merge(self, d: dict, override=False):
        '''
        :param d: dictionary to merge into config
        :param override: when true, will override any values that are already in config
        The function will always add values that are not in the original config
        '''
        merger = Merger(
            # pass in a list of tuple, with the
            # strategies you are looking to apply
            # to each type.
            [
                (list, ["override"]),
                (dict, ["merge"])
            ],
            # next, choose the fallback strategies,
            # applied to all other types:
            ["override"],
            # finally, choose the strategies in
            # the case where the types conflict:
            ["override"]
        )
        if override:
            self._load_from_dict(merger.merge(self, d))
        else:
            self._load_from_dict(merger.merge(d, self))

    def _load_config(self, path):
        print(path)
        with open(path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            self._load_from_dict(data)

    def _load_from_dict(self, d):
        for key, val in d.items():
            if key == 'device':
                continue
            if type(val) == str:
                if val.lower() == 'nan' or val.lower() == 'none':
                    val = None
                elif val == 'empty_dict':
                    val = dict()
            elif type(val) == dict and val.keys():
                val = Config(d=val, add_device=False)
            self[key] = val

    @staticmethod
    def _override_config(config, args):
        for key in config.keys():
            if type(config[key]) == Config:
                Config._override_config(config[key], args)
            elif key in args.keys():
                val = args[key]
                if val and val != "None" and val != -1 and val != "empty_dict":
                    config[key] = val

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<Config ' + dict.__repr__(self) + '>'
