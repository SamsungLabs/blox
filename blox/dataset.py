import re
import random
import pathlib
import functools
import collections.abc as cabc

from . import graph_utils
from . import search_space
from . import io as bio


preferred_ext = None


class _Dataset():
    def __init__(self, dataset_files, validate_data, db_type):
        if isinstance(dataset_files, str):
            dataset_files = [dataset_files]

        self.dbs = []
        self.header = None
        current_columns = None
        if db_type == 'training':
            self.seeds = []
            exts = []
            self.has_extension = False
            _ext_offset = None
        elif db_type == 'benchmarking':
            self.devices = []
        elif db_type == 'static':
            if len(dataset_files) != 1:
                raise ValueError('Expected exactly one dataste file')
        elif db_type == 'env':
            self.seeds = []

        for db_file in dataset_files:
            header, data = bio.read_dataset_file(db_file)
            if header.pop('format') == 'custom':
                header.pop('column_types')

            if header['dataset_type'] != db_type:
                raise ValueError(f'Expected a dataset file with {db_type} information')

            cols = header.pop('columns')
            cols_offset = 1
            cols_end = None
            if db_type == 'training':
                seed = header.pop('seed')
                part = header.pop('part')
                if part == 'base':
                    cols_end = _ext_offset
                elif part == 'ext':
                    if seed not in self.seeds:
                        raise ValueError('Extension to a training dataset should follow base file')
                    if _ext_offset is None:
                        _ext_offset = len(current_columns)
                        current_columns.extend(cols[1:])
                    cols_offset = _ext_offset
                else:
                    raise ValueError(f'Unknown part {part}')
            elif db_type == 'benchmarking':
                device = header.pop('device')
            elif db_type == 'env':
                seed = header.pop('seed')

            if self.header is None:
                self.header = header
                current_columns = cols

            if self.header != header or cols[0] != current_columns[0] or cols[1:] != current_columns[slice(cols_offset, cols_end)]:
                raise ValueError('Different dataset files contain data for different settings')

            # TODO: we could relax this if needed
            if db_type == 'training':
                if part == 'base':
                    if cols[:4] != ['model_hash', 'val_top1', 'test_top1', 'train_time_s']:
                        raise ValueError('In the current implementation we expect the dataset to contain information in order: model hash, top1 val accuracy, top1 test accuracy')
                elif part == 'ext':
                    if cols[:8] != ['model_hash', 'train_top1', 'train_loss', 'train_top5', 'val_loss', 'val_top5', 'test_loss', 'test_top5']:
                        raise ValueError('In the current implementation we expect the dataset to contain information in order: model hash, top1 train accuracy, followed by pairs of (loss, top5 accuracy) for validation and test')
                else:
                    raise ValueError(f'Unknown part! {part}')
            elif db_type == 'benchmarking':
                if cols[:2] != ['model_hash', 'latency']:
                    raise ValueError('In the current implementation we expect the dataset to contain information in order: model hash, latency')
            elif db_type == 'static':
                if cols[:4] != ['model_hash', 'params', 'flops', 'arch_vec']:
                    raise ValueError('In the current implementation we expect the dataset to contain information in order: model hash, number of parameters, number of FLOPs, architecture vector')
            elif db_type == 'env':
                pass

            if db_type == 'training':
                if part == 'base':
                    if seed in self.seeds:
                        raise ValueError(f'Duplicated seed! {seed}')
                    self.seeds.append(seed)
                    exts.append(False)
                elif part == 'ext':
                    exts[self.seeds.index(seed)] = True
            elif db_type == 'benchmarking':
                self.devices.append(device)
            elif db_type == 'env':
                self.seeds.append(seed)

            data_dict = { model_hash: rest for model_hash, *rest in data }
            if db_type == 'training' and part == 'ext':
                sidx = self.seeds.index(seed)
                for model_hash, ext in data_dict.items():
                    self.dbs[sidx][model_hash].extend(ext)
            else:
                self.dbs.append(data_dict)

        if not self.dbs:
            raise ValueError('At least one dataset should be read')

        self.header['columns'] = current_columns

        if db_type == 'training':
            assert exts
            self.has_extension = all(exts)

        if validate_data and len(self.dbs) > 1:
            if db_type == 'training':
                if not (all(exts) or not any(exts)):
                    raise ValueError('Inconsistent extensions! Some seeds have extra information but not all of them!')

            models = { model_hash: model_pt for model_hash, (*_, model_pt) in self.dbs[0].items() }
            for fidx, db in enumerate(self.dbs[1:]):
                if len(db) != len(models):
                    raise ValueError(f'Dataset file at position {fidx+1} has {len(db)} entries but the one at position 0 has {len(models)}')
                for model_hash, (*_, model_pt) in db.items():
                    if model_hash not in models:
                        raise ValueError(f'{model_hash} is present in dataset file {fidx+1} but no in 0')
                    if db_type == 'training':
                        # even if this is not true, the same model hash should guarantee that the architectures are the same
                        # however, internally we'd expect the points to be the same
                        assert model_pt == models[model_hash]

    @property
    def version(self):
        ''' Version of the dataset.
        '''
        return self.header['version']

    @property
    def search_space(self):
        ''' Search space shape. A (potentially nested) list of integers identifying
            different choices and their related number of options.
        '''
        return self.header['search_space']['shape']

    @property
    def ops(self):
        ''' List of the operations which were considered when creating the dataset.
        '''
        return self.header['search_space']['ops']

    @property
    def nodes(self):
        ''' Number of nodes which was considered when creating the dataset.
        '''
        return self.header['search_space']['nodes']

    @property
    def stages(self):
        ''' Number of stages which was considered when creating the dataset.
        '''
        return self.header['search_space']['stages']

    @property
    def columns(self):
        ''' Names of values stored in the dataset, in-order.
            Can be used to identify specific information from values returned by
            functions which do not convert their results to dictionaries.
            See the remaining API for more information.
        '''
        return self.header['columns']

    def __contains__(self, arch):
        h = search_space.get_model_hash(arch, ops=self.ops)
        return h in self.dbs[0]

    def __len__(self):
        return len(self.dbs[0])

    def __iter__(self):
        for k, v in self.dbs[0].items():
            yield (k,) + v


class StaticInfoDataset(_Dataset):
    def __init__(self, dataset_file):
        super().__init__([dataset_file], False, 'static')

    def _get(self, model_hash, return_dict):
        r = self.dbs[0].get(model_hash)
        if return_dict and r is not None:
            return dict(zip(self.columns[1:], r))
        return r

    def params(self, arch):
        ''' Return the number of parameters in a specific architecture.

            Arguments:
                arch - a point from the search space identifying a model

            Returns:
                ``None`` if information about a given ``arch`` cannot be found in the dataset,
                otherwise a ``dict`` or a ``list`` containing information about the model.
        '''
        model_hash = search_space.get_model_hash(arch, ops=self.ops)
        ret = self._get(model_hash, False)
        return ret[0]

    def flops(self, arch):
        ''' Return the number of FLOPs in a specific architecture.

            Arguments:
                arch - a point from the search space identifying a model

            Returns:
                ``None`` if information about a given ``arch`` cannot be found in the dataset,
                otherwise a ``dict`` or a ``list`` containing information about the model.
        '''
        model_hash = search_space.get_model_hash(arch, ops=self.ops)
        ret = self._get(model_hash, False)
        return ret[1]

    def arch_vec(self, arch):
        ''' Architecture vector used to train a model from the dataset.

            This can be different from the ``arch`` parameter due to graph
            isomorphism.

            Arguments:
                arch - a point from the search space identifying a model
            Returns:
                ``None`` if information about a given ``arch`` cannot be found in the dataset,
                otherwise a ``dict`` or a ``list`` containing information about the model.
        '''
        model_hash = search_space.get_model_hash(arch, ops=self.ops)
        ret = self._get(model_hash, False)
        return ret[2]


class EnvInfoDataset(_Dataset):
    def __init__(self, dataset_files):
        super().__init__(dataset_files, False, 'env')

    def _get(self, model_hash, seed, return_dict):
        r = self.dbs[self.seeds.index(seed)].get(model_hash)
        if return_dict and r is not None:
            return dict(zip(self.columns[1:], r))
        return r


class BenchmarkingDataset(_Dataset):
    ''' An object representing a queryable dataset containing benchmarking information
        of Nasbench-ASR models.
        The dataset is constructed by reading a set of pickle files containing
        information about models benchmarked on different devices.
        All the files used to create a single ``BenchmarkingDataset`` object should contian information
        about models coming from the same search space and can only differ by the type of device used.
        If you want to compare performance of models from different search spaces you'd need to create
        different objects for each case.
    '''
    def __init__(self, dataset_files, validate_data=True):
        ''' Create a new dataset by loading data from the provided list of files.
            If multiple files are given, they should contain information about models
            from the same search space, benchmarked on different devices.
            If ``validate_data`` is set to ``True``, the data from the files will be validated
            to check if it's consistent. If the files are known to be ok, the argument can be
            set to ``False`` to speed up loading time a little bit (or to hack the code if you know
            what you are doing).
        '''
        super().__init__(dataset_files, validate_data, 'benchmarking')

    def _get(self, model_hash, devices, ret_dict):
        if devices is None:
            devices = self.devices
            indices = list(range(len(self.devices)))
        else:
            if isinstance(devices, str):
                devices = [devices]
            indices = [self.devices.index(d) for d in devices]

        raw = [] if not ret_dict else {}
        for didx, device_name in zip(indices, devices):
            value = self.dbs[didx].get(model_hash)
            if value is None:
                return None
            if not ret_dict:
                raw.append(value)
            else:
                value = dict(zip(self.columns[1:], value))
                raw[device_name] = value

        return raw

    def latency(self, arch, devices=None, return_dict=False):
        ''' Return benchmarking information about a specific architecture on the provided
            devices from the dataset.

            Arguments:
                arch - a point from the search space identifying a model
                device - (optional) if provided, the returned will be information about
                    the model's performance when run on the device with the given name(s),
                    otherwise latency on all devices will be returned; accepted values are:
                    Str, List[Str] and None
                return_dict - (optional) determinates if the returned values will be provided
                    as a ``dict`` or a simple ``list``. A ``dict`` contains the same values as
                    the ``list`` but allows the user to extract them by their names, whereas
                    a list can be thought of as a single row in a table containing values only.
                    The user can map particular elements of the returned ``list`` by considering
                    the values in provided ``devices`` argument. Default: ``False``.

            Returns:
                ``None`` if information about a given ``arch`` cannot be found in the dataset,
                otherwise a ``dict`` or a ``list`` containing information about the model.

            Raises:
                ValueError - if invalid ``device`` is given 
        '''
        model_hash = search_space.get_model_hash(arch, ops=self.ops)
        return self._get(model_hash, devices, return_dict)


class Dataset(_Dataset):
    ''' An object representing a queryable NasBench-ASR dataset.

        The dataset is constructed by reading a set of pickle files containing training
        information about models using different configurations (different initialization
        seed and/or total number of epochs).
        The training information can be optionally extended with benchmarking and static
        (e.g. number of parameters) information.
        All the files used to create a single ``Dataset`` object should contian information
        about models trained in the same setting and can only differ by the initialization seed.
        If you want to compare performance of models in different settings, e.g. using full training
        or reduced training of 10 epochs, you'd need to create different objects for each case.
    '''
    def __init__(self, dataset_files,  devices_files=None, static_info=None, env_info=None, validate_data=True):
        ''' Create a new dataset by loading data from the provided list of files.
            If multiple files are given, they should contain information about models
            trained in the same setting, differing only by their initialization seed.
            If ``validate_data`` is set to ``True``, the data from the files will be validated
            to check if it's consistent. If the files are known to be ok, the argument can be
            set to ``False`` to speed up loading time a little bit (or to hack the code if you know
            what you are doing).
        '''
        super().__init__(dataset_files, validate_data, 'training')
        self.bench_info = None
        self.static_info = None
        self.env_info = None
        if devices_files:
            self.bench_info = BenchmarkingDataset(devices_files, validate_data=validate_data)
        if static_info:
            self.static_info = StaticInfoDataset(static_info)
        if env_info:
            self.env_info = EnvInfoDataset(env_info)

    @property
    def epochs(self):
        ''' Total number of epochs for which the models were trained when creating the dataset.
        '''
        return self.header['epochs']

    @property
    def dataset(self):
        ''' Dataset used to train and validate models
        '''
        return self.header['dataset']

    @property
    def is_extended(self):
        ''' Whether the dataset has been extended with extra training information.
        '''
        return self.has_extension

    def _get_raw_info(self, seed_idx, model_hash):
        raw = self.dbs[seed_idx].get(model_hash)
        if raw is None:
            return None
        return [model_hash] + list(raw) + [self.seeds[seed_idx]]

    def _get_info_dict(self, seed_idx, model_hash):
        raw = self.dbs[seed_idx].get(model_hash)
        if raw is not None:
            raw = dict(zip(self.columns[1:], raw))
            raw[self.columns[0]] = model_hash
            raw['seed'] = self.seeds[seed_idx]
        return raw

    def _get_info(self, seed_idx, model_hash, return_dict):
        if return_dict:
            return self._get_info_dict(seed_idx, model_hash)
        else:
            return self._get_raw_info(seed_idx, model_hash)

    def _query(self, model_hash, seed, devices, include_static_info, include_env_info, return_dict):
        if seed is None:
            seed_idx = random.randrange(len(self.seeds))
        else:
            seed_idx = self.seeds.index(seed)

        ret = self._get_info(seed_idx, model_hash, return_dict)
        if devices != False and (devices is not None or self.bench_info):
            if not self.bench_info:
                raise ValueError('No benchmarking information attached')
            lat = self.bench_info._get(model_hash, devices, return_dict)
            if return_dict:
               ret.update(lat)
            else:
                ret.extend(lat)

        if include_static_info != False and (include_static_info is not None or self.static_info):
            if not self.static_info:
                raise ValueError('No static information attached')
            info = self.static_info._get(model_hash, return_dict)
            if return_dict:
                ret['info'] = info
            else:
                ret.append(info)

        if include_env_info != False and (include_env_info is not None or self.env_info):
            if not self.env_info:
                raise ValueError('No environment information attached')
            info = self.env_info._get(model_hash, self.seeds[seed_idx], return_dict)
            if return_dict:
                ret['env'] = info
            else:
                ret.append(info)

        return ret

    def full_info(self, arch, seed=None, devices=None, include_static_info=None, include_env_info=None, return_dict=True):
        ''' Return all information about a specific architecture from the dataset.

            If multiple seeds are available, the can either return information about
            a specific one or a random one.

            Arguments:
                arch - a point from the search space identifying a model
                seed - (optional) if provided, the returned will be information about
                    the model's performance when initialized with this particular seed,
                    otherwise information related to a randomly chosen seed from the list
                    if available ones will be used. Default: random seed
                devices - (optional) add information about benchmarking on the provided devices,
                    if ``None`` all available devices are included, otherwise should be a name of
                    the device or a list of names, can also be exactly ``False`` to avoid including
                    benchmarking information even when they are available
                include_static_info - (optional) include static information about the model,
                    such as number of parameters, if set to ``None`` static information will be
                    added only if available
                return_dict - (optional) determinates if the returned values will be provided
                    as a ``dict`` or a simple ``list``. A ``dict`` contains the same values as
                    the ``list`` but alolws the user to extract them by their names, whereas
                    a list can be thought of as a single row in a table containing values only.
                    The user can map particular elements of the returned ``list`` by considering
                    the values in ``columns``. Default: ``True``.

            Returns:
                ``None`` if information about a given ``arch`` cannot be found in the dataset,
                otherwise a ``dict`` or a ``list`` containing information about the model.

            Raises:
                ValueError - if invalid ``seed`` is given 
        '''
        model_hash = search_space.get_model_hash(arch, ops=self.ops)
        return self._query(model_hash, seed, devices, include_static_info, include_env_info, return_dict)

    def full_info_by_graph(self, graph, seed=None, devices=None, include_static_info=None, include_env_info=None, return_dict=True):
        ''' Return all information about an architecture identified by the provided model
            graph.

            If multiple seeds are available, the can either return information about
            a specific one or a random one.

            Arguments:
                graph - a graph of a model from the search space, obtained by calling
                    ``nasbench_asr.graph_utils.get_model_graph(arch)``
                seed - (optional) if provided, the returned will be information about
                    the model's performance when initialized with this particular seed,
                    otherwise information related to a randomly chosen seed from the list
                    if available ones will be used. Default: random seed
                devices - (optional) add information about benchmarking on the provided devices,
                    if ``None`` all available devices are included, otherwise should be a name of
                    the device or a list of names, can also be exactly ``False`` to avoid including
                    benchmarking information even when they are available
                include_static_info - (optional) include static information about the model,
                    such as number of parameters, if set to ``None`` static information will be
                    added only if available
                return_dict - (optional) determinates if the returned values will be provided
                    as a ``dict`` or a simple ``list``. A ``dict`` contains the same values as
                    the ``list`` but allows the user to extract them by their names, whereas
                    a list can be thought of as a single row in a table containing values only.
                    The user can map particular elements of the returned ``list`` by considering
                    the values in ``columns``. Default: ``True``.

            Returns:
                ``None`` if information about a given ``arch`` cannot be found in the dataset,
                otherwise a ``dict`` or a ``list`` containing information about the model.

            Raises:
                ValueError - if invalid ``seed`` is given 
        '''
        model_hash = graph_utils.graph_hash(graph)
        return self._query(model_hash, seed, devices, include_static_info, include_env_info, return_dict)

    def test_acc(self, arch, seed=None):
        ''' Return test accuracy of a model.

            Test accuracy is currently defined as the test accuracy of the model at epoch
            with the lowest validation accuracy.

            Arguments:
                arch - a point from the search space identifying a model
                seed - (optional) an initialization seed to use, if not provided information
                    will be queried for a random seed (default: ``None``)

            Returns:
                ``None`` if the dataset does not contain information about a model ``arch``,
                otherwise a scalar ``float``.
        '''
        info = self.full_info(arch, seed=seed, devices=False, include_static_info=False, include_env_info=False, return_dict=False)
        if info is None:
            return None
        return info[2]

    def val_acc(self, arch, epoch=None, best=True, seed=None):
        ''' Return validation accuracy of a model.

            The returned accuracy can be either the best accuracy or the accuracy at the last epoch.
            The maximum number of epochs to consider can be controlled by ``epoch``.
            If ``vals`` is a list of validation accuracies, the returned value can be
            defined as:
                epoch = epoch if epoch is not None else len(vals)
                return max(vals[:epoch]) if best else vals[epoch-1]

            Arguments:
                arch - a point from the search space identifying a model
                epoch - (optional) number of epochs to consider, if not provided
                    all epochs will be considered (default: ``None``)
                best - (optional) return best validation accuracy from epoch 1 to the
                    maximum considered epochs, otherwise return accuracy at the last
                    considered epoch (default: ``True``)
                seed - (optional) an initialization seed to use, if not provided information
                    will be queried for a random seed (default: ``None``)
        '''
        info = self.full_info(arch, seed=seed, devices=False, include_static_info=False, include_env_info=False, return_dict=False)
        if info is None:
            return None
        if epoch is None:
            epoch = len(info[1])
        if best:
            return max(info[1][:epoch])
        else:
            return info[1][epoch-1]

    def train_time(self, arch, epoch=None, seed=None):
        ''' Return training time of a model in seconds.

            The returned time can be either the full training time, as stored in the dataset,
            or time required for only ``epoch`` epochs of training in which case the value
            from the dataset is linearly interpolated.

            Arguments:
                arch - a point from the search space identifying a model
                epoch - (optional) number of epochs to consider, if not provided
                    training time of full training will be returned, otherwise
                    a linear interpolation will be done (default: ``None``)
                seed - (optional) an initialization seed to use, if not provided information
                    will be queried for a random seed (default: ``None``)
        '''
        info = self.full_info(arch, seed=seed, devices=False, include_static_info=False, include_env_info=False, return_dict=False)
        if info is None:
            return None
        if epoch is None:
            return info[3]
        return info[3] * (epoch / self.epochs)

    def train_acc(self, arch, epoch=None, best=True, seed=None):
        ''' Return training accuracy of a model.

            > **Note**: this information is only available if using extended dataset

            The returned accuracy can be either the best accuracy or the accuracy at the last epoch.
            The maximum number of epochs to consider can be controlled by ``epoch``.
            If ``vals`` is a list of training accuracies, the returned value can be
            defined as:
                epoch = epoch if epoch is not None else len(vals)
                return max(vals[:epoch]) if best else vals[epoch-1]

            Arguments:
                arch - a point from the search space identifying a model
                epoch - (optional) number of epochs to consider, if not provided
                    all epochs will be considered (default: ``None``)
                best - (optional) return best training accuracy from epoch 1 to the
                    maximum considered epochs, otherwise return accuracy at the last
                    considered epoch (default: ``True``)
                seed - (optional) an initialization seed to use, if not provided information
                    will be queried for a random seed (default: ``None``)
        '''
        if not self.is_extended:
            raise ValueError('Information require extended training dataset')

        info = self.full_info(arch, seed=seed, devices=False, include_static_info=False, include_env_info=False, return_dict=False)
        if info is None:
            return None
        if epoch is None:
            epoch = len(info[4])
        if best:
            return max(info[4][:epoch])
        else:
            return info[4][epoch-1]

    def train_loss(self, arch, epoch=None, best=False, seed=None):
        ''' Return training loss of a model.

            > **Note**: this information is only available if using extended dataset

            The returned loss can be either the best loss or the loss at the last epoch.
            The maximum number of epochs to consider can be controlled by ``epoch``.
            If ``vals`` is a list of training loss values, the returned value can be
            defined as:
                epoch = epoch if epoch is not None else len(vals)
                return min(vals[:epoch]) if best else vals[epoch-1]

            Arguments:
                arch - a point from the search space identifying a model
                epoch - (optional) number of epochs to consider, if not provided
                    all epochs will be considered (default: ``None``)
                best - (optional) return best training loss from epoch 1 to the
                    maximum considered epochs, otherwise return loss at the last
                    considered epoch (default: ``False``)
                seed - (optional) an initialization seed to use, if not provided information
                    will be queried for a random seed (default: ``None``)
        '''
        if not self.is_extended:
            raise ValueError('Information require extended training dataset')

        info = self.full_info(arch, seed=seed, devices=False, include_static_info=False, include_env_info=False, return_dict=False)
        if info is None:
            return None
        if epoch is None:
            epoch = len(info[5])
        if best:
            return min(info[5][:epoch])
        else:
            return info[5][epoch-1]

    def train_top5(self, arch, epoch=None, best=True, seed=None):
        ''' Return training top5 accuracy of a model.

            > **Note**: this information is only available if using extended dataset

            The returned top5 accuracy can be either the best top5 accuracy or the top5
            accuracy at the last epoch.
            The maximum number of epochs to consider can be controlled by ``epoch``.
            If ``vals`` is a list of training top5 accuracies, the returned value can be
            defined as:
                epoch = epoch if epoch is not None else len(vals)
                return max(vals[:epoch]) if best else vals[epoch-1]

            Arguments:
                arch - a point from the search space identifying a model
                epoch - (optional) number of epochs to consider, if not provided
                    all epochs will be considered (default: ``None``)
                best - (optional) return best training top5 accuracy from epoch 1 to the
                    maximum considered epochs, otherwise return top5 accuracy at the last
                    considered epoch (default: ``False``)
                seed - (optional) an initialization seed to use, if not provided information
                    will be queried for a random seed (default: ``None``)
        '''
        if not self.is_extended:
            raise ValueError('Information require extended training dataset')

        info = self.full_info(arch, seed=seed, devices=False, include_static_info=False, include_env_info=False, return_dict=False)
        if info is None:
            return None
        if epoch is None:
            epoch = len(info[6])
        if best:
            return max(info[6][:epoch])
        else:
            return info[6][epoch-1]

    def val_loss(self, arch, epoch=None, best=False, seed=None):
        ''' Return validation loss of a model.

            > **Note**: this information is only available if using extended dataset

            The returned loss can be either the best loss or the loss at the last epoch.
            The maximum number of epochs to consider can be controlled by ``epoch``.
            If ``vals`` is a list of validation loss values, the returned value can be
            defined as:
                epoch = epoch if epoch is not None else len(vals)
                return min(vals[:epoch]) if best else vals[epoch-1]

            Arguments:
                arch - a point from the search space identifying a model
                epoch - (optional) number of epochs to consider, if not provided
                    all epochs will be considered (default: ``None``)
                best - (optional) return best validation loss from epoch 1 to the
                    maximum considered epochs, otherwise return loss at the last
                    considered epoch (default: ``False``)
                seed - (optional) an initialization seed to use, if not provided information
                    will be queried for a random seed (default: ``None``)
        '''
        if not self.is_extended:
            raise ValueError('Information require extended training dataset')

        info = self.full_info(arch, seed=seed, devices=False, include_static_info=False, include_env_info=False, return_dict=False)
        if info is None:
            return None
        if epoch is None:
            epoch = len(info[7])
        if best:
            return min(info[7][:epoch])
        else:
            return info[7][epoch-1]

    def val_top5(self, arch, epoch=None, best=True, seed=None):
        ''' Return validation top5 accuracy of a model.

            > **Note**: this information is only available if using extended dataset

            The returned top5 accuracy can be either the best top5 accuracy or the top5
            accuracy at the last epoch.
            The maximum number of epochs to consider can be controlled by ``epoch``.
            If ``vals`` is a list of validation top5 accuracies, the returned value can be
            defined as:
                epoch = epoch if epoch is not None else len(vals)
                return max(vals[:epoch]) if best else vals[epoch-1]

            Arguments:
                arch - a point from the search space identifying a model
                epoch - (optional) number of epochs to consider, if not provided
                    all epochs will be considered (default: ``None``)
                best - (optional) return best validation top5 accuracy from epoch 1 to the
                    maximum considered epochs, otherwise return top5 accuracy at the last
                    considered epoch (default: ``False``)
                seed - (optional) an initialization seed to use, if not provided information
                    will be queried for a random seed (default: ``None``)
        '''
        if not self.is_extended:
            raise ValueError('Information require extended training dataset')

        info = self.full_info(arch, seed=seed, devices=False, include_static_info=False, include_env_info=False, return_dict=False)
        if info is None:
            return None
        if epoch is None:
            epoch = len(info[8])
        if best:
            return max(info[8][:epoch])
        else:
            return info[8][epoch-1]

    def test_loss(self, arch, seed=None):
        ''' Return test loss of a model.

            > **Note**: this information is only available if using extended dataset

            The returned loss can be either the best loss or the loss at the last epoch.
            The maximum number of epochs to consider can be controlled by ``epoch``.
            If ``vals`` is a list of test loss values, the returned value can be
            defined as:
                epoch = epoch if epoch is not None else len(vals)
                return min(vals[:epoch]) if best else vals[epoch-1]

            Arguments:
                arch - a point from the search space identifying a model
                epoch - (optional) number of epochs to consider, if not provided
                    all epochs will be considered (default: ``None``)
                best - (optional) return best test loss from epoch 1 to the
                    maximum considered epochs, otherwise return loss at the last
                    considered epoch (default: ``False``)
                seed - (optional) an initialization seed to use, if not provided information
                    will be queried for a random seed (default: ``None``)
        '''
        if not self.is_extended:
            raise ValueError('Information require extended training dataset')

        info = self.full_info(arch, seed=seed, devices=False, include_static_info=False, include_env_info=False, return_dict=False)
        if info is None:
            return None
        return info[9]

    def test_top5(self, arch, seed=None):
        ''' Return test top5 accuracy of a model.

            > **Note**: this information is only available if using extended dataset

            The returned top5 accuracy can be either the best top5 accuracy or the top5
            accuracy at the last epoch.
            The maximum number of epochs to consider can be controlled by ``epoch``.
            If ``vals`` is a list of test top5 accuracies, the returned value can be
            defined as:
                epoch = epoch if epoch is not None else len(vals)
                return max(vals[:epoch]) if best else vals[epoch-1]

            Arguments:
                arch - a point from the search space identifying a model
                epoch - (optional) number of epochs to consider, if not provided
                    all epochs will be considered (default: ``None``)
                best - (optional) return best test top5 accuracy from epoch 1 to the
                    maximum considered epochs, otherwise return top5 accuracy at the last
                    considered epoch (default: ``False``)
                seed - (optional) an initialization seed to use, if not provided information
                    will be queried for a random seed (default: ``None``)
        '''
        if not self.is_extended:
            raise ValueError('Information require extended training dataset')

        info = self.full_info(arch, seed=seed, devices=False, include_static_info=False, include_env_info=False, return_dict=False)
        if info is None:
            return None
        return info[10]

    @functools.wraps(BenchmarkingDataset.latency)
    def latency(self, *args, **kwargs):
        if not self.bench_info:
            raise ValueError('No benchmarking information attached')

        return self.bench_info.latency(*args, **kwargs)

    @functools.wraps(StaticInfoDataset.params)
    def params(self, *args, **kwargs):
        if not self.static_info:
            raise ValueError('No static information attached')

        return self.static_info.params(*args, **kwargs)

    @functools.wraps(StaticInfoDataset.flops)
    def flops(self, *args, **kwargs):
        if not self.static_info:
            raise ValueError('No static information attached')

        return self.static_info.flops(*args, **kwargs)

    @functools.wraps(StaticInfoDataset.arch_vec)
    def arch_vec(self, *args, **kwargs):
        if not self.static_info:
            raise ValueError('No static information attached')

        return self.static_info.arch_vec(*args, **kwargs)

    def __iter__(self):
        for k in self.dbs[0].keys():
            yield self[k]

    def __getitem__(self, k):
        return [k] + [
            None if not part else [db[k] for db in part.dbs]
            for part in [self, self.static_info, self.bench_info, self.env_info]
        ]


def from_folder(folder, extended=False, seeds=None, devices=None, include_static_info=False, include_env_info=False, validate_data=True, preferred_ext=None):
    ''' Create a ``Dataset`` object from files in a given directory.

        Arguments control what subset of the files will be used.
        Recognizable files should have names following the pattern::
            - blox-base-{seed}.pickle for base information for training datasets
            - blox-ext-{seed}.pickle for extended information for training datasets
            - blox-bench-{device}.pickle for benchmarking datasets
            - blox-info.pickle for static information dataset
            - blox-env-{seed}.pickle for information about training environment of each model

        Arguments:
            extended - whether to include information from the extended training
                dataset. The base dataset only contains top1 validation accuracy
                for all epochs and the final test top1 accuracy, as well as training time
                for each model. Extended dataset extends this with loss and top5 values,
                and also includes information about performance on the training set (the same 3 values).
            seeds - if not provided the created dataset will use all available
                seeds (each file should hold information about one seed only).
                Otherwise it can be a single value or a list seeds to use.
                The function will not check if the file(s) for the provided seed(s)
                exist(s) and will fail silently (i.e., the resulting
                dataset simply won't include results for the provided seed)
            devices - (optional) add information about benchmarking on the provided devices,
                if ``None`` all available devices are included, otherwise should be a name of
                the device or a list of names, can also be exactly ``False`` to avoid including
                benchmarking information even when they are available
            include_static_info - (optional) include static information about the model,
                such as number of parameters
            include env_info - (optional) include information about training environment for each
                model.
            validate_data - passed to ``Dataset`` constructor, if ``True`` the dataset
                will be validated to check consistency of the data. Can be set to ``False``
                to speed up loading if the data is known to be valid.
            preferred_ext - an optional list of extensions defining the preferred order in which
                different versions of the dataset will be loaded. Possible choices include::
                    - ".blox" is the custom format optimised for storage size and loading speed,
                      however it requires availability of the C backend for fast decoding (the backend
                      is shipped together with the rest of the package and should be combined as a python
                      module during installation, to verify it's availability run:
                      import blox.io as bio; print(bio.has_C))
                    - ".pickle" is using standard pickle to (de)serialize dataset, its featuring decent 
                      loading times but requires more storage than ".blox"; on the up side, it does not
                      require custom decoding, on the down side it can be considered a security thread
                      in some cases
                    - ".csv" is widely compatible and safe to load, but it's also the least efficient
                      one out of the three

                if the argument is present, the underlying mechanism will always prefer loading files with
                relevant extensions first, falling back to extensions later in the list independently for
                each file it might need to load. If an extension is not specified in the list, it will never
                be used - consequently, passing an empty list will always result in a likely error.
                If the argument is not specified (``None``), value of the global variable 
                `blox.dataset.preferred_ext` is used (defaults to ``None``).
                If the value is still ``None``, it defaults to ``[blox, pickle, csv]`` if `blox.io.has_C`
                evaluates to ``True``, and ``[pickle, csv, blox]`` otherwise

                (Note: values of extensions can also be specified without the leading dot)
                (Note: a single string can also be passed instead of a list, it is understood then as a singleton list)

        Raises:
            ValueError - if ``folder`` is not a directory or does not exist
            ValueError - if any of the loaded dataset files contain 
    '''
    if preferred_ext is None:
        preferred_ext = globals().get('preferred_ext', None)
        if preferred_ext is None:
            if bio.has_C:
                preferred_ext = ['.blox', '.pickle', '.csv']
            else:
                preferred_ext = ['.pickle', '.csv', '.blox']

    if isinstance(preferred_ext, str):
        preferred_ext = [preferred_ext]

    preferred_ext = [ext if ext.startswith('.') else '.'+ext for ext in preferred_ext]

    f = pathlib.Path(folder).expanduser()
    if not f.exists() or not f.is_dir():
        raise ValueError(f'{folder} is not a directory')

    if seeds is not None:
        if isinstance(seeds, cabc.Sequence) and not isinstance(seeds, str):
            seeds = '(' + '|'.join(map(str, seeds)) + ')'
        else:
            seeds = str(seeds)
    else:
        seeds = '[0-9]+'

    if devices != False:
        if devices is not None:
            if isinstance(devices, cabc.Sequence) and not isinstance(devices, str):
                devices = '(' + '|'.join(map(str, devices)) + ')'
            else:
                devices = str(devices)
        else:
            devices = '[a-zA-Z0-9-]+'

    bases = []
    exts = []
    bench_info = []
    static_info = None
    env_info = []

    files = []

    for ff in f.iterdir():
        if ff.is_file():
            files.append(ff)

    matches = set()
    for ext in preferred_ext:
        for ff in files:
            raw_name = ff.with_suffix('').name
            if raw_name in matches:
                continue

            regex_b = re.compile(f'blox-base-{seeds}\\{ext}')
            regex_e = re.compile(f'blox-ext-{seeds}\\{ext}') if extended else None
            regex2 = re.compile(f'blox-bench-{devices}\\{ext}') if devices else None
            regex_env = re.compile(f'blox-env-{seeds}\\{ext}') if include_env_info else None

            if regex_b.fullmatch(ff.name):
                bases.append(str(ff))
                matches.add(raw_name)
            if extended and regex_e.fullmatch(ff.name):
                exts.append(str(ff))
                matches.add(raw_name)
            if devices and regex2.fullmatch(ff.name):
                bench_info.append(str(ff))
                matches.add(raw_name)
            if include_static_info and ff.name == 'blox-info.pickle':
                static_info = str(ff)
                matches.add(raw_name)
            if include_env_info and regex_env.fullmatch(ff.name):
                env_info.append(str(ff))
                matches.add(raw_name)

    print(bases + exts, bench_info, static_info, env_info)
    return Dataset(bases + exts, bench_info, static_info, env_info, validate_data=validate_data)
