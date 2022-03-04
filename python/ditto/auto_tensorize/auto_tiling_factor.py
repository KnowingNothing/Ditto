import tvm
import numpy as np
import random

class ATFLogFile(object):
    """ATFLogFile object"""
    def __init__(self, file_name, mode):
        self.mode = mode
        self.file = open(file_name, mode)
    
    def close_file(self):
        self.file.close()

    def write_log(self, index, cost):
        assert self.mode == "w"
        self.file.write("%d %f\n" % (index, cost))
    
    def read_log(self):
        assert self.mode == "r"
        s = self.file.readline()
        if s == "":
            return -1, -1
        index, cost = s.split()
        index = int(index)
        cost = float(cost)
        return index, cost



class TensorizeParamChoices(dict):
    """TensorizeParamChoices dict"""
    def __init__(self, param_entry):
        self.param_entry = param_entry

    def build_param(self, index):
        param = {}
        for key, value in self.items():
            param[key] = value[index % len(value)]
            index = index // len(value)
        return self.param_entry(**param)
    
    def search_size(self):
        size = 1
        for key, value in self.items():
            size *= len(value)
        return size
    
    def __str__(self):
        s = "TensorizeParamChoices(\n"
        s += "  size = " + str(self.search_size()) + "\n"
        for key, value in self.items():
            s += "    " + key + " : " + str(value) + "\n"
        s += ")\n"
        return s

    
class ATFTask(object):
    """ATFTask object"""
    def __init__(self, auto_scheduler, tensors, **task_params):
        self.auto_scheduler = auto_scheduler
        self.tensors = tensors
        self.task_params = task_params

        # only one value with type TensorizeParamChoices
        self.tensorize_param_key = None
        for key, value in self.task_params.items():
            if isinstance(value, TensorizeParamChoices):
                if self.tensorize_param_key != None:
                    raise RuntimeError(
                        "Specified more than two TensorizeParamChoices!"
                    )
                self.tensorize_param_key = key
        if self.tensorize_param_key is None:
            raise RuntimeError(
                "Did not specify TensorizeParamChoices!"
            )
    
    def search_size(self):
        return self.task_params[self.tensorize_param_key].search_size()
    
    def get_task_params(self, index):
        ret_task_params = {}
        for key, value in self.task_params.items():
            ret_task_params[key] = value
        key = self.tensorize_param_key
        ret_task_params[key] = self.task_params[key].build_param(index)
        return ret_task_params


class ATFTuner(object):
    """ATFTuner (AutoTilingFactorTuner) object"""
    def __init__(self):
        pass


class DataDrivenATFTuner(object):
    """Tune with real input"""
    def __init__(self, task, dev, ctx, **timeeval_params):
        self.task = task
        self.dev = dev
        self.ctx = ctx
        self.timeeval_params = timeeval_params

        # for GridSearch only
        self.now_index = 0
    
    def next_batch(self, num = 1):
        raise NotImplementedError()
    
    def tune(self, n_trial, log_file = "ATFLog.log"):
        tensors_np = [
            np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype(y.dtype)
            for y in self.task.tensors
        ]
        tensors_tvm = [tvm.nd.array(x, self.dev) for x in tensors_np]
        log = ATFLogFile(log_file, "w")
        for _ in range(n_trial):
            index = self.next_batch()
            sch = self.task.auto_scheduler(**self.task.get_task_params(index))
            func = tvm.build(sch, self.task.tensors, self.ctx)
            evaluator = func.time_evaluator(func.entry_name, self.dev, **self.timeeval_params)
            cost = evaluator(*tensors_tvm).mean
            log.write_log(index, cost)
        log.close_file()
        pass

    def schedule(self, log_file = "ATFLog.log"):
        best_index = -1
        best_cost = -1
        log = ATFLogFile(log_file, "r")
        while 1:
            index, cost = log.read_log()
            if index == -1:
                break
            if best_cost == -1 or cost < best_cost:
                best_index, best_cost = index, cost
        log.close_file()
        return self.task.auto_scheduler(**self.task.get_task_params(best_index))

    def tune_and_schedule(self, n_trial, log_file = "ATFLog.log"):
        self.tune(n_trial, log_file)
        return self.schedule(log_file)


class GridSearchATFTuner(DataDrivenATFTuner):
    """Enumerate the search space in a grid search order"""
    def next_batch(self, num = 1):
        # TODO: support parallel tuner
        if num != 1:
            raise RuntimeError(
                "Only support serial tuner for now"
            )
        ret = self.now_index
        self.now_index = (self.now_index + 1) % self.task.search_size()
        return ret
        

class RandomATFTuner(DataDrivenATFTuner):
    """Enumerate the search space in a random order"""
    def next_batch(self, num = 1):
        return random.randint(0, self.task.search_size()-1)
