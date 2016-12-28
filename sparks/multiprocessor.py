from multiprocessing import Event, JoinableQueue, Process, Queue
from queue import Empty, Full
from time import sleep

class MultiProcessor(object):
    def __init__(self, 
                    functions, 
                    output_names=['val'], 
                    initializator=None, 
                    initializator_args={}, 
                    max_size=1024, 
                    threads_num=8, 
                    update=False,
                    resources_demanded=1,
                    mode="consumer",
                    counter=None):

        self.functions = functions
        self.output_names = output_names
        self.initializator = initializator
        self.initializator_args = initializator_args
        self.threads_num = threads_num
        self.update = update
        self.resources_demanded = resources_demanded
        self.mode = mode
        self.counter = counter

        self.out_queue = JoinableQueue(max_size)

        self.in_queue = None
        self.runners = None
        self.runners_events = None

    def get_output(self):
        return self.out_queue

    def init_input_queue(self, func, max_size=0):
        self.in_queue = JoinableQueue(max_size)
        func(self.in_queue)

    def join(self, clear_in=False, clear_out=False):
        if self.mode == "consumer":
            self.in_queue.join()
        elif self.mode == "producer":
            if self.counter:
                while self.counter.value() != 0:
                    sleep(0.2)

        list(map(lambda event: event.set(), self.runners_events))
        if clear_in:
            while not self.in_queue.empty():
                self.in_queue.get_nowait()
        if clear_out:
            while not self.out_queue.empty():
                self.out_queue.get_nowait()

    def put_into_out_queue(self, values):
        to_put = {}
        for idx, value in enumerate(values):
            to_put[self.output_names[idx]] = value            
        self.out_queue.put(to_put)

    def set_input_queue(self, queue):
        self.in_queue = queue

    def start(self):
        if self.mode == "consumer":
            if not self.in_queue:
                raise RuntimeError("Please set input queue before start")

        self.runners_events = list(map(lambda x: Event(), 
                                            range(self.threads_num)))
        self.runners = list(map(lambda x: Process(target=self._run, 
                                                    args=(
                                                    self.runners_events[x],)), 
                                    range(self.threads_num)))
        list(map(lambda proc: proc.start(), self.runners))

    def _run(self, event):
        init_ret = {}
        if self.initializator:
            init_ret = self.initializator(**self.initializator_args)
            if not isinstance(init_ret, dict):
                raise RuntimeError("Initializator must return dictionary.")

        while not event.is_set():
            args = {} 
            if self.in_queue:
                res_left = self.resources_demanded                        
                while res_left > 0:
                    try:
                        resource = self.in_queue.get(timeout=0.3)
                    except Empty:
                        if event.is_set():
                            return
                        continue
                    if self.resources_demanded == 1:
                        args = resource
                        break

                    for key, value in resource.items():
                        if key in args.keys():
                            args[key].append(value)
                        else:
                            args[key] = [value]
                    res_left -= 1

            if self.initializator:
                for key, value in init_ret.items():
                    args[key] = value
            if self.update:
                while True:                
                    try:
                        current_value = self.out_queue.get(timeout=0.3)
                    except Empty:
                        if event.is_set():
                            return   
                        continue 
                    else:                
                        break
                
                for key, value in current_value.items():
                    args[key] = value

            ret = {}
            proc_error = False
            for function in self.functions:
                value = function(**args)
                if value is None:
                    proc_error = True
                    break
                if not isinstance(value, tuple):
                    value = (value, )
                for idx, name in enumerate(self.output_names):
                    if idx == len(value):
                        break
                    args[name] = value[idx]
                    ret[name] = args[name]

            if self.in_queue:            
                [self.in_queue.task_done() 
                    for x in range(self.resources_demanded)]
            if not proc_error:
                if self.counter:
                    if self.counter.value() == 0:
                        continue
                    else:                        
                        self.counter.decrement()
                while True:                                   
                    try:
                        self.out_queue.put(ret, timeout=0.3)
                    except Full:
                        if event.is_set():
                            return
                        continue
                    else:                        
                        break

            