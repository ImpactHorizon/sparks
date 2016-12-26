from multiprocessing import Event, JoinableQueue, Process, Queue
from queue import Empty

class MultiProcessor(object):
    def __init__(self, 
                    functions, 
                    output_names=['val'], 
                    initializator=None, 
                    initializator_args={}, 
                    max_size=1024, 
                    threads_num=8, 
                    update=False,
                    resources_demanded=1):

        self.functions = functions
        self.output_names = output_names
        self.initializator = initializator
        self.initializator_args = initializator_args
        self.threads_num = threads_num
        self.update = update
        self.resources_demanded = resources_demanded

        self.out_queue = JoinableQueue(max_size)

        self.in_queue = None
        self.consumers = None
        self.consumers_events = None

    def get_output(self):
        return self.out_queue

    def init_input_queue(self, func, max_size=0):
        self.in_queue = JoinableQueue(max_size)
        func(self.in_queue)

    def join(self):
        self.in_queue.join()
        list(map(lambda event: event.set(), self.consumers_events))

    def put_into_out_queue(self, values):
        to_put = {}
        for idx, value in enumerate(values):
            to_put[self.output_names[idx]] = value            
        self.out_queue.put(to_put)

    def set_input_queue(self, queue):
        self.in_queue = queue

    def start(self):
        if not self.in_queue:
            raise RuntimeError("Please set input queue before start")

        self.consumers_events = list(map(lambda x: Event(), 
                                            range(self.threads_num)))
        self.consumers = list(map(lambda x: Process(target=self._consume, 
                                                    args=(
                                                    self.consumers_events[x],)), 
                                    range(self.threads_num)))
        list(map(lambda proc: proc.start(), self.consumers))

    def _consume(self, event):
        init_ret = {}
        if self.initializator:
            init_ret = self.initializator(**self.initializator_args)
            if not isinstance(init_ret, dict):
                raise RuntimeError("Initializator must return dictionary.")

        while not event.is_set():
            res_left = self.resources_demanded
            args = {}
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
                current_value = self.out_queue.get()
                for key, value in current_value.items():
                    args[key] = value

            ret = {}
            for function in self.functions:
                value = function(**args)
                if not isinstance(value, tuple):
                    value = (value, )
                for idx, name in enumerate(self.output_names):
                    if idx == len(value):
                        break
                    args[name] = value[idx]
                    ret[name] = args[name]
            
            self.out_queue.put(ret)
            [self.in_queue.task_done() for x in range(self.resources_demanded)]
            