class Reader(object):

    def __init__(self):
        pass

    def get_data(self):
        raise NotImplemented("Method get_data is not implemented for Reader " + 
                                "object.")

    def to_file(self):
        raise NotImplemented("Method to_file is not implemented for Reader " + 
                                "object.")

    def list(self):
        raise NotImplemented("Method list is not implemented for Reader " + 
                                "object.")