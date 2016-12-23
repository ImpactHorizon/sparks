from azure.common import AzureException
from azure.storage.blob import BlobService
from sparks.reader import Reader

BLOB_RETRIES = 5

def progress(current, total):
    print ("Got %sMBs from %sMBs.") % (current/(1024*1024), total/(1024*1024))

class BlobReader(Reader):

    def __init__(self, account, key, container):
        self.block_blob_service = BlobService(account_name=account, 
                                                account_key=key)
        self.container = container

    def get_data(self, name):
        counter = BLOB_RETRIES
        while counter:
            try:
                data = self.block_blob_service.get_blob_to_bytes(self.container, 
                                                                    name)
            except AzureException as azure_exc:
                counter -= 1
            else:
                return data        
        raise RuntimeError("Couldn't read from blob, %s" % (azure_exc.args[0]))

    def to_file(self, handle, blobpath):
        counter = BLOB_RETRIES        
        while counter:
            try:
                self.block_blob_service.get_blob_to_file(
                                            self.container, 
                                            blobpath, 
                                            handle,
                                            max_connections=2,
                                            progress_callback=None)
            except AzureException as azure_exc:
                counter -= 1
            else:
                return
        raise RuntimeError("Couldn't download blob, %s" % (azure_exc.args[0]))

    def list(self, prefix):
        return self.block_blob_service.list_blobs(self.container, prefix)