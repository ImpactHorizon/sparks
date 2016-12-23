from azure.common import AzureException
from azure.storage.blob import BlobService
import os
from sparks.saver import Saver

BLOB_RETRIES = 5

class BlobSaver(Saver):

    def __init__(self, account, key, container, prefix):
        self.block_blob_service = BlobService(account_name=account,
                                                account_key=key)
        self.container = container
        self.prefix = prefix
        self.block_blob_service.create_container(self.container)

    def send_data(self, name, data):
        counter = BLOB_RETRIES
        while counter:
            try:
                self.block_blob_service.put_block_blob_from_bytes(
                                                self.container, 
                                                os.path.join(self.prefix, name), 
                                                data)
            except AzureException as azure_exc:
                counter -= 1
            else:
                return
        raise RuntimeError("Couldn't send to blob." % (azure_exc.args[0]))