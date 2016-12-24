from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from sparks.reader import Reader

class GDriveReader():
    def __init__(self, settings_file="settings.yaml"):
        self.gauth = GoogleAuth(settings_file)
        self.gauth.ServiceAuth()
        self.drive = GoogleDrive(self.gauth)

    def get_data(self):
        pass

    def to_file(self):
        pass

    def list(self, query=None):
        if not query:
            query = "trashed=false"
    
        return self.drive.ListFile({'q': query}).GetList()

o = GDriveReader()
o=o.list("title contains '.tif' and trashed=false")
print(o[0]['fileSize'])
#o[0].GetContentFile("cat.tif")