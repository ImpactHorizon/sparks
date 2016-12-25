from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from sparks.reader import Reader

class GDriveReader():
    def __init__(self, settings_file="settings.yaml"):
        self.gauth = GoogleAuth(settings_file)
        self.gauth.ServiceAuth()
        self.drive = GoogleDrive(self.gauth)        

    def get_data(self, fileid):
        file = self.drive.CreateFile({'id' : fileid})
        try:
            return file.GetContentString()
        except:
            raise RuntimeError("Fetching file error.")

    def to_file(self, handle, fileid):
        file = self.drive.CreateFile({'id' : fileid})
        try:
            file.FetchContent()
        except:
            raise RuntimeError("Fetching file error.")
        handle.write(file.content.getvalue())        

    def list(self, query=None):
        if not query:
            query = "trashed=false"
    
        files = self.drive.ListFile({'q': query}).GetList()
        return list(map(lambda file: (file['id'], file['title']), files))