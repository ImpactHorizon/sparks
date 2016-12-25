import io
import mimetypes
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from sparks.saver import Saver

class GDriveSaver(Saver):
    def __init__(self, settings_file="settings.yaml"):
        self.gauth = GoogleAuth(settings_file)
        self.gauth.ServiceAuth()
        self.drive = GoogleDrive(self.gauth) 

    def send_data(self, name, data):
        folders = name.split("/")[:-1]
        name = name.split("/")[-1]        
        current_parent = None
        for folder in folders:
            if folder == '':
                continue
            params = {"title" : folder, 
                        "mimeType": "application/vnd.google-apps.folder"}
            if current_parent:
                params["parents"] = [ {"id" : current_parent} ]
            file = self.drive.CreateFile(params)
            file.Upload()
            current_parent = file['id']

        params = {'title' : name}
        if current_parent:
            params["parents"] = [ {"id" : current_parent} ]
        file = self.drive.CreateFile(params)        
        file.content = io.BytesIO(data)
        file['mimeType'] = mimetypes.guess_type(name)[0]
        file.Upload()
