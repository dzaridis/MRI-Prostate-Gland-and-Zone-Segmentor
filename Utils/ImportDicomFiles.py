#!/usr/bin/python

import os
import sys
import os.path
import httplib2
import base64

# if len(sys.argv) != 4 and len(sys.argv) != 6:
#     print("""
# Sample script to recursively import in Orthanc all the DICOM files
# that are stored in some path. Please make sure that Orthanc is running
# before starting this script. The files are uploaded through the REST
# API.

# Usage: %s [hostname] [HTTP port] [path]
# Usage: %s [hostname] [HTTP port] [path] [username] [password]
# For instance: %s localhost 8042 .
# """ % (sys.argv[0], sys.argv[0], sys.argv[0]))
#     exit(-1)

SUCCESS = 0
# This function will upload a single file to Orthanc through the REST API
def UploadFile(
        path:str,
        ip:str,
        port:int,
        username:str,
        password:str
    ):
    global SUCCESS

    if ip is None:
        ip = os.environ["ORTHANC_SERVICE_NAME"]

    if port is None:
        port = os.environ["PORT"]

    if username is None:
        username = os.environ["USERNAME"]

    if password is None:
        password = os.environ["PASSWORD"]

    URL = f'http://{ip}:{port}/instances'


    f = open(path, "rb")
    content = f.read()
    f.close()

    try:
        sys.stdout.write("Importing %s" % path)

        h = httplib2.Http()

        headers = { 'content-type' : 'application/dicom' }

        credentials = bytes(username + ':' + password, 'utf-8')
        encoded_credentials = base64.b64encode(credentials)
        headers['authorization'] = 'Basic ' + encoded_credentials.decode('utf-8') 
        
            
        resp, content = h.request(URL, 'POST', 
                                  body = content,
                                  headers = headers)

        if resp.status == 200:
            sys.stdout.write(" => success\n")
            SUCCESS += 1
        else:
            sys.stdout.write(" => failure (Is it a DICOM file?)\n")
    
    # Unknown exception (must be set)
    except:
        sys.stdout.write(" => unable to connect (Is Orthanc running? Is there a password?)\n")

def upload(
        ip:str=None,
        port:int=None,
        user:str=None,
        password:str=None,
        filename:str=None,
    )->None:
    ''' Upload files except if directory, uploads the dcm files inside directory, no recursive'''

    if ip is None:
        ip = os.environ["ORTHANC_SERVICE_NAME"]

    if port is None:
        port = os.environ["PORT"]

    if user is None:
        user = os.environ["USERNAME"]

    if password is None:
        password = os.environ["PASSWORD"]

    if os.path.isfile(filename):
        # Upload a single file
        UploadFile(filename,ip,port,user,password)
    else:
        # Recursively upload a directory
        for file in os.listdir(filename):
            if file.endswith(".dcm"):
                UploadFile(os.path.join(filename, file),ip,port,user,password)
            

    print("\nSummary: %d DICOM file(s) have been imported" % SUCCESS)
