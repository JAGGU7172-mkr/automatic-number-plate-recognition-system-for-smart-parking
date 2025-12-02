import os, glob
from anpr_utils import detect_license_plate

upload_folder = 'static/uploads'
files = glob.glob(os.path.join(upload_folder, '*'))
if not files:
    print('[ERROR] No uploaded files')
else:
    latest = max(files, key=os.path.getctime)
    print('[TEST FILE]', latest)
    plate, conf, path = detect_license_plate(latest)
    print('RESULT:', plate, conf, path)
