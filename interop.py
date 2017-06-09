import requests

class InterOp(object):
    """Send plane location

    """
    def __init__(self, username, password, ip, port):
        self.session = requests.session()
        self.ip = ip
        self.port = port
        payload = {'username': username, 'password': password}
        response = self.session.post("http://%s:%s/api/login"%(self.ip, self.port), data = payload, timeout = 5)
        print(response.text)

    def get_missions(self):
        missions = self.session.get('http://%s:%s/api/missions'%(self.ip, self.port))
        return missions.json()

    def get_obstacles(self):
        obstacles = self.session.get('http://%s:%s/api/obstacles'%(self.ip, self.port))
        return obstacles.json()

    def get_targets(self):
        targets = self.session.get('http://%s:%s/api/targets'%(self.ip, self.port))
        return targets.json()

    def delete_target(self, i):
        self.session.delete('http://%s:%s/api/targets/%d/image'%(self.ip, self.port, i))
        self.session.delete('http://%s:%s/api/targets/%d'%(self.ip, self.port, i))

    def send_coord(self, lat, lon, alt, heading):
        coord = {'latitude': lat, 'longitude': lon, 'altitude_msl': alt, 'uas_heading': heading}
        plane_location = self.session.post("http://%s:%s/api/telemetry"%(self.ip, self.port), data = coord)
        print(plane_location.text)


