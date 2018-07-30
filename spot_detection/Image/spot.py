class Spot(object):

    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.name = '.'.join(map(str, coordinates))
        self.model = None
        self.region = None

    def __getitem__(self, item):
        return self.coordinates[item]

    def __repr__(self):
        msg = 'Spot at %s.' % self.name
        if self.model is not None:
            msg += '\nFit with: %s' % self.model.__repr__()
        else:
            return msg
        return msg
