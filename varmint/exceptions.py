class DimensionError(Exception):
  ''' Base class to throw dimension-based errors. '''
  pass

class SplineError(Exception):
  ''' The thing you throw when the spline degree can't work. '''
  pass
