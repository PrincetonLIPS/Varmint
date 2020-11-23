class DimensionError(Exception):
  ''' Base class to throw dimension-based errors. '''
  pass

class SplineError(Exception):
  ''' The thing you throw when the spline degree can't work. '''
  pass

class LabelError(Exception):
  ''' Throw these if labels aren't working correctly. '''
  pass

class UnitsError(Exception):
  ''' Throw if something seems surprising about a quantity with known units. '''
  pass

class MaterialError(Exception):
  ''' Throw if something weird happens with the material model. '''
