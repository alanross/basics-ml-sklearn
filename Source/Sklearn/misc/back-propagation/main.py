import urllib2
from nn import NeuralNetwork


# http://stackoverflow.com/questions/23353635/python-neural-network-and-stock-prices-what-to-use-for-input

## ================================================================

def normalizePrice(price, minimum, maximum):
    return ((2 * price - (maximum + minimum)) / (maximum - minimum))


def denormalizePrice(price, minimum, maximum):
    return (((price * (maximum - minimum)) / 2) + (maximum + minimum)) / 2


## ================================================================

def rollingWindow(seq, windowSize):
    it = iter( seq )
    win = [ it.next( ) for cnt in xrange( windowSize ) ]  # First window
    yield win
    for e in it:  # Subsequent windows
        win[ :-1 ] = win[ 1: ]
        win[ -1 ] = e
        yield win


def getMovingAverage(values, windowSize):
    movingAverages = [ ]

    for w in rollingWindow( values, windowSize ):
        movingAverages.append( sum( w ) / len( w ) )

    return movingAverages


def getMinimums(values, windowSize):
    minimums = [ ]

    for w in rollingWindow( values, windowSize ):
        minimums.append( min( w ) )

    return minimums


def getMaximums(values, windowSize):
    maximums = [ ]

    for w in rollingWindow( values, windowSize ):
        maximums.append( max( w ) )

    return maximums


## ================================================================

def getTimeSeriesValues(values, window):
    movingAverages = getMovingAverage( values, window )
    minimums = getMinimums( values, window )
    maximums = getMaximums( values, window )

    returnData = [ ]

    # build items of the form [[average, minimum, maximum], normalized price]
    for i in range( 0, len( movingAverages ) ):
        inputNode = [ movingAverages[ i ], minimums[ i ], maximums[ i ] ]
        price = normalizePrice( values[ len( movingAverages ) - (i + 1) ], minimums[ i ], maximums[ i ] )
        outputNode = [ price ]
        tempItem = [ inputNode, outputNode ]
        returnData.append( tempItem )

    return returnData


## ================================================================

def getHistoricalData(stockSymbol):
    historicalPrices = [ ]

    # login to API
    urllib2.urlopen( "http://api.kibot.com/?action=login&user=guest&password=guest" )

    # get 14 days of data from API (business days only, could be < 10)
    url = "http://api.kibot.com/?action=history&symbol=" + stockSymbol + "&interval=daily&period=14&unadjusted=1&regularsession=1"
    apiData = urllib2.urlopen( url ).read( ).split( "\n" )
    for line in apiData:
        if (len( line ) > 0):
            tempLine = line.split( ',' )
            price = float( tempLine[ 1 ] )
            historicalPrices.append( price )

    return historicalPrices


## ================================================================

def getTrainingData(stockSymbol):
    historicalData = getHistoricalData( stockSymbol )

    # reverse it so we're using the most recent data first, ensure we only have 9 data points
    historicalData.reverse( )
    del historicalData[ 9: ]

    # get five 5-day moving averages, 5-day lows, and 5-day highs, associated with the closing price
    trainingData = getTimeSeriesValues( historicalData, 5 )

    return trainingData


## ================================================================

def runANDTest():
    data = [
        [ [ 0, 0 ], [ 0 ] ],
        [ [ 0, 1 ], [ 0 ] ],
        [ [ 1, 0 ], [ 0 ] ],
        [ [ 1, 1 ], [ 1 ] ]
    ]

    print "run ANT test"
    n = NeuralNetwork( 2, 2, 1 )
    n.train( data, 1000 )
    n.test( data )


def runXORTest():
    data = [
        [ [ 0, 0 ], [ 0 ] ],
        [ [ 0, 1 ], [ 1 ] ],
        [ [ 1, 0 ], [ 1 ] ],
        [ [ 1, 1 ], [ 0 ] ]
    ]

    print "run XOR test"
    n = NeuralNetwork( 2, 2, 1 )
    n.train( data, 1000 )
    n.test( data )


if __name__ == '__main__':
    runANDTest( )
