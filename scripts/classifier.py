import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier



columns = [
	'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
	'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
	'veil-color',
	'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat' ]
le = LabelEncoder( )
dataset = pd.read_csv( 'agaricus-lepiota.data.csv', names=columns, index_col=None )
X = dataset.drop( 'class', axis=1 )
y = le.fit_transform( dataset[ 'class' ].values )

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=.3 )
X_test = pd.get_dummies( X_test[ columns[ 1: ] ] )
X_train = pd.get_dummies( X_train[ columns[ 1: ] ] )
cf = SGDClassifier( )
cf.fit( X_train.values, y_train )
errors = 0
for (x, y) in zip( X_test.values, y_test ):
	predict = cf.predict( x.reshape( 98, ).reshape( 1, -1 ) )
	if (predict[ 0 ] != y):
		errors += 1

if errors == 0:
	print( 'accuracy 100%' )
if errors > 0:
	print( 'accuracy: ' + str( 1694 / errors * 100 ) )
print( 'errors count: ' + str( errors ) )
