
print('setting UP')
from utils import *
from sklearn.model_selection import train_test_split
### STEP 1
path = 'myData'
data = importDataInfo(path)

#### STEP 2
data = balanceData(data, display=False)

### STEP 3
imagesPath, steering = loadData(path,data)
#print(imagesPath[0], steering[0])

#### STEP 4
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath,steering,test_size=0.2, random_state=5)
print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))

#### STEP 5
# [----- Data Augmentation (in Utils.py)]

#### STEP 6
# [----- Data Preprocessing (in Utils.py)]

#### STEP 7
# [----- Batch Generation (in Utils.py)]

#### STEP 8
model = creatModel()
model.summary()

#### STEP 9
history = model.fit(batchGen(xTrain,yTrain,100,1),steps_per_epoch=300,epochs=10,
          validation_data=batchGen(xVal,yVal,100,0),validation_steps=200)

#### STEP 10
model.save('model.h5')
print('model saved')      

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.title('loss')
plt.xlabel('Epoch')
plt.show()

