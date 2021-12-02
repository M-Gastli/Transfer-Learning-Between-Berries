#%%
import matplotlib.pyplot as plt
import numpy as np

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
hists = np.load('lagged_hists_35rasp.npy')
# restore np.load for future normal usage
np.load = np_load_old
print(hists.shape)

# Separate histograms from yields
lagged_hists = []
for i in range(len(hists)):
    lagged_hists.append(hists[i,0])
lagged_hists = np.array(lagged_hists)
#lagged_hists = np.delete(lagged_hists, np.arange(7), axis=3)
print(lagged_hists.shape)
''
lagged_hists = np.delete(lagged_hists, [0,1,2,3,4,5,6],3)
print(lagged_hists.shape)
''

lagged_yields = []
for i in range(len(hists)):
    lagged_yields.append(hists[i,1])
lagged_yields = np.array(lagged_yields)
print(lagged_yields.shape)

# Reshape
lagged_hists = np.transpose(lagged_hists, [0,2,1,3])
lagged_hists = np.reshape(lagged_hists,[lagged_hists.shape[0],-1,lagged_hists.shape[2]*lagged_hists.shape[3]])
print('Reshaped:', lagged_hists.shape)

split = int(0.8 * len(lagged_hists))
hists_train = lagged_hists[:split]
yields_train = lagged_yields[:split]
hists_val = lagged_hists[split:]
yields_val = lagged_yields[split:]
print('Train:', hists_train.shape, yields_train.shape)
print('Validate:', hists_val.shape, yields_val.shape)
#%%
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
tf.keras.backend.clear_session()
from numpy.random import seed
seed(1)
tf.random.set_seed(2)

model = models.load_model(r'C:\Users\MohamedSadok\Documents\UofWStuff\Research Projects\Img. Proc. Cali\Common Experiment NSR Models/best_ensemblensr351.hdf5')
pred = model.predict(hists_val).flatten()
RMSE_CNN = np.sqrt(np.mean((pred - yields_val)**2))
MAE_CNN = np.mean(np.abs(pred - yields_val))
r2_CNN = r2_score(yields_val, pred)
agm_CNN = ((RMSE_CNN + MAE_CNN)/2)*(1-r2_CNN)
plt.plot(yields_val, label='True yield');
plt.plot(pred, label='Predicted yield');
plt.legend(loc='upper right'); plt.grid()
plt.xlabel('Days'); plt.ylabel('Yield')
plt.title('Strawberry Model on Raspberry Data')
print(MAE_CNN)
print(RMSE_CNN)
print(r2_CNN)
print(agm_CNN)

#%%
import time
#for layer in model.layers[:5]:
#            layer.trainable = False

#model.pop()
#model.add(layers.Dense(units=512, activation='relu'))
#model.add(layers.Dense(units=1, activation='relu'))

print(model.summary())
start = time.time()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss = 'mean_absolute_error'
model.compile(optimizer=optimizer, loss = loss)
mcp_save = ModelCheckpoint('best_s2y_unfrozen35rasp.hdf5', save_best_only=True, monitor='val_loss', mode='min')
history = model.fit(hists_train, yields_train, validation_data=(hists_val, yields_val), epochs=20\
                    , batch_size=32, callbacks=[mcp_save], verbose=2)

print('Training time:',time.time() - start,' seconds')
model = models.load_model('best_s2y_unfrozen35rasp.hdf5')
pred = model.predict(hists_val).flatten()

RMSE_CNN = np.sqrt(np.mean((pred - yields_val)**2))
MAE_CNN = np.mean(np.abs(pred - yields_val))
r2_CNN = r2_score(yields_val, pred)
agm_CNN = ((RMSE_CNN + MAE_CNN)/2)*(1-r2_CNN)
plt.plot(yields_val, label='True yield');
plt.plot(pred, label='Predicted yield');
plt.legend(loc='upper right'); plt.grid()
plt.xlabel('Days'); plt.ylabel('Yield')
plt.title('Frozen Base Model on Raspberry Data')
print(MAE_CNN)
print(RMSE_CNN)
print(r2_CNN)
print(agm_CNN)
# %%
n=2
pred = []
for j in range(n):
    model = models.load_model('TL Models/best_s2y_frozen35rasp'+str(j)+'.hdf5')
    pred.append(model.predict(hists_val).flatten())
    RMSE_CNN = np.sqrt(np.mean((pred[j] - yields_val)**2))
    MAE_CNN = np.mean(np.abs(pred[j] - yields_val))
    r2_CNN = r2_score(yields_val, pred[j])
    agm_CNN = ((RMSE_CNN + MAE_CNN)/2)*(1-r2_CNN)
    print ("AGM:",agm_CNN)

avg_preds = np.mean(pred,axis=0)

RMSE_CNN = np.sqrt(np.mean((avg_preds - yields_val)**2))
MAE_CNN = np.mean(np.abs(avg_preds - yields_val))
r2_CNN = r2_score(yields_val, avg_preds)
agm_CNN = ((RMSE_CNN + MAE_CNN)/2)*(1-r2_CNN)
print ("MAE of CNN:",MAE_CNN)
print ("RMSE of CNN:", RMSE_CNN)
print ("R2 score of CNN:",r2_CNN)
print ("AGM score of CNN:",agm_CNN)

plt.plot(yields_val, label='True Values');
plt.plot(avg_preds, label='Predicted Values');
plt.legend(); plt.grid()
plt.xlabel('Days');
plt.ylabel('Yield');
plt.show()
# %%

model = models.load_model('Old Models/best_s2y_unfrozen35rasp.hdf5')
pred = model.predict(hists_val).flatten()
RMSE_CNN = np.sqrt(np.mean((pred - yields_val)**2))
MAE_CNN = np.mean(np.abs(pred - yields_val))
r2_CNN = r2_score(yields_val, pred)
agm_CNN = ((RMSE_CNN + MAE_CNN)/2)*(1-r2_CNN)
print ("AGM:",agm_CNN)
plt.plot(yields_val, label='True Values');
plt.plot(pred, label='Predicted Values');
plt.legend(); plt.grid()
plt.title('Unfrozen Base Model on Raspberry Data')
plt.xlabel('Days');
plt.ylabel('Yield');
plt.show()
# %%
yields_Mohita = lagged_yields[2543-175:3106-175]
hists_Mohita = lagged_hists[2543-175:3106-175]
n=2
pred = []
for j in range(n):
    model = models.load_model('Old TL Models/best_s2y_frozen35rasp'+str(j)+'.hdf5')
    pred.append(model.predict(hists_val).flatten())
avg_preds = np.mean(pred,axis=0)
np.savetxt('Raspberry_yield_frozen.txt', avg_preds)

# %%
