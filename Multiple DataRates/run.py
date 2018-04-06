
# coding: utf-8

# In[38]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')

for x in range(1,19):
	data = pd.read_csv('./d'+str(x)+'.csv',index_col=0)
	filter_by_col = data[data['dropped']=='COL'].groupby('node')
	time = filter_by_col['time'].max() - filter_by_col['time'].min()
	collision_rate = filter_by_col.size()/time
	collision_rate = pd.DataFrame(collision_rate,columns=['Collision_rate'])
	collision_rate[time == 0] = 0
	collision_rate.fillna(0)

	collision_rate = collision_rate.ix[0:,:]
	#collision_rate.to_csv('c1.csv')

	df = data 

	df[df["rts_check"] == "RTS"]
	received = df[df["rts_check"] == 'RTS']
	received = received[['time','node']]
	group_node = df.groupby('node')

	received_packets = group_node.size()
	time = group_node['time'].max() - group_node['time'].min()
	received_rate = received_packets/time
	received_rate[time == 0] = 0
	received_rate.fillna(0)

	received_rate = pd.DataFrame(received_rate)
	received_rate = received_rate.rename(columns = {0:'Received Rate'})
	received_rate = received_rate.ix[0:,:]
	#received_rate.to_csv('r1.csv')

	data = df

	# calculate probability


	df1 = pd.DataFrame(data)
	df1['dos'] = 0
	df1.loc[df1['from']==1,'dos'] = 1
	proba = pd.DataFrame(df1.groupby('node')['dos'].mean())
	proba = proba.ix[0:,:]
	proba.columns = ["Probability"]
	#df2.to_csv('p1.csv')


	# In[39]:


	df1 = collision_rate
	df2 = received_rate
	df3 = proba


	# In[41]:


	df = pd.concat([df1,df2,df3],axis = 1)

	df.drop(1,inplace=True)
	df.to_csv('fd'+str(x)+'.csv',index = 'node')

