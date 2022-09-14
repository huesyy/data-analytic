import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots(X['Annual_Income_(k$)'], X['Spending_Score'])
ax.hist(arr, bins=20)

st.pyplot(fig)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots(X['Annual_Income_(k$)'], X['Spending_Score'], c=y_kmeans, s=50, cmap='viridis')
ax.hist(arr, bins=20)

st.pyplot(fig)
