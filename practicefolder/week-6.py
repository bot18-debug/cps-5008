import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

email = ['A','B','C','D','E','F','G','H'] 

x1 = np.array([8,7,1,2,6,3,5,2])
x2 = np.array([3,3,8,7,4,7,5,9])

# Combine features
x = np.column_stack((x1, x2))

labels = ['Spam','Spam','Not Spam','Not Spam','Spam','Not Spam','Spam','Not Spam']
y = np.array([1 if label == 'Spam' else 0 for label in labels])

# Train SVM
svm = SVC(kernel='linear')
svm.fit(x, y)

w = svm.coef_[0]
b = svm.intercept_[0]

# Assign colors
colors = ['red' if label == 'Spam' else 'green' for label in labels]

plt.figure(figsize=(8,6))
plt.scatter(x1, x2, c=colors, s=100)

# Label each point
for i in range(len(email)):
    plt.text(x1[i] + 0.1, x2[i] + 0.1, email[i])

# Decision boundary
x_vals = np.linspace(0, 10, 100)
y_vals = -(w[0]/w[1])*x_vals - b/w[1]
plt.plot(x_vals, y_vals, 'b--', label='SVM Decision Boundary')

# Support vectors
plt.scatter(
    svm.support_vectors_[:,0],
    svm.support_vectors_[:,1],
    s=200,
    facecolors='none',
    edgecolors='blue',
    label='Support Vectors'
)

plt.xlabel('Suspicious Words (x₁)')
plt.ylabel('Email Length (x₂)')
plt.title('Linear SVM: Maximising the Margin')
plt.legend()
plt.grid(True)
plt.show()