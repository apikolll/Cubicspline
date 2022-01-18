import numpy as np
from scipy.interpolate import CubicSpline
import csv
import matplotlib.pyplot as plt

#This Clamped Cubic Spline Interpolation program by Shah , Azrad and q

# calculate 5 natural cubic spline polynomials for 9 points
# (x,y) = (1,3.0) (2,3.7) (5,3.9) (6,4.2) (7,5.7) (8,6.6) (10,7.1) (13,6.7) (17,4.5)
x = np.array([1, 2, 5, 6, 7, 8, 10, 13, 17])
y = np.array([3.0, 3.7, 3.9, 4.2, 5.7, 6.6, 7.1, 6.7, 4.5])
yp = np.array([1.0, 0, 0, 0, 0, 0, 0, 0, -0.67])
aph = np.arange(9.0)
temp = np.arange(9)
h = np.arange(9.0)
l = np.arange(9.0)
m = np.arange(9.0)
z = np.arange(9.0)
c = np.arange(9.0)
b = np.arange(9.0)
d = np.arange(9.0)
a = np.arange(9.0)



#length of array
n = 8

#STEP 1
for i in range(n): 
    a=y
    temp[i] = x[i+1] - x[i]
    h[i] = temp[i]
    #print(h[i])

    
#STEP 2
aph[0] = (((a[1] - a[0])*3)/h[0]) - (3*yp[0])
aph[8] = 3*yp[8] - ((3*(a[8] - a[7]))/h[7])
formatted_float = "{:.2f}".format(aph[0])
formatted_float1 = "{:.2f}".format(aph[8])



#STEP 3


for i in range(1 , n):
    aph[i] = ((3/h[i])*(a[i+1] - a[i])) - ((3/h[i-1])*(a[i] - a[i-1]))
    formatted_float2 = "{:.2f}".format(aph[i])



#STEP 4

l[0] = 2*h[0]
m[0] = 0.5
z[0] = aph[0]/l[0]


#STEP 5

for i in range(1 , n):
    l[i] = (2*(x[i+1] - x[i-1])) - (h[i-1]*m[i-1])
    m[i] = h[i]/l[i]
    z[i] = (aph[i] - (h[i-1]*z[i-1]))/l[i]



#STEP 6

l[8] = h[7]*(2-m[7])
z[8] = (aph[8] - (h[7]*z[7]))/l[8]
c[8] = z[8]

    
#STEP 7

for i in range(n-1 , -1, -1):
    c[i] = z[i] - (m[i]*c[i+1])
    b[i] = ((a[i+1] - a[i])/h[i]) - ((h[i]*(c[i+1] + 2*c[i]))/3)
    d[i] = (c[i+1] - c[i])/(3*h[i])

#STEP 8

print("    Ai" , "\t\t" , "Bi" , "\t\t\t" , "Ci" , "\t\t\t" , "Di")
print("0 " , a[0] , "\t\t" , b[0] , "\t\t" , c[0] , "\t" , d[0])
for j in range(1 , n):
    print(j , "" , a[j] , "\t" , b[j] , "\t" , c[j] , "\t" , d[j])

#STEP 9

f = CubicSpline(x, y, bc_type='natural')
x_new = np.linspace(1, 17, 100)
y_new = f(x_new)

plt.figure(figsize = (20,10))
plt.plot(x_new, y_new, 'b')
plt.plot(x, y, 'ro')
plt.title('Cubic Spline Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()














