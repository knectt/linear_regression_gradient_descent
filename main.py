import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

learning_rate = 0.00001
lines = np.array([[0, 0]])

#4 Gradient descent
def gradient_descent_runner(points, num_iterations):
    global lines
    w0 = w1 = 0

    for i in range(num_iterations):
        w0, w1 = step_gradient(w0, w1, np.array(points))
        lines = np.append(lines, [[w0, w1]], axis=0)

    return [w0, w1]

def step_gradient(w0, w1, points):
    w0_gradient = w1_gradient = 0
    N = float(len(points))

    #2 Unary linear regression function
    h = lambda theta_0,theta_1,x: theta_0 + theta_1*x

    for l in points:
        x = l[0]
        y = l[1]
        w0_gradient += -(2/N) * (y - h(w0,w1,x))
        w1_gradient += -(2/N) * (x * (y - h(w0,w1,x)))

    return [(w0 - (learning_rate * w0_gradient)), (w1 - (learning_rate * w1_gradient))]


#3 Least square method 
def least_square(x,y):
    ls_w1=((x*y).mean()-x.mean()*y.mean())/((x**2).mean()-(x.mean())**2)
    ls_w0=y.mean()-ls_w1*x.mean()
    print('least square method w0: '+str(ls_w0))
    print('least square method w1: '+str(ls_w1))

def run():
    # 1 The experimental data
    x=np.array([55, 71, 68, 87, 101, 87, 75, 78, 93, 73])
    y=np.array([91, 101, 87, 109, 129, 98, 95, 101, 104, 93])
    num_iterations = 1000
    points= np.stack((x,y),axis=1)
    least_square(x,y)

    print("Running...")
    [w0, w1] = gradient_descent_runner(points, num_iterations)
    print("After {0} iterations w0 = {1}, w1 = {2}".format(num_iterations, w0, w1))

    #5 The plot
    xarr = points[:,0]
    yarr = points[:,1]

    fig1 = plt.figure()
    plt.scatter(xarr, yarr)
    l, = plt.plot([], [], 'r-')
    plt.xlim(40, max(xarr)+10)
    plt.ylim(10, max(yarr)+10)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression - Gradient Descent')
    line_ani = animation.FuncAnimation(fig1, update_line, len(lines), fargs=(lines, xarr, l), interval=300, blit=True)

    plt.show()
    return

def update_line(num, lines, xarr, line):
    w0, w1 = lines[num]

    res = lambda e: w1*e + w0
    new_y = np.array([res(x) for x in xarr])

    line.set_data(xarr, new_y)
    return line,

if __name__ == '__main__':
    run()
