import multiprocessing as mp
import numpy as np

# with mp.get_context("fork").Pool(8) as p:
#     p.map(f, range(10))


  

def idk(*args):
    # global shared_a
    return shared_a.sum(), shared_a.shape

def initializer(a, shape):
    # print(type(a))
    global shared_a
    print("new process")
    shared_a = np.asarray(a).reshape(shape)

def tester():
    # global shared_a
    shared_a[0,0] = -1

def main():
    shape = (1000, 1000)
    a = np.random.random(shape)
    # print(a.sum(), a.shape)
    shared_a_outer = mp.RawArray("d", a.flatten())
    shared_np = np.asarray(shared_a_outer).reshape(shape)
    print(shared_np[0, 0])
    with mp.Pool(8, initializer=initializer, initargs=(shared_a_outer, a.shape)) as p:
        res = list(p.map(idk, range(100)))
        p.apply(tester, args = tuple())
    # print(shared_a)
    print(shared_np[0, 0])
    # print(res)

if __name__ == '__main__':
    main()
    