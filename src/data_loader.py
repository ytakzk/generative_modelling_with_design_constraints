import glob
import numpy as np

class Target():
    
    def __init__(self, grid):
        
        self.grid = grid
    
    @property
    def augmented(self):
        
        grid = self.grid
        
        if np.random.random() < 0.5:
            grid = np.swapaxes(grid, 0, 1)
        
        if np.random.random() < 0.5:
            grid = np.swapaxes(grid, 0, 2)
        
        if np.random.random() < 0.5:
            grid = np.swapaxes(grid, 1, 2)
        
        return Target(grid=grid)

def load_voxels(resolution=32, path='../voxelizer/32', furniture_list = ['table', 'bookshelf', 'chair', 'lamp'], num=400):
    
    # targets = []

    # voxel = np.zeros((resolution, resolution, resolution))

    # voxel[0:6, 0:50, 0:6] = 1
    # target = Target(grid=voxel)
    # targets.append(target)

    # return np.array(targets)

    voxels = []

    paths = []

    for f in furniture_list:
        
        p = '%s/%s/*' % (path, f)
        files = glob.glob(p)[:num]
        paths.extend(files)

    voxel_dic = {}
    average = []
        
    for i, path in enumerate(paths):
                    
        voxel = np.zeros((resolution, resolution, resolution))
        indexes = np.load(path)
        
        if len(indexes) < 5:
            print('FILE NOT FOUND')
            continue

        for index in indexes:

            voxel[index[0], index[1], index[2]] = 1   
            
        voxel = np.swapaxes(voxel, 0, 1)
        voxel = np.swapaxes(voxel, 0, 2)

        if voxel.sum() == 0:
            print('no voxel')
            continue
        
        average.append(voxel.sum())
        voxels.append(voxel)

    average = np.array(average)

    if len(average) == 0:
        raise ValueError('Data not found')

    print('average:', average.mean(), 'max:', average.max(), 'min:', average.min(), 'median:', np.median(average))

    targets    = []
    target_map = {}

    for i, v in enumerate(voxels):

        target = Target(grid=v)
        targets.append(target)
        target_map[i] = paths[i]

    return np.array(targets), target_map

def load_dummy(resolution=32, num=1):
    
    targets = []

    voxel = np.zeros((resolution, resolution, resolution))

    for i in range(num):
        voxel[0:6, 0:50, 0:6] = 1
        target = Target(grid=voxel)
        targets.append(target)

    return np.array(targets), {}