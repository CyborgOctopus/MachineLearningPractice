import numpy as np

# constants
num_blocks = 26
axes = [0, 1, 2]


# class that defines a 3 x 3 x 3 Rubik's cube
class RubiksCube:

    def __init__(self, transform=False):
        self.state = np.identity(num_blocks)
        if not transform:
            self.transforms = []
            self.generate_transforms()

    # defines equality of Rubik's cubes
    def __eq__(self, other):
        return np.all(self.state == other.state)

    # defines multiplication of Rubik's cube states
    def __mul__(self, other):
        product = RubiksCube()
        product.state = self.state.dot(other.state)
        return product

    # gets the current state of the Rubik's cube
    def get_state(self):
        return self.state

    # gets the transforms of the Rubik's cube
    def get_transforms(self):
        return self.transforms

    # applies a sequence of random transformations to the rubik's cube
    def scramble(self, length):
        new = RubiksCube().__mul__(self)
        for i in range(length):
            new = new.__mul__(np.random.choice(self.get_transforms()))
        return new

    # generates all transformations as RubiksCube objects
    def generate_transforms(self):
        for transform_state in self.transform_states():
            transform = RubiksCube(transform=True)
            transform.state = transform_state
            self.transforms.append(transform)

    # creates a list of the Rubik's cube transformation states
    def transform_states(self):
        transform_states = []
        for axis in axes:
            transform_states += self.transforms_along_axis(axis)
        return transform_states

    # gets the allowable rotations (in one direction) of the Rubik's cube components about a given axis.
    def transforms_along_axis(self, axis):
        rot_axes = axes.copy()
        rot_axes.remove(axis)
        padded_state = np.concatenate((self.state[0:13], [np.zeros(26)], self.state[13:26]))
        rot_state = np.rot90(padded_state.reshape((3, 3, 3, 26)), axes=rot_axes)
        #print(rot_state)
        #rot_state = np.concatenate((padded_rot_state[0:13], padded_rot_state[14:27]))
        top_rotation = np.concatenate(([np.rot90(rot_state[0])], rot_state[1:]))
        mid_rotation = np.concatenate(([rot_state[0]], [np.rot90(rot_state[1])], [rot_state[2]]))
        bottom_rotation = np.concatenate((rot_state[:2], [np.rot90(rot_state[2])]))
        transforms = [top_rotation, mid_rotation, bottom_rotation]
        transforms = [np.rot90(transform, axes=rot_axes[::-1]) for transform in transforms]
        transforms = [transform.reshape(27, 26) for transform in transforms]
        transforms = [np.concatenate((transform[0:13], transform[14:27])) for transform in transforms]
        return transforms
